import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
from util.parse_utils import BIWIParser
from util.helper import bce_loss
from util.debug_utils import Logger


class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_size, lstm_size, hidden_size, relu_slope):
        super(Generator, self).__init__()

        # Embedding
        embed_layers = [nn.Linear(4, embedding_size[0]), nn.LeakyReLU(relu_slope)]
        for ii in range(1, len(embedding_size)):
            embed_layers.extend([nn.Linear(embedding_size[ii-1], embedding_size[ii]), nn.LeakyReLU(relu_slope)])
        self.embedding = nn.Sequential(*embed_layers)

        # LSTM
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_size[-1], lstm_size, num_layers=1, batch_first=True)

        # Decoder
        fc_layers = [nn.Linear(lstm_size + noise_dim, hidden_size[0]), nn.LeakyReLU(relu_slope)]
        for ii in range(1, len(hidden_size)):
            fc_layers.extend([nn.Linear(hidden_size[ii-1], hidden_size[ii]), nn.LeakyReLU(relu_slope)])
        fc_layers.append(nn.Linear(hidden_size[-1], 2))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x_in, noise, x_lengths):
        bs = noise.size(0)
        last_indices = [[i for i in range(bs)], (np.array(x_lengths) - 1)]

        # calc velocities and concat to x_in
        x_in_vel = x_in[:, 1:] - x_in[:, :-1]
        x_in_vel = torch.cat((x_in_vel, torch.zeros((bs, 1, 2), device=x_in.device, dtype=x_in.dtype)), dim=1)
        last_indices_1 = [[i for i in range(bs)], (np.array(x_lengths) - 2)]
        x_in_vel[last_indices] = x_in_vel[last_indices_1]
        x_in_aug = torch.cat([x_in, x_in_vel], dim=2)

        e_in = self.embedding(x_in_aug)

        h_init, c_init = (torch.zeros((1, bs, self.lstm_size), device=noise.device) for _ in range(2))
        lstm_out, (h_out, c_out) = self.lstm(e_in, (h_init, c_init))
        lstm_out_last = lstm_out[last_indices]

        hid_vector = torch.cat((lstm_out_last, noise), dim=1)
        x_out = self.fc(hid_vector) + x_in[last_indices]
        return x_out


class Discriminator(nn.Module):
    def __init__(self,  embedding_size, lstm_size, hidden_size, relu_slope):
        super(Discriminator, self).__init__()

        # Embedding
        embed_layers = [nn.Linear(2, embedding_size[0]), nn.LeakyReLU(relu_slope), nn.BatchNorm1d(embedding_size[0])]
        for ii in range(1, len(embedding_size)):
            embed_layers.extend([nn.Linear(embedding_size[ii - 1], embedding_size[ii]), nn.LeakyReLU(relu_slope)])
        self.embedding = nn.Sequential(*embed_layers)

        # LSTM
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_size[-1], lstm_size, num_layers=1, batch_first=True)

        # Classifier
        fc_layers = [nn.Linear(lstm_size + embedding_size[-1], hidden_size[0]), nn.LeakyReLU(relu_slope)]
        for ii in range(1, len(hidden_size)):
            fc_layers.extend([nn.Linear(hidden_size[ii - 1], hidden_size[ii]), nn.LeakyReLU(relu_slope)])
        fc_layers.append(nn.Linear(hidden_size[-1], 1))
        self.fc = nn.Sequential(*fc_layers)

    def load_backup(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, x_in, x_out, x_lengths):
        bs = x_in.size(0)
        T = x_in.size(1)
        e_in = self.embedding(x_in.view(-1, 2)).view(bs, T, -1)
        e_out = self.embedding(x_out)

        h_init, c_init = (torch.zeros(1, bs, self.lstm_size, device=x_in.device) for _ in range(2))
        lstm_out, (h_out, c_out) = self.lstm(e_in, (h_init, c_init))
        inds = [[i for i in range(bs)], (np.array(x_lengths) - 1)]
        lstm_out_last = lstm_out[inds]

        hid_vector = torch.cat((lstm_out_last, e_out), dim=1)
        x_out = self.fc(hid_vector)
        return x_out


class PredictorGAN:
    def __init__(self, config):

        # Hyper-params
        self.noise_dim = config['PredictorGAN']['NoiseDim']
        self.unrolling_steps = config['PredictorGAN']['UnrolleingSteps']
        self.n_epochs = config['PredictorGAN']['nEpochs']
        self.checkpoints = config['PredictorGAN']['Checkpoint']

        hidden_size_G = config['PredictorGAN']['HiddenSizeG']
        hidden_size_D = config['PredictorGAN']['HiddenSizeD']

        embedding_size = config['PredictorGAN']['EmbeddingSize']
        lstm_size_G = config['PredictorGAN']['LstmSizeG']
        lstm_size_D = config['PredictorGAN']['LstmSizeD']
        relu_slope = config['PredictorGAN']['ReluSlope']

        lr_G = config['PredictorGAN']['LearningRateG']
        lr_D = config['PredictorGAN']['LearningRateD']
        beta1 = config['PredictorGAN']['beta1']
        beta2 = config['PredictorGAN']['beta2']

        self.batch_size = config['PredictorGAN']['BatchSize']
        self.use_l2_loss = config['PredictorGAN']['UseL2Loss']

        self.G = Generator(self.noise_dim, embedding_size, lstm_size_G, hidden_size_G, relu_slope).cuda()
        self.D = Discriminator(embedding_size, lstm_size_D, hidden_size_D, relu_slope).cuda()
        self.G_optimizer = opt.Adam(self.G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.D_optimizer = opt.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCELoss()

        self.data = {'obsvs': [], 'obsv_lengths': [], 'preds': []}
        self.n_data = 0
        self.n_batches = 0
        self.test_data_init = []

    def load_dataset(self, parser, max_obsv_len):
        nPed = len(parser.p_data)

        continuation_data_obsv = []
        obsv_lengths = []
        continuation_data_pred = []
        entry_point_data = []
        for ii, Pi in enumerate(parser.p_data):
            Pi = parser.scale.normalize(Pi)
            for tt in range(2, len(Pi) - 1):
                x_obsv_t = Pi[max(0, tt-max_obsv_len):tt]
                obsv_lengths.append(len(x_obsv_t))
                continuation_data_obsv.append(torch.FloatTensor(x_obsv_t))
                continuation_data_pred.append(torch.FloatTensor(Pi[tt]))
            entry_point_data.append(torch.FloatTensor(Pi[0:2]))

        continuation_data_pred = torch.stack(continuation_data_pred, dim=0).cuda()
        continuation_data_obsv = pad_sequence(continuation_data_obsv, batch_first=True).cuda()

        self.n_data = len(continuation_data_pred)
        self.n_batches = int(np.ceil(self.n_data / self.batch_size))
        bs = self.batch_size
        for bi in range(self.n_batches):
            self.data['obsvs'].append(continuation_data_obsv[bi * bs:min((bi + 1) * bs, self.n_data)])
            self.data['obsv_lengths'].append(obsv_lengths[bi * bs:min((bi + 1) * bs, self.n_data)])
            self.data['preds'].append(continuation_data_pred[bi * bs:min((bi + 1) * bs, self.n_data)])
        self.data['entry_points'] = entry_point_data

    def save(self, checkpoints, epoch):
        logger.print_me('Saving model to ', checkpoints)
        torch.save({
            'epoch': epoch,
            'G_dict': self.G.state_dict(),
            'D_dict': self.D.state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict()
        }, checkpoints)

    def load_model(self, checkpoints=''):
        if not checkpoints:
            checkpoints = self.checkpoints
        epoch = 1
        if os.path.isfile(checkpoints):
            print('loading from ' + checkpoints)
            checkpoint = torch.load(checkpoints)
            epoch = checkpoint['epoch'] + 1
            self.G.load_state_dict(checkpoint['G_dict'])
            self.D.load_state_dict(checkpoint['D_dict'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        return epoch

    def batch_train(self, obsvs, preds, obsv_lengths):
        bs = len(preds)
        self.D_optimizer.zero_grad()
        zeros = Variable(torch.zeros(bs, 1) + np.random.uniform(0, 0.05), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * np.random.uniform(0.95, 1.0), requires_grad=False).cuda()
        noise = Variable(torch.FloatTensor(torch.rand(bs, self.noise_dim)), requires_grad=False).cuda()  # uniform

        for u in range(self.unrolling_steps + 1):
            with torch.no_grad():
                preds_fake = self.G(obsvs, noise, obsv_lengths)

            fake_labels = self.D(obsvs, preds_fake, obsv_lengths)
            d_loss_fake = bce_loss(fake_labels, zeros)

            real_labels = self.D(obsvs, preds, obsv_lengths)  # classify real samples
            d_loss_real = bce_loss(real_labels, ones)
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()  # update D
            self.D_optimizer.step()

            if u == 0 and self.unrolling_steps > 0:
                backup = copy.deepcopy(self.D)

        # =============== Train Generator ================= #
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        preds_fake = self.G(obsvs, noise, obsv_lengths)

        fake_labels = self.D(obsvs, preds_fake, obsv_lengths)
        g_loss_fooling = bce_loss(fake_labels, ones)
        g_loss = g_loss_fooling

        mse_loss = torch.empty(1)
        if self.use_l2_loss:
            mse_loss = F.mse_loss(preds_fake, preds)
            g_loss += 100 * mse_loss

        g_loss.backward()
        self.G_optimizer.step()

        if self.unrolling_steps > 0:
            self.D.load_backup(backup)
            del backup

        return g_loss.item(), d_loss.item(), mse_loss.item()

    def train(self):
        start_epoch = self.load_model(self.checkpoints)

        # TODO: separate train and test
        nTrain = self.n_batches * 4 // 5
        nTest = self.n_batches - nTrain

        for epoch in range(start_epoch, self.n_epochs):
            g_loss, d_loss, mse_loss = 0, 0, 0

            tic = time.clock()
            for ii in range(nTrain):
                g_loss_ii, d_loss_ii, mse_loss_ii = self.batch_train(self.data['obsvs'][ii],
                                                                     self.data['preds'][ii],
                                                                     self.data['obsv_lengths'][ii])
                g_loss += g_loss_ii
                d_loss += d_loss_ii
                mse_loss += mse_loss_ii
            toc = time.clock()

            if epoch % 50 == 0:  # FIXME : set the interval for running tests
                logger.print_me('[%4d] MSE= %.5f, Loss G= %.3f, Loss D= %.3f, time=%.3f s'
                    % (epoch, mse_loss, g_loss,       d_loss,       toc - tic))
                self.save(self.checkpoints, epoch)

    def generate(self, entry_points, n_samples, n_step, max_obsv_len):
        x_in = [entry_points.clone() for _ in range(n_samples)]
        x_in = torch.cat(x_in, dim=0)
        N = len(x_in)
        generated_trajecs = np.zeros((N, n_step, 2), np.float32)
        generated_trajecs[:, 0:2, :] = x_in.data.cpu().numpy()

        for tt in range(2, n_step):
            obsv_len = min(tt, max_obsv_len)
            x_lengths = [obsv_len for _ in range(N)]
            noise = Variable(torch.FloatTensor(torch.rand(N, self.noise_dim)), requires_grad=False).cuda()
            x_in_recent = x_in[:, tt-obsv_len:tt]
            x_out = self.G(x_in_recent, noise, x_lengths)
            x_in = torch.cat([x_in, x_out.unsqueeze(1)], dim=1)  # new vel
            generated_trajecs[:, tt, :] = x_out.data.cpu().numpy()
        return generated_trajecs


if __name__ == '__main__':
    # Read config file
    config_file = '../config/config.yaml'
    stream = open(config_file)
    conf = yaml.load(stream, Loader=yaml.FullLoader)
    annotation_file = conf['Dataset']['Annotation']
    down_sample = conf['Dataset']['DownSample']
    max_observation_length = conf['Generation']['MaxObservationLength']
    parser = BIWIParser(interval_=down_sample)
    parser.load(annotation_file)
    parser.scale.calc_scale(keep_ratio=False, zero_based=False)
    logger = Logger(conf['Debug']['LogFile'])

    gan = PredictorGAN(conf)
    gan.load_dataset(parser, max_observation_length)
    gan.load_model()
    gan.train()
