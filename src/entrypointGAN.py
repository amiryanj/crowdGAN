import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import torch.nn as nn
import torch.optim as opt
from torch import FloatTensor
from torch.autograd import Variable
from util.helper import bce_loss, l1_loss
from util.parse_utils import BIWIParser
from util.debug_utils import Logger


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_size, relu_slope):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(noise_dim, hidden_size[0]), nn.LeakyReLU(relu_slope),
                                # nn.BatchNorm1d(hidden_size[0]),
                                nn.Linear(hidden_size[0], hidden_size[1]), nn.LeakyReLU(relu_slope),
                                # nn.BatchNorm1d(hidden_size[1]),
                                nn.Linear(hidden_size[1], hidden_size[2]), nn.LeakyReLU(relu_slope),
                                # nn.BatchNorm1d(hidden_size[2]),
                                nn.Linear(hidden_size[2], 4)
                                )

    def forward(self, noise):
        y = self.fc(noise)
        return y   # return first loc & vel


class Discriminator(nn.Module):
    def __init__(self, hidden_size, relu_slope):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(4, hidden_size[0]), nn.LeakyReLU(relu_slope),
                                nn.BatchNorm1d(hidden_size[0]),
                                nn.Linear(hidden_size[0], hidden_size[1]), nn.LeakyReLU(relu_slope),
                                # nn.BatchNorm1d(hidden_size[1]),
                                nn.Linear(hidden_size[1], hidden_size[2]), nn.LeakyReLU(relu_slope),
                                # nn.BatchNorm1d(hidden_size[2]),
                                nn.Linear(hidden_size[2], 1))

    def forward(self, x):
        y = self.fc(x)
        return y

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


class EntryPointGAN:
    def __init__(self, config):
        # Hyper-params
        self.noise_dim = config['EntryPointGAN']['NoiseDim']
        self.unrolling_steps = config['EntryPointGAN']['UnrolleingSteps']
        self.n_epochs = config['EntryPointGAN']['nEpochs']
        self.checkpoints = config['EntryPointGAN']['Checkpoint']

        hidden_size_G = config['EntryPointGAN']['HiddenSizeG']
        hidden_size_D = config['EntryPointGAN']['HiddenSizeD']
        relu_slope = config['EntryPointGAN']['ReluSlope']
        lr_G = config['EntryPointGAN']['LearningRateG']
        lr_D = config['EntryPointGAN']['LearningRateD']
        beta1 = config['EntryPointGAN']['beta1']
        beta2 = config['EntryPointGAN']['beta2']

        self.G = Generator(self.noise_dim, hidden_size_G, relu_slope).cuda()
        self.D = Discriminator(hidden_size_D, relu_slope).cuda()
        self.G_optimizer = opt.Adam(self.G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.D_optimizer = opt.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, beta2))

        self.train_samples = []

    def load_model(self, checkpoints=''):
        if not checkpoints:
            checkpoints = self.checkpoints
        epoch = 1
        if os.path.exists(checkpoints):
            print('loading from ' + checkpoints)
            checkpoint = torch.load(checkpoints)
            epoch = checkpoint['epoch'] + 1
            self.G.load_state_dict(checkpoint['G_dict'])
            self.D.load_state_dict(checkpoint['D_dict'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        return epoch

    def save_model(self, checkpoints, epoch):
        print('Saving model to file ...', checkpoints)
        torch.save({
            'epoch': epoch,
            'G_dict': self.G.state_dict(),
            'D_dict': self.D.state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict()
        }, checkpoints)

    def set_data(self, samples):
        self.train_samples = samples

    def load_dataset(self, parser):
        entry_points = []
        for ii in range(len(parser.p_data)):
            _pi = parser.scale.normalize(parser.p_data[ii])
            _vi = _pi[1] - _pi[0]
            _len_i = len(_pi)
            _pi = _pi[0, :]
            entry_points.append(FloatTensor(np.hstack((_pi, _vi))))
        self.train_samples = torch.stack(entry_points, 0).cuda()

    def train_step(self, real_points):
        bs = len(real_points)
        zeros = Variable(torch.zeros(bs, 1) + np.random.uniform(0, 0.1), requires_grad=False).cuda()
        ones = Variable(torch.ones(bs, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).cuda()
        noise_U = Variable(FloatTensor(torch.rand(bs, self.noise_dim)), requires_grad=False).cuda()

        d_acc_fake = 0; d_err_real = 0; g_loss_tot = 0

        # ================ Train D ===================== #
        with torch.no_grad():
            fake_pnts = self.G(noise_U)

        for u in range(self.unrolling_steps + 1):
            fake_labels = self.D(fake_pnts)
            d_loss_fake = l1_loss(fake_labels, zeros)

            real_labels = self.D(real_points)  # classify real samples
            d_loss_real = l1_loss(real_labels, ones)
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()  # update D
            self.D_optimizer.step()
            self.D_optimizer.zero_grad()

            d_acc_real = real_labels.mean().item() * 100
            d_acc_fake = (1-fake_labels.mean().item()) * 100

            if u == 0 and self.unrolling_steps > 0:
                backup = copy.deepcopy(self.D)

        # ================ Train G ===================== #
        fake_pnts = self.G(noise_U)
        fake_labels = self.D(fake_pnts)
        g_loss_fooling = l1_loss(fake_labels, ones)

        g_loss = g_loss_fooling
        g_loss_tot += g_loss.item()
        g_loss.backward()
        self.G_optimizer.step()
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        if self.unrolling_steps > 0:
            self.D.load(backup)
            del backup

        return g_loss_tot, d_acc_real, d_acc_fake

    # =============  TRAIN  ================
    def train(self):
        start_epoch = self.load_model(self.checkpoints)

        for epoch in range(start_epoch, self.n_epochs):
            tic = time.clock()
            g_loss_tot, d_acc_real, d_acc_fake = self.train_step(self.train_samples)
            toc = time.clock()
            if epoch % 100 == 0:  # FIXME : set the interval for running tests
                self.save_model(self.checkpoints, epoch)
                print('%d] Loss G = %.2f | D(Real) = %.1f | D(Fake) = %.1f | time=%.4f'
                    % (epoch, g_loss_tot, d_acc_real, d_acc_fake, toc - tic))

            if epoch % 100 == 0:
                # draw and save
                with torch.no_grad():
                    fake_pnts = self.generate()
                fig, ax = plt.subplots()
                for ii, traj_i in enumerate(fake_pnts):
                    traj_i = traj_i.data.cpu().numpy()
                    # plt.plot(traj_i[:fake_lens[ii], 0], traj_i[:fake_lens[ii], 1])
                    plt.plot(traj_i[0], traj_i[1], 'r.')
                    ax.arrow(traj_i[0], traj_i[1],
                             traj_i[2], traj_i[3], head_width=0.003, head_length=0.004, fc='k', ec='m', alpha=0.3)
                # plt.show()
                plt.xlim([-1.5, 1.5])
                plt.ylim([-1.5, 1.5])
                plt.savefig('../training-results/entry-points-gan/' + str(epoch) + '.png')
                plt.clf()
                plt.close()

    def generate(self, n=200):
        noise_U = Variable(FloatTensor(torch.rand(n, self.noise_dim)), requires_grad=False).cuda()
        fake_pnts = self.G(noise_U)
        return fake_pnts


if __name__ == '__main__':
    # Read config file
    config_file = '../config/config.yaml'
    stream = open(config_file)
    conf = yaml.load(stream, Loader=yaml.FullLoader)
    annotation_file = conf['Dataset']['Annotation']

    parser = BIWIParser()
    parser.load(annotation_file)
    parser.scale.calc_scale(keep_ratio=False, zero_based=False)

    gan = EntryPointGAN(conf)
    gan.load_dataset(parser)
    gan.load_model()
    gan.train()
