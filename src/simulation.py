import time
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from scipy.stats import expon
from predictorGAN import PredictorGAN
from entrypointGAN import EntryPointGAN
from util.parse_utils import BIWIParser

from util.debug_utils import Logger
from util.roi import RegionOfInterest


def fit_exp(start_ts):
    start_ts = np.sort(np.array(start_ts))
    intervals = start_ts[1:] - start_ts[:-1]
    # intervals = np.sort(intervals)
    # print(intervals)
    exp_loc, exp_scale = expon.fit(intervals)

    return 1./exp_scale


def get_starting_ts(exp_rate, n):
    random = expon.rvs(size=n, loc=0, scale=1./exp_rate)
    random = np.cumsum(random)
    return random

    # hist, _ = np.histogram(random, range=(0, random.max()), bins=50)
    # n, bins, patches = plt.hist(x=hist, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    # plt.show()


# ============= LOAD DATASET =============
config_file = '../config/config.yaml'
stream = open(config_file)
conf = yaml.load(stream, Loader=yaml.FullLoader)
annotation_file = conf['Dataset']['Annotation']
down_sample = conf['Dataset']['DownSample']
max_way_points = conf['Generation']['MaxWayPoints']
max_observation_length = conf['Generation']['MaxObservationLength']
parser = BIWIParser(interval_=down_sample)
parser.load(annotation_file)
parser.scale.calc_scale(keep_ratio=False, zero_based=False)

t_data = parser.t_data
first_frames = np.array([t_data_i[0] for t_data_i in t_data])
nPed = len(parser.p_data)
n_train_peds = nPed * 4 // 5

train_entry_points = []
train_path_points = []
test_entry_points = []
test_path_points = []

for ii, Pi in enumerate(parser.p_data):
    Pi = parser.scale.normalize(Pi)
    Vi = Pi[1:] - Pi[:-1]
    for tt in range(len(Pi) - 2):
        x_t = np.hstack((Pi[tt], Vi[tt]))
        x_t_1 = np.hstack((Vi[tt+1], Pi[tt+2]))
        if ii < n_train_peds:
            train_path_points.append(torch.FloatTensor(np.hstack((x_t, x_t_1))))
        else:
            test_path_points.append(torch.FloatTensor(np.hstack((x_t, x_t_1))))

    if ii < n_train_peds:
        train_entry_points.append(torch.FloatTensor(np.hstack((Pi[0], Vi[0]))))
    else:
        test_entry_points.append(torch.FloatTensor(np.hstack((Pi[0], Vi[0]))))

train_path_points = torch.stack(train_path_points, 0)
train_entry_points = torch.stack(train_entry_points, 0)
test_path_points = torch.stack(test_path_points, 0)
test_entry_points = torch.stack(test_entry_points, 0)

# =========== Read ROI ============
ROI_vertices = np.loadtxt(conf['Dataset']['ROI'])
ROI_vertices = parser.scale.normalize(ROI_vertices)
ROI = RegionOfInterest(ROI_vertices)
# =================================

# Simulate
np.random.seed(0)
torch.manual_seed(0)

gan_entry_point = EntryPointGAN(conf)
gan_predictor = PredictorGAN(conf)

gan_entry_point.load_dataset(parser)
gan_entry_point.load_model(conf['EntryPointGAN']['Checkpoint'])

gan_predictor.load_dataset(parser, max_observation_length)
gan_predictor.load_model(conf['PredictorGAN']['Checkpoint'])


def simulate(N):
    tic = time.clock()
    entry_pnts = gan_entry_point.generate(N)
    entry_pnts = entry_pnts.view((-1, 2, 2))
    entry_pnts[:, 1] = entry_pnts[:, 1] + entry_pnts[:, 0]  # calc second loc of the trajectories

    # Reject invalid entry_pnts
    entry_pnts_fltr = []
    for ep in entry_pnts:
        if not ROI.contains(ep[0]):
            entry_pnts_fltr.append(ep)
    if len(entry_pnts_fltr) == 0:
        print('Error! all the entry points are invalid')
        return []
    entry_pnts_fltr = torch.stack(entry_pnts_fltr)

    tic = time.clock()
    trajectories = gan_predictor.generate(entry_pnts_fltr, 1, max_way_points, max_observation_length)
    toc = time.clock()
    # for traj in trajectories:
    #     plt.plot(traj[:, 0], traj[:, 1])
    #     plt.plot(traj[0, 0], traj[0, 1], 'ro')
    # plt.show()

    lens = [len(tr_i) for tr_i in trajectories]
    avg_len = np.mean(lens)
    print('Generating Full Trajectories with avg length of %.1f took %.3f ms' % (avg_len, (toc - tic)*1000))

    # Reject invalid trajectories
    trajectories_fltr = []
    for ii, traj_i in enumerate(trajectories):
        in_roi_flag = False
        for tt in range(2, len(traj_i)):
            # trajectories_fltr = np.stack(trajectories_fltr)
            if ROI.contains(traj_i[tt]):
                #if (ROI[0] < traj_i[tt, 0] < ROI[2] and ROI[1] < traj_i[tt, 1] < ROI[3]):
                in_roi_flag = True
            elif in_roi_flag:
                # if ROI[1] < traj_i[tt, 1] < ROI[3]:
                #     break  # it has collision
                trajectories_fltr.append(traj_i[:tt + 1])
                break
    # trajectories_fltr = trajectories_fltr[:min(N, len(trajectories_fltr))]
    return trajectories_fltr


def write_to_file(trajectories, filename):
    sim_entries = []
    for ii, traj_i in enumerate(trajectories):
        for tt in range(len(traj_i)):
            frame_id = starting_frames[ii] + tt * down_sample
            xy = parser.scale.denormalize(traj_i[tt][0:2] / 2 + 0.5)
            sim_entries.append(np.array([frame_id, ii, xy[0], xy[1]]))

    sim_entries = np.stack(sim_entries)

    with open(filename, 'w') as f:
        print('writing the simulation into ', filename)
        f.write('MATRIX 1 0 0 0 0 1 0 1 0\n')
        f.write('FPS 10\n')
        for entry in sim_entries:
            f.write('%d %d %.5f 0 %.5f\n' % (entry[0], entry[1], entry[2], entry[3]))


for kk in range(200):
    enter_rate = fit_exp(first_frames / down_sample)
    starting_frames = get_starting_ts(enter_rate * 20, n=1000).astype(np.int32) * down_sample
    fps = 10
    n_frames = 2 * fps * 60
    for ii in range(len(starting_frames)):
        if starting_frames[ii] > n_frames:
            starting_frames = starting_frames[:ii]
            break

    # nSimPed = len(starting_frames)
    nSimPed = 100
    trajectories = simulate(nSimPed)
    if len(trajectories) == 0: continue
    # write_to_file(trajectories, '../variety/2ped_%d.txt' % kk)

    # ============= PLOT =================
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)

    for traj_i in trajectories:
        start_label, = plt.plot(traj_i[0, 0], traj_i[0, 1], 'b.', alpha=0.4)
        traj_label, = plt.plot(traj_i[:, 0], traj_i[:, 1], 'r', LineWidth=2, alpha=0.4)
        end_label, = plt.plot(traj_i[-1, 0], traj_i[-1, 1], 'mx')

    ax.add_patch(patches.Polygon(ROI.vertices, closed=True,
                 fill=False,
                 zorder=2, linestyle='--' ))

    # ax.add_patch(patches.Polygon(ROI_Polygon, closed=True, fill=False, zorder=2, linestyle='-.'))

    plt.xlim([-1.3, 1.3])
    plt.ylim([-1.3, 1.3])

    plt.legend((start_label, end_label, traj_label), ('Entry Point', 'Exit Point', 'Generated Trajectory'))
    plt.show()

    fig_file = '../variety/2ped_' + str(kk) + '_' + str(nSimPed) + '.png'
    print('Writing figure to ', fig_file)
    plt.savefig(fig_file)
    plt.clf()
    plt.close()

