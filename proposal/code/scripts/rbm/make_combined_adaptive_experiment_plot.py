import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

import numpy as np
import pickle

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
import matplotlib as matplotlib
from utils import plot_log_likelihood_training_curves, take_closest, plot_vnce_loss, save_fig

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=8)
rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project/ben-rhodes-masters-project/proposal/experiments/rbm/adaptive/adaptive_finetuned_combined/')
parser.add_argument('--exp_name', type=str, default=None, help='name of set of experiments this one belongs to')
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/rbm/adaptive/adaptive_finetuned_combined/')

args = parser.parse_args()
load_dir = os.path.join(args.load_dir, args.exp_name)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

"""
==========================================================================================================
                            Combined log likelihood plot for multiple experiments
==========================================================================================================
"""

sorted_files = sorted([float(learning_rate) for learning_rate in os.listdir(load_dir)])

fig, axs = plt.subplots(3, 2, figsize=(5.7, 8))
axs = axs.ravel()
for i, file in enumerate(sorted_files):
    file = str(file)
    exp = os.path.join(load_dir, file)
    globals().update(np.load(os.path.join(exp, 'data.npz')))
    globals().update(np.load(os.path.join(exp, 'init_theta_and_likelihood.npz')))
    globals().update(np.load(os.path.join(exp, 'vnce_results.npz')))
    globals().update(np.load(os.path.join(exp, 'cd_results.npz')))
    globals().update(np.load(os.path.join(exp, 'nce_results.npz')))

    # to avoid log(0) and inconsistencies in starting points
    reduced_vnce_times[0] += 1e-4
    reduced_cd_times[0] += 1e-4
    reduced_nce_times[0] += 1e-4
    av_log_like_vnce[0] = init_log_like
    av_log_like_cd[0] = init_log_like
    av_log_like_nce[0] = init_log_like

    # LOG-LIKELIHOOD PLOT
    ax = axs[i]
    ax.semilogx(reduced_vnce_times, av_log_like_vnce, label='AVNCE', color='blue')
    ax.annotate(r"{}".format(round(av_log_like_vnce[-1], 2)), xy=(reduced_vnce_times[-1], av_log_like_vnce[-1] + 1), fontsize=5, color='blue')

    ax.semilogx(reduced_cd_times, av_log_like_cd, label='CD', color='red')
    ax.annotate(r"{}".format(round(av_log_like_cd[-1], 2)), xy=(reduced_cd_times[-1], av_log_like_cd[-1] - 1), fontsize=5, color='red')

    # ax.semilogx(reduced_nce_times, av_log_like_nce, label='NCE')
    # ax.annotate(r"{}".format(round(av_log_like_nce[-1], 2)), xy=(reduced_nce_times[-1], av_log_like_nce[-1]), fontsize=5)

    # Set axes limits and add labels
    ax.set_xlim(1, axs[i].get_xlim()[1] + 100)
    y_max = max(av_log_like_cd.max(), av_log_like_vnce.max())
    ax.set_ylim(top=y_max + 5)

    ax.set_title('learning rate: {}'.format(file[-4:]))
    ax.set_xlabel('time (seconds)', fontsize=10)
    ax.set_ylabel('log likelihood', fontsize=10)
    ax.legend(loc='lower right')

    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
    with open(os.path.join(save_dir, "config-{}.txt".format(file)), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))

fig.tight_layout()
save_fig(fig, save_dir, 'av_log_like')

