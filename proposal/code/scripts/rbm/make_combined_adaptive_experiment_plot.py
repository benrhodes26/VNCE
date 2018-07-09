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
from utils import plot_log_likelihood_learning_curves, take_closest, plot_vnce_loss, save_fig

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

train_fig, train_axs = plt.subplots(3, 2, figsize=(5.7, 8))
test_fig, test_axs = plt.subplots(3, 2, figsize=(5.7, 8))
train_axs, test_axs = train_axs.ravel(), test_axs.ravel()

for i, file in enumerate(sorted_files):
    file = str(file)
    exp = os.path.join(load_dir, file)
    globals().update(np.load(os.path.join(exp, 'data.npz')))
    globals().update(np.load(os.path.join(exp, 'init_theta_and_likelihood.npz')))
    globals().update(np.load(os.path.join(exp, 'vnce_results.npz')))
    globals().update(np.load(os.path.join(exp, 'cd_results.npz')))
    globals().update(np.load(os.path.join(exp, 'nce_results.npz')))
    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))

    # to avoid log(0) and inconsistencies in starting points
    reduced_vnce_times[0] += 1e-4
    reduced_cd_times[0] += 1e-4
    reduced_nce_times[0] += 1e-4
    av_log_like_vnce_train[0] = init_log_like
    av_log_like_cd_train[0] = init_log_like
    av_log_like_nce_train[0] = init_log_like

    train_ax, test_ax = train_axs[i], test_axs[i]

    train_curves = [[reduced_vnce_times, av_log_like_vnce_train, 'vnce', 'blue'], [reduced_cd_times, av_log_like_cd_train, 'cd', 'red']]
    test_curves = [[reduced_vnce_times, av_log_like_vnce_test, 'vnce', 'blue'], [reduced_cd_times, av_log_like_cd_test, 'cd', 'red']]

    x_lim = (1, reduced_vnce_times[-1] + 100)
    title = 'learning rate: {}'.format(config['cd_learn_rate'])
    plot_log_likelihood_learning_curves(train_curves, [], save_dir, x_lim=x_lim, file_name='train', title=title, logx=True, ax=train_ax)
    plot_log_likelihood_learning_curves(test_curves, [], save_dir, x_lim=x_lim, file_name='test', title=title, logx=True, ax=test_ax)

    with open(os.path.join(save_dir, "config-{}.txt".format(file)), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))

train_fig.tight_layout()
save_fig(train_fig, save_dir, 'train_learning_curves')
test_fig.tight_layout()
save_fig(test_fig, save_dir, 'test_learning_curves')
