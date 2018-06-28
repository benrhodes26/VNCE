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

from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from utils import plot_log_likelihood_training_curves, take_closest, make_nce_minus_vnce_loss_plot, get_nce_loss_for_vnce_params, save_fig

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

# For reproducibility
rng = rnd.RandomState(1083463236)

exp_name = 'usps-sgd/'  # CHANGE ME
model_name = 'rbm'  # CHANGE ME
skip_files = []

data_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data/usps/'
exp_res_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/' + model_name
save_res_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/experiments/' + model_name
exp_dir = os.path.join(exp_res_dir, exp_name)
save_dir = os.path.join(save_res_dir, exp_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def annotate(axis, annotation, xy, a_length=100):
    xy_text = xy[0] + 0.1, xy[1] + 0.1
    axis.annotate(annotation, xy=xy, xycoords='data', xytext=xy_text,
                  arrowprops=dict(facecolor='black', width=0.05, headwidth=0.05, shrink=0.05), horizontalalignment='right',
                  verticalalignment='bottom', fontsize=5)

def plot_and_annotate(axis, times, vals, annotation, annotate_val, a_length=100):
    axis.semilogx(times, vals)
    ind = take_closest(vals, annotate_val)
    annotate(axis, annotation, (times[ind], vals[ind]), a_length)
    axis.annotate(r"{}".format(round(vals[-1], 2)), xy=(times[-1], vals[-1]), fontsize=5)


"""
==========================================================================================================
                                            Log likelihood plot
==========================================================================================================
"""

fig, axs = plt.subplots(2, 2, figsize=(5.7, 5))
axs = axs.ravel()
for i, file in enumerate(os.listdir(exp_dir)):
    exp = os.path.join(exp_dir, file)

    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))

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

    rng = np.random.RandomState(config['random_seed'])
    true_log_like_exists = config['which_dataset'] == 'usps_3by3patches'
    if true_log_like_exists:
        emp_dist = pickle.load(open(os.path.join(data_dir, 'usps_3by3_emp_dist.p'), 'rb'))
        emp_log_like = np.mean(np.log(emp_dist(X_test)))

    if config['noise'] == 'marginal':
        noise = MultivariateBernoulliNoise(np.mean(X, axis=0))
        noise_log_like = np.mean(np.log(noise(X)))
    else:
        print('unexpected noise dist')
        raise TypeError

    # specify arrow position and lengths for annotations
    arrow_y_val_1 = noise_log_like + ((av_log_like_cd.max() + 0.5 - noise_log_like)/2)  # log-like plot vnce
    arrow_y_val_2 = noise_log_like + ((av_log_like_cd.max() + 0.5 - noise_log_like)/2)  # log-like plot cd
    arrow_length = 1

    # LOG-LIKELIHOOD PLOT
    ax = axs[i]
    ax.set_title(file)
    plot_and_annotate(ax, reduced_vnce_times, av_log_like_vnce, "VNCE", arrow_y_val_1, a_length=arrow_length)
    plot_and_annotate(ax, reduced_cd_times, av_log_like_cd, "CD {}".format(config['cd_num_steps']), arrow_y_val_2, a_length=arrow_length)
    plot_and_annotate(ax, reduced_nce_times, av_log_like_nce, "NCE", arrow_y_val_1-0.1, a_length=arrow_length+1)

    # PLOT NOISE AND OPTIMAL LOG-LIKELIHOODS AS HORIZONTAL LINES
    ax.plot((0, plt.get(ax, 'xlim')[1]), (noise_log_like, noise_log_like), 'g--', label='Noise distribution')
    if true_log_like_exists:
        ax.plot((0, plt.get(ax, 'xlim')[1]), (emp_log_like, emp_log_like), 'b--', label='Empirical distribution')

    # TIDY UP AND LABEL THE PLOT
    ax.set_xlim(1e-1, axs[i].get_xlim()[1])
    if true_log_like_exists:
        ax.set_ylim((noise_log_like-0.05, emp_log_like + 0.05))
    else:
        ax.set_ylim((noise_log_like-0.05, av_log_like_cd.max() + 0.2))

    ax.set_xlabel('time (seconds)', fontsize=10)
    ax.set_ylabel('log likelihood', fontsize=10)
    ax.legend(loc='lower right')

    with open(os.path.join(save_dir, "config-{}.txt".format(file)), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))

fig.tight_layout()
save_fig(fig, save_dir, 'usps-different_num_hiddens')

"""
==========================================================================================================
                                            J/J1 diff plot
==========================================================================================================
"""

# for i, file in enumerate(os.listdir(exp_dir)):
#     exp = os.path.join(exp_dir, file)
#
#     # LOAD EVERYTHING FROM FILE
#     config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
#
#     globals().update(np.load(os.path.join(exp, 'data.npz')))
#     globals().update(np.load(os.path.join(exp, 'init_theta_and_likelihood.npz')))
#     globals().update(np.load(os.path.join(exp, 'vnce_results.npz')))
#     globals().update(np.load(os.path.join(exp, 'nce_results.npz')))
#
#     J_plot = make_nce_minus_vnce_loss_plot(nce_losses_for_vnce_params, vnce_losses, vnce_times, e_step_ids)
#     axs[0].set_xlim(1e-1, 100)
#     axs[0].set_ylim((-0.2, 0.2))
#
#     J_plot.suptitle('{}'.format(file))
#     J_plot.set_size_inches(15, 15)
#
#     J_plot.savefig(os.path.join(save_dir, 'two-terms-of-J1-{}.pdf'.format(file)))
