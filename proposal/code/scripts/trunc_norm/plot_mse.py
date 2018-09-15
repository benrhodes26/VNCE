import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_3 = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dir_4 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dirs = [code_dir, code_dir_2, code_dir_3, code_dir_4]
for code_dir in code_dirs:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import numpy as np
import pickle
import seaborn as sns

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
import matplotlib as matplotlib
from plot import *
from utils import take_closest, mean_square_error

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd

rc('lines', linewidth=0.5)
rc('font', size=8)
rc('legend', fontsize=9)
rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='plot relationship between fraction of training data missing and final mean-squared error for'
                                    'a truncated normal model trained with VNCE', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project-non-code/experiments/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='test/', help='name of set of experiments this one belongs to')  # 5d-vlr0.1-nz=10-final
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm/')
# parser.add_argument('--load_dir', type=str, default='~/masters-project-non-code/experimental-results/trunc-norm/')

args = parser.parse_args()
main_load_dir = os.path.join(args.load_dir, args.exp_name)
main_load_dir = os.path.expanduser(main_load_dir)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def split_params(theta):
    mean = theta[1:1+d]
    prec_flat = theta[1+d:]
    prec = np.zeros((d, d))
    prec[np.tril_indices(d)] = prec_flat
    prec[np.diag_indices(d)] = np.exp(prec[np.diag_indices(d)])
    prec_diag = deepcopy(prec[np.diag_indices(d)])
    prec_non_diag = deepcopy(prec[np.tril_indices(d, -1)])
    prec += prec.T
    prec[np.diag_indices(d)] -= prec_diag

    cov = np.linalg.inv(prec)
    cov_diag = cov[np.diag_indices(d)]
    cov_n_diag = cov[np.tril_indices(d, -1)]

    return mean, prec_diag, prec_non_diag, prec, cov_diag, cov_n_diag, cov


def get_mses(true_theta, theta, d):
    true_mean, true_prec_diag, true_prec_n_diag, true_prec, true_cov_diag, true_cov_n_diag, true_cov = split_params(true_theta)
    mean, prec_diag, prec_n_diag, prec, cov_diag, cov_n_diag, cov = split_params(theta)

    true_b = np.dot(true_prec, true_mean)
    b = np.dot(prec, mean)

    #mean_mse = mean_square_error(true_mean, mean)
    mean_b = mean_square_error(true_b, b)
    diag_prec_mse = mean_square_error(true_prec_diag, prec_diag)
    n_diag_prec_mse = mean_square_error(true_prec_n_diag, prec_n_diag)
    # diag_prec_mse = mean_square_error(true_cov_diag, cov_diag)
    # n_diag_prec_mse = mean_square_error(true_cov_n_diag, cov_n_diag)
    # diag_prec_mse = mean_square_error(np.ones_like(A_diag), A_diag)
    # n_diag_prec_mse = mean_square_error(np.ones_like(A_ndiag), A_ndiag)

    return mean_b, diag_prec_mse, n_diag_prec_mse


all_vnce1_means_mses = []
all_vnce2_means_mses = []
all_nce1_means_mses = []
all_nce2_means_mses = []
all_nce3_means_mses = []

all_vnce1_diag_mses = []
all_vnce2_diag_mses = []
all_nce1_diag_mses = []
all_nce2_diag_mses = []
all_nce3_diag_mses = []

all_vnce1_ndiag_mses = []
all_vnce2_ndiag_mses = []
all_nce1_ndiag_mses = []
all_nce2_ndiag_mses = []
all_nce3_ndiag_mses = []
for outer_file in os.listdir(main_load_dir):
    load_dir = os.path.join(main_load_dir, outer_file, 'best')

    frac_to_mean_mse_dict = {}
    frac_to_prec_diag_mse_dict = {}
    frac_to_n_prec_diag_mse_dict = {}

    # frac_to_rel_mse_dict = {}
    # loop through files and get mses of vnce and nce with 0s
    sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
    sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]
    for i, file in enumerate(sorted_dirs):
        exp = os.path.join(load_dir, file)
        config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
        frac = float(config.frac_missing)
        d = config.d

        loaded = np.load(os.path.join(exp, 'theta0_and_theta_true.npz'))
        loaded1 = np.load(os.path.join(exp, 'vnce_results1.npz'))
        loaded2 = np.load(os.path.join(exp, 'vnce_results2.npz'))
        loaded3 = np.load(os.path.join(exp, 'nce_results1.npz'))
        loaded4 = np.load(os.path.join(exp, 'nce_results2.npz'))
        loaded5 = np.load(os.path.join(exp, 'nce_results3.npz'))
        true_theta = loaded['theta_true']
        vnce_theta1 = loaded1['vnce_thetas'][-1][-1]
        vnce_theta2 = loaded2['vnce_thetas'][-1][-1]
        nce_means_theta = loaded3['nce_thetas'][-1]
        nce_noise_theta = loaded4['nce_thetas'][-1]
        nce_rnd_theta = loaded5['nce_thetas'][-1]

        vnce_mean_mse1, vnce_diag_prec_mse1, vnce_n_diag_prec_mse1 = get_mses(true_theta, vnce_theta1, d)
        vnce_mean_mse2, vnce_diag_prec_mse2, vnce_n_diag_prec_mse2 = get_mses(true_theta, vnce_theta2, d)
        nce1_mean_mse, nce1_diag_prec_mse, nce1_n_diag_prec_mse = get_mses(true_theta, nce_means_theta, d)
        nce2_mean_mse, nce2_diag_prec_mse, nce2_n_diag_prec_mse = get_mses(true_theta, nce_noise_theta, d)
        nce3_mean_mse, nce3_diag_prec_mse, nce3_n_diag_prec_mse = get_mses(true_theta, nce_rnd_theta, d)

        frac_to_mean_mse_dict[str(frac)] = [vnce_mean_mse1, vnce_mean_mse2, nce1_mean_mse, nce2_mean_mse, nce3_mean_mse]
        frac_to_prec_diag_mse_dict[str(frac)] = [vnce_diag_prec_mse1, vnce_diag_prec_mse2, nce1_diag_prec_mse, nce2_diag_prec_mse, nce3_diag_prec_mse]
        frac_to_n_prec_diag_mse_dict[str(frac)] = [vnce_n_diag_prec_mse1, vnce_n_diag_prec_mse2, nce1_n_diag_prec_mse, nce2_n_diag_prec_mse, nce3_n_diag_prec_mse]

        # frac_to_rel_mse_dict[str(frac)] = [1, nce_missing_mse / vnce_mse, nce_missing_mse_2 / vnce_mse]

    fracs = sorted([float(key) for key in frac_to_mean_mse_dict.keys()])

    all_vnce1_means_mses.append([frac_to_mean_mse_dict[str(frac)][0] for frac in fracs])
    all_vnce2_means_mses.append([frac_to_mean_mse_dict[str(frac)][1] for frac in fracs])
    all_nce1_means_mses.append([frac_to_mean_mse_dict[str(frac)][2] for frac in fracs])
    all_nce2_means_mses.append([frac_to_mean_mse_dict[str(frac)][3] for frac in fracs])
    all_nce3_means_mses.append([frac_to_mean_mse_dict[str(frac)][4] for frac in fracs])

    all_vnce1_diag_mses.append([frac_to_prec_diag_mse_dict[str(frac)][0] for frac in fracs])
    all_vnce2_diag_mses.append([frac_to_prec_diag_mse_dict[str(frac)][1] for frac in fracs])
    all_nce1_diag_mses.append([frac_to_prec_diag_mse_dict[str(frac)][2] for frac in fracs])
    all_nce2_diag_mses.append([frac_to_prec_diag_mse_dict[str(frac)][3] for frac in fracs])
    all_nce3_diag_mses.append([frac_to_prec_diag_mse_dict[str(frac)][4] for frac in fracs])

    all_vnce1_ndiag_mses.append([frac_to_n_prec_diag_mse_dict[str(frac)][0] for frac in fracs])
    all_vnce2_ndiag_mses.append([frac_to_n_prec_diag_mse_dict[str(frac)][1] for frac in fracs])
    all_nce1_ndiag_mses.append([frac_to_n_prec_diag_mse_dict[str(frac)][2] for frac in fracs])
    all_nce2_ndiag_mses.append([frac_to_n_prec_diag_mse_dict[str(frac)][3] for frac in fracs])
    all_nce3_ndiag_mses.append([frac_to_n_prec_diag_mse_dict[str(frac)][4] for frac in fracs])

# rel_vnce_mses = [frac_to_rel_mse_dict[str(frac)][0] for frac in fracs]
# rel_nce_missing_mses = [frac_to_rel_mse_dict[str(frac)][1] for frac in fracs]
# rel_nce_missing_mses_2 = [frac_to_rel_mse_dict[str(frac)][2] for frac in fracs]


def plot_errorbar(ax, fracs, mses, deciles, method, color):
    median = np.median(np.array(mses), 0)
    ax.errorbar(fracs, median, yerr=[median - deciles[0], deciles[1] - median], fmt='o',
                markersize=1, linestyle='None', label=method, color=color, capsize=1, capthick=1)


sns.set_style("darkgrid")
fig, axs = plt.subplots(3, 1, figsize=(5.7, 8.5))
axs = axs.ravel()
fracs = np.array(fracs)
fracs1 = fracs - 0.015
fracs2 = fracs - 0.0075
fracs3 = fracs
fracs4 = fracs + 0.0075
fracs5 = fracs + 0.015
percentiles = [25, 75]

ax = axs[0]
vnce1_means_deciles = np.percentile(all_vnce1_means_mses, percentiles, axis=0)
vnce2_means_deciles = np.percentile(all_vnce2_means_mses, percentiles, axis=0)
nce1_means_deciles = np.percentile(all_nce1_means_mses, percentiles, axis=0)
nce2_means_deciles = np.percentile(all_nce2_means_mses, percentiles, axis=0)
nce3_means_deciles = np.percentile(all_nce3_means_mses, percentiles, axis=0)

plot_errorbar(ax, fracs1, all_vnce1_means_mses, vnce1_means_deciles, 'VNCE (true)', 'black')
plot_errorbar(ax, fracs2, all_vnce2_means_mses, vnce2_means_deciles, 'VNCE (approx)', 'blue')
plot_errorbar(ax, fracs3, all_nce1_means_mses, nce1_means_deciles, 'NCE (means fill in)', 'orange')
plot_errorbar(ax, fracs4, all_nce2_means_mses, nce2_means_deciles, 'NCE (noise fill in)', 'green')
plot_errorbar(ax, fracs5, all_nce3_means_mses, nce3_means_deciles, 'NCE (random fill in)', 'red')

ax.set_ylim(0, 2)
ax.set_title(r'\textbf{b}')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
ax.legend(loc='upper left')

ax = axs[1]
vnce1_diag_deciles = np.percentile(all_vnce1_diag_mses, percentiles, axis=0)
vnce2_diag_deciles = np.percentile(all_vnce2_diag_mses, percentiles, axis=0)
nce1_diag_deciles = np.percentile(all_nce1_diag_mses, percentiles, axis=0)
nce2_diag_deciles = np.percentile(all_nce2_diag_mses, percentiles, axis=0)
nce3_diag_deciles = np.percentile(all_nce3_diag_mses, percentiles, axis=0)

plot_errorbar(ax, fracs1, all_vnce1_diag_mses, vnce1_diag_deciles, 'VNCE (true)', 'black')
plot_errorbar(ax, fracs2, all_vnce2_diag_mses, vnce2_diag_deciles, 'VNCE (approx)', 'blue')
plot_errorbar(ax, fracs3, all_nce1_diag_mses, nce1_diag_deciles, 'NCE (means fill in)', 'orange')
plot_errorbar(ax, fracs4, all_nce2_diag_mses, nce2_diag_deciles, 'NCE (noise fill in)', 'green')
plot_errorbar(ax, fracs5, all_nce3_diag_mses, nce3_diag_deciles, 'NCE (random fill in)', 'red')

ax.set_ylim(0, 0.125)
ax.set_title('Diagonal of K')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
ax.legend(loc='upper left')

ax = axs[2]
vnce1_ndiag_deciles = np.percentile(all_vnce1_ndiag_mses, percentiles, axis=0)
vnce2_ndiag_deciles = np.percentile(all_vnce2_ndiag_mses, percentiles, axis=0)
nce1_ndiag_deciles = np.percentile(all_nce1_ndiag_mses, percentiles, axis=0)
nce2_ndiag_deciles = np.percentile(all_nce2_ndiag_mses, percentiles, axis=0)
nce3_ndiag_deciles = np.percentile(all_nce3_ndiag_mses, percentiles, axis=0)

plot_errorbar(ax, fracs1, all_vnce1_ndiag_mses, vnce1_ndiag_deciles, 'VNCE (true)', 'black')
plot_errorbar(ax, fracs2, all_vnce2_ndiag_mses, vnce2_ndiag_deciles, 'VNCE (approx)', 'blue')
plot_errorbar(ax, fracs3, all_nce1_ndiag_mses, nce1_ndiag_deciles, 'NCE (means fill in)', 'orange')
plot_errorbar(ax, fracs4, all_nce2_ndiag_mses, nce2_ndiag_deciles, 'NCE (noise fill in)', 'green')
plot_errorbar(ax, fracs5, all_nce3_ndiag_mses, nce3_ndiag_deciles, 'NCE (random fill in)', 'red')

ax.set_ylim(0, 0.125)
ax.set_title('Off-diagonal of K')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
ax.legend(loc='upper left')

fig.tight_layout()
save_fig(fig, save_dir, 'fraction_missing_vs_mse')
