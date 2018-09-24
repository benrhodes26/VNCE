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

import matplotlib as matplotlib
import numpy as np
import pickle
import seaborn as sns

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
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
parser.add_argument('--exp_name', type=str, default='test5d_nu10/', help='name of set of experiments this one belongs to')  # 5d-vlr0.1-nz=10-final
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm/')
# parser.add_argument('--load_dir', type=str, default='/home/ben/masters-project-non-code/experimental-results/trunc-norm/')

args = parser.parse_args()
main_load_dir = os.path.join(args.load_dir, args.exp_name)
main_load_dir = os.path.expanduser(main_load_dir)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def split_params(theta):
    try:
        mean = theta[1:1+d]
    except IndexError:
        pass
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

    mean = mean_square_error(true_mean, mean)
    mean_b = mean_square_error(true_b, b)
    diag_prec_mse = mean_square_error(true_prec_diag, prec_diag)
    n_diag_prec_mse = mean_square_error(true_prec_n_diag, prec_n_diag)

    return mean, mean_b, diag_prec_mse, n_diag_prec_mse


def plot_errorbar_helper(ax, fracs, mses, deciles, label, color):
    median = np.median(np.array(mses), 0)
    ax.errorbar(fracs, median, yerr=[median - deciles[0], deciles[1] - median], fmt='o',
                markersize=1, linestyle='None', label=label, color=color, capsize=1, capthick=1)


def plot_errorbar(ax, fracs, mses, label, color):
    deciles = np.percentile(mses, percentiles, axis=0)
    plot_errorbar_helper(ax, fracs, mses, deciles, label, color)


def plot_theta0_mses(axs, theta0_mu_mse, theta0_mean_mse, theta0_diag_mse, theta0_ndiag_mse):
    axs[0].plot((0, 1), (theta0_mu_mse, theta0_mu_mse), linestyle='--')
    axs[1].plot((0, 1), (theta0_mean_mse, theta0_mean_mse), linestyle='--')
    axs[2].plot((0, 1), (theta0_diag_mse, theta0_diag_mse), linestyle='--')
    axs[3].plot((0, 1), (theta0_ndiag_mse, theta0_ndiag_mse), linestyle='--')


# methods = ['VNCE (true)', 'VNCE (cdi approx)', 'VNCE (lognormal approx)', 'NCE (means fill in)', 'NCE (noise fill in)', 'NCE (random fill in)']
# filenames = ['vnce_results1.npz', 'vnce_results2.npz', 'vnce_results3.npz', 'nce_results1.npz', 'nce_results2.npz', 'nce_results3.npz']
# colors = ['black', 'blue', 'purple', 'orange', 'green', 'red']
methods = ['VNCE (true)', 'VNCE (lognormal approx)', 'NCE (means fill in)', 'NCE (noise fill in)']
filenames = ['vnce_results1.npz', 'vnce_results3.npz', 'nce_results1.npz', 'nce_results2.npz']
colors = ['black', 'purple', 'orange', 'green']
num_methods = len(methods)

all_mus = [[] for i in range(num_methods)]
all_means = [[] for i in range(num_methods)]
all_diags = [[] for i in range(num_methods)]
all_ndiags = [[] for i in range(num_methods)]

for outer_file in os.listdir(main_load_dir):
    load_dir = os.path.join(main_load_dir, outer_file, 'best')

    frac_to_mu_mse_dict = {}
    frac_to_mean_mse_dict = {}
    frac_to_prec_diag_mse_dict = {}
    frac_to_n_prec_diag_mse_dict = {}

    # loop through files and get mses of vnce and nce with 0s
    sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
    sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]
    for i, file in enumerate(sorted_dirs):
        exp = os.path.join(load_dir, file)
        config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
        frac = float(config.frac_missing)
        d = config.d

        loaded_theta = np.load(os.path.join(exp, 'theta0_and_theta_true.npz'))
        true_theta = loaded_theta['theta_true']
        theta0 = loaded_theta['theta0']
        if i == 0:
            theta0_mu_mse, theta0_mean_mse, theta0_diag_mse, theta0_ndiag_mse = get_mses(true_theta, theta0, d)

        for filename in filenames:
            loaded = np.load(os.path.join(exp, filename))
            if filename[0] == 'v':
                thetas = loaded['vnce_thetas']
                if isinstance(thetas[-1][-1], float):
                    theta = thetas[-1]
                else:
                    theta = thetas[-1][-1]
            else:
                theta = loaded['nce_thetas'][-1]

            mu_mse, mean_mse, diag_prec_mse, n_diag_prec_mse = get_mses(true_theta, theta, d)
            frac_to_mu_mse_dict.setdefault(str(frac), []).append(mu_mse)
            frac_to_mean_mse_dict.setdefault(str(frac), []).append(mean_mse)
            frac_to_prec_diag_mse_dict.setdefault(str(frac), []).append(diag_prec_mse)
            frac_to_n_prec_diag_mse_dict.setdefault(str(frac), []).append(n_diag_prec_mse)

    fracs = sorted([float(key) for key in frac_to_mean_mse_dict.keys()])

    for i in range(num_methods):
        all_mus[i].append([frac_to_mu_mse_dict[str(frac)][i] for frac in fracs])
        all_means[i].append([frac_to_mean_mse_dict[str(frac)][i] for frac in fracs])
        all_diags[i].append([frac_to_prec_diag_mse_dict[str(frac)][i] for frac in fracs])
        all_ndiags[i].append([frac_to_n_prec_diag_mse_dict[str(frac)][i] for frac in fracs])


sns.set_style("darkgrid")
fig, axs = plt.subplots(4, 1, figsize=(5.7, 10))
axs = axs.ravel()
fracs = np.array(fracs)
x_positions = []
for i in np.linspace(-0.015, 0.015, num_methods):
    x_positions.append(fracs + i)
percentiles = [25, 75]

for i in range(num_methods):
    plot_errorbar(axs[0], x_positions[i], all_mus[i], methods[i], colors[i])
    plot_errorbar(axs[1], x_positions[i], all_means[i], methods[i], colors[i])
    plot_errorbar(axs[2], x_positions[i], all_diags[i], methods[i], colors[i])
    plot_errorbar(axs[3], x_positions[i], all_ndiags[i], methods[i], colors[i])
    plot_theta0_mses(axs, theta0_mu_mse, theta0_mean_mse, theta0_diag_mse, theta0_ndiag_mse)

titles = [r'$\mu$', r'\textbf{b}', 'Diagonal of K', 'Off-diagonal of K']
# upper_lims = [1, 1, 0.2, 0.2]
upper_lims = [1, 10, 10, 0.2]
for i, ax in enumerate(axs):
    ax.set_ylim(0, upper_lims[i])
    ax.set_title(titles[i])
    ax.set_xlabel('fraction of data missing')
    ax.set_ylabel('MSE')
    ax.legend(loc='upper left')

fig.tight_layout()
save_fig(fig, save_dir, 'fraction_missing_vs_mse')
