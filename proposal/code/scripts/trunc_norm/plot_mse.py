import os
import sys
CODE_DIRS = ['~/masters-project/ben-rhodes-masters-project/proposal/code', '/home/s1771906/code']
CODE_DIRS_2 = [d + '/neural_network' for d in CODE_DIRS] + CODE_DIRS
CODE_DIRS_2 = [os.path.expanduser(d) for d in CODE_DIRS_2]
for code_dir in CODE_DIRS_2:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import matplotlib as matplotlib
import numpy as np
import pickle
import seaborn as sns

from plot import *
from project_statics import *
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
parser.add_argument('--save_dir', type=str, default=RESULTS + '/trunc-norm/')
# parser.add_argument('--exp_name', type=str, default='20d_reg_param_hub/0/separated_results/3/', help='name of set of experiments this one belongs to')  # 5d-vlr0.1-nz=10-final
parser.add_argument('--exp_name', type=str, default='20d_reg_param_cir/', help='name of set of experiments this one belongs to')  # 5d-vlr0.1-nz=10-final
parser.add_argument('--load_dir', type=str, default=EXPERIMENT_OUTPUTS + '/trunc_norm/')

args = parser.parse_args()
main_load_dir = os.path.join(args.load_dir, args.exp_name)
main_load_dir = os.path.expanduser(main_load_dir)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def split_params(dist, theta, d):
    dist.theta = theta
    mean, prec = dist.get_joint_pretruncated_params()[:2]
    prec_diag = deepcopy(prec[np.diag_indices(d)])
    prec_non_diag = deepcopy(prec[np.tril_indices(d, -1)])

    return mean, prec, prec_diag, prec_non_diag


def get_mses(data_dist, model, true_theta, theta, d):
    true_mean, true_prec, true_prec_diag, true_prec_n_diag = split_params(data_dist, true_theta, d)
    mean, prec, prec_diag, prec_n_diag = split_params(model, theta, d)

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


# methods = ['VNCE (lognormal approx)', 'NCE (means fill in)', 'MLE (sampling)']
methods = ['VNCE (lognormal approx)', 'NCE (means fill in)']

# param_file = ['vnce_results3.npz', 'nce_results1.npz', 'cd_results.npz']
param_file = ['vnce_results3.npz', 'nce_results1.npz']

# model_files = ['vnce_model3.p', 'nce_model1.p', 'cd_model.p']
model_files = ['vnce_model3.p', 'nce_model1.p']

# colors = ['purple', 'orange', 'black']
colors = ['purple', 'orange']

num_methods = len(methods)
# sorted_fracs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
# sorted_fracs = np.array([0.1, 0.3, 0.5])
sorted_fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

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
    # sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
    sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]
    for i, file in enumerate(sorted_dirs):
        exp = os.path.join(load_dir, file)
        config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
        frac = float(config.frac_missing)
        d = config.d

        theta_true = config.theta_true
        data_dist = config.data_dist

        for p_file, m_file in zip(param_file, model_files):
            loaded = np.load(os.path.join(exp, p_file))
            if p_file[0] == 'v':
                thetas = loaded['vnce_thetas']
                print('VNCE times:', loaded['vnce_times'][-1])
                if isinstance(thetas[-1][-1], float):
                    theta = thetas[-1]
                else:
                    theta = thetas[-1][-1]

            elif p_file[0] == 'n':
                theta = loaded['nce_thetas'][-1]
            else:
                theta = loaded['cd_thetas'][-1]
                print('CD times:', loaded['cd_times'][-1])

            model = pickle.load(open(os.path.join(exp, m_file), 'rb'))
            mu_mse, mean_mse, diag_prec_mse, n_diag_prec_mse = get_mses(data_dist, model, true_theta, theta, d)

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
# fig, axs = plt.subplots(4, 1, figsize=(5.7, 10))
fig, axs = plt.subplots(2, 1, figsize=(5.7, 5.7))
axs = axs.ravel()
fracs = np.array(fracs)
x_positions = []
for i in np.linspace(-0.015, 0.015, num_methods):
    x_positions.append(fracs + i)
percentiles = [25, 75]

for i in range(num_methods):
    # plot_errorbar(axs[0], x_positions[i], all_mus[i], methods[i], colors[i])
    # plot_errorbar(axs[1], x_positions[i], all_means[i], methods[i], colors[i])
    plot_errorbar(axs[0], x_positions[i], all_diags[i], methods[i], colors[i])
    plot_errorbar(axs[1], x_positions[i], all_ndiags[i], methods[i], colors[i])

titles = ['Diagonal of K', 'Off-diagonal of K']
# upper_lims = [1, 1, 0.2, 0.2]
upper_lims = [2, 0.1]
for i, ax in enumerate(axs):
    ax.set_ylim(0, upper_lims[i])
    ax.set_title(titles[i])
    ax.set_xlabel('fraction of data missing')
    ax.set_ylabel('MSE')
    ax.legend(loc='upper left')

fig.tight_layout()
save_fig(fig, save_dir, 'fraction_missing_vs_mse')
