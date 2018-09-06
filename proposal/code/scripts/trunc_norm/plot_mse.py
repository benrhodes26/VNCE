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

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=8)
#   rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='plot relationship between fraction of training data missing and final mean-squared error for'
                                    'a truncated normal model trained with VNCE', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project-non-code/experiments/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='4d/cross-val/best', help='name of set of experiments this one belongs to')
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm/')
# parser.add_argument('--load_dir', type=str, default='~/masters-project-non-code/experimental-results/trunc-norm/')

args = parser.parse_args()
load_dir = os.path.join(args.load_dir, args.exp_name)
load_dir = os.path.expanduser(load_dir)
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

    true_prec_inv = np.linalg.inv(true_prec)
    A = np.dot(true_prec_inv, prec)
    A_diag = A[np.diag_indices(d)]
    A_ndiag = A[np.tril_indices(d, -1)]

    # print(A)
    print(true_prec)
    print(prec)
    # print(true_cov)
    # print(cov)

    mean_mse = mean_square_error(true_mean, mean)
    diag_prec_mse = mean_square_error(true_prec_diag, prec_diag)
    n_diag_prec_mse = mean_square_error(true_prec_n_diag, prec_n_diag)
    # diag_prec_mse = mean_square_error(true_cov_diag, cov_diag)
    # n_diag_prec_mse = mean_square_error(true_cov_n_diag, cov_n_diag)
    # diag_prec_mse = mean_square_error(np.ones_like(A_diag), A_diag)
    # n_diag_prec_mse = mean_square_error(np.ones_like(A_ndiag), A_ndiag)

    return mean_mse, diag_prec_mse, n_diag_prec_mse


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
    loaded1 = np.load(os.path.join(exp, 'vnce_results.npz'))
    loaded2 = np.load(os.path.join(exp, 'nce_filled_in_means_results.npz'))
    loaded3 = np.load(os.path.join(exp, 'nce_filled_in_noise_results.npz'))

    # vnce_mse = loaded1['vnce_mse']
    # nce_missing_mse = loaded2['nce_missing_mse']  # missing data filled-in with means
    # nce_missing_mse_2 = loaded3['nce_missing_mse_2']  # missing data filled-in with noise
    true_theta = loaded['theta_true']
    vnce_theta = loaded1['vnce_thetas'][-1][-1]
    nce_means_theta = loaded2['nce_means_thetas'][-1]
    nce_noise_theta = loaded3['nce_noise_thetas'][-1]

    print('--------------{}------------\n'.format(frac))
    vnce_mean_mse, vnce_diag_prec_mse, vnce_n_diag_prec_mse = get_mses(true_theta, vnce_theta, d)
    nce1_mean_mse, nce1_diag_prec_mse, nce1_n_diag_prec_mse = get_mses(true_theta, nce_means_theta, d)
    nce2_mean_mse, nce2_diag_prec_mse, nce2_n_diag_prec_mse = get_mses(true_theta, nce_noise_theta, d)

    frac_to_mean_mse_dict[str(frac)] = [vnce_mean_mse, nce1_mean_mse, nce2_mean_mse]
    frac_to_prec_diag_mse_dict[str(frac)] = [vnce_diag_prec_mse, nce1_diag_prec_mse, nce2_diag_prec_mse]
    frac_to_n_prec_diag_mse_dict[str(frac)] = [vnce_n_diag_prec_mse, nce1_n_diag_prec_mse, nce2_n_diag_prec_mse]

    # frac_to_rel_mse_dict[str(frac)] = [1, nce_missing_mse / vnce_mse, nce_missing_mse_2 / vnce_mse]

fracs = sorted([float(key) for key in frac_to_mean_mse_dict.keys()])
# fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
vnce_means_mses = [frac_to_mean_mse_dict[str(frac)][0] for frac in fracs]
nce1_means_mses = [frac_to_mean_mse_dict[str(frac)][1] for frac in fracs]
nce2_means_mses = [frac_to_mean_mse_dict[str(frac)][2] for frac in fracs]

vnce_diag_mses = [frac_to_prec_diag_mse_dict[str(frac)][0] for frac in fracs]
nce1_diag_mses = [frac_to_prec_diag_mse_dict[str(frac)][1] for frac in fracs]
nce2_diag_mses = [frac_to_prec_diag_mse_dict[str(frac)][2] for frac in fracs]

vnce_ndiag_mses = [frac_to_n_prec_diag_mse_dict[str(frac)][0] for frac in fracs]
nce1_ndiag_mses = [frac_to_n_prec_diag_mse_dict[str(frac)][1] for frac in fracs]
nce2_ndiag_mses = [frac_to_n_prec_diag_mse_dict[str(frac)][2] for frac in fracs]

# rel_vnce_mses = [frac_to_rel_mse_dict[str(frac)][0] for frac in fracs]
# rel_nce_missing_mses = [frac_to_rel_mse_dict[str(frac)][1] for frac in fracs]
# rel_nce_missing_mses_2 = [frac_to_rel_mse_dict[str(frac)][2] for frac in fracs]

sns.set_style("darkgrid")
fig, axs = plt.subplots(3, 1, figsize=(5, 4))
axs = axs.ravel()

ax = axs[0]
ax.plot(np.array(fracs), np.array(vnce_means_mses), label='VNCE')
ax.plot(np.array(fracs), np.array(nce1_means_mses), label='NCE (means fill in)')
ax.plot(np.array(fracs), np.array(nce2_means_mses), label='NCE (noise fill in)')
# ax.plot((0, 1), (loaded['init_mse'], loaded['init_mse']), label='Initial params')
ax.set_title('Mean')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
ax.legend(loc='best')

ax = axs[1]
ax.plot(np.array(fracs), np.array(vnce_diag_mses), label='VNCE')
ax.plot(np.array(fracs), np.array(nce1_diag_mses), label='NCE (means fill in)')
ax.plot(np.array(fracs), np.array(nce2_diag_mses), label='NCE (noise fill in)')
# ax.plot((0, 1), (loaded['init_mse'], loaded['init_mse']), label='Initial params')
ax.set_title('Diagonal')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
# ax.set_ylim(0, 200)
ax.legend(loc='best')

ax = axs[2]
ax.plot(np.array(fracs), np.array(vnce_ndiag_mses), label='VNCE')
ax.plot(np.array(fracs), np.array(nce1_ndiag_mses), label='NCE (means fill in)')
ax.plot(np.array(fracs), np.array(nce2_ndiag_mses), label='NCE (noise fill in)')
# ax.plot((0, 1), (loaded['init_mse'], loaded['init_mse']), label='Initial params')
ax.set_title('Off-diagonal')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
# ax.set_ylim(0, 200)
ax.legend(loc='best')

fig.tight_layout()
save_fig(fig, save_dir, 'fraction_missing_vs_mse')

# ax = axs[1]
# ax.plot(np.array(fracs), np.array(rel_vnce_mses), label='VNCE')
# ax.plot(np.array(fracs), np.array(rel_nce_missing_mses), label='NCE (means fill in)')
# ax.plot(np.array(fracs), np.array(rel_nce_missing_mses_2), label='NCE (noise fill in)')
# ax.plot((0, 1), (loaded['init_mse'], loaded['init_mse']), label='Initial params')
# ax.set_xlabel('fraction of data missing')
# ax.set_ylabel('relative MSE')
# ax.legend(loc='best')

