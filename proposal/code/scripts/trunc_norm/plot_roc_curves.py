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

import graph_tool as gt
import matplotlib as matplotlib
import numpy as np
import pickle
import seaborn as sns

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from graph_tool.draw import graph_draw
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd
from sklearn.metrics import roc_curve, auc

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from plot import *
from utils import take_closest

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=10)
# rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='plot relationship between fraction of training data missing and final mean-squared error for'
                                    'a truncated normal model trained with VNCE', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project-non-code/experiments/trunc-norm/')
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm/')
# parser.add_argument('--load_dir', type=str, default='~/masters-project-non-code/experimental-results/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='5d-vlr0.1-nz=10-final', help='name of set of experiments this one belongs to')

args = parser.parse_args()
main_load_dir = os.path.join(args.load_dir, args.exp_name)
main_load_dir = os.path.expanduser(main_load_dir)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.switch_backend('cairo')


def get_lower_diag_matrix(d, theta):
    precision = np.zeros((d, d))
    i_tril = np.tril_indices(d)
    precision[i_tril] = theta[1 + d:]
    precision[np.diag_indices(d)] = 0
    return precision

def get_auc_fpr_tpr(metrics_dict, true_theta, est_theta, d):
    true_precision = get_lower_diag_matrix(d, true_theta)
    est_precision = get_lower_diag_matrix(d, est_theta)

    true_edges = np.nonzero(true_precision)
    true_edges = list(zip(true_edges[0], true_edges[1]))
    all_edges = np.tril_indices(d, -1)
    all_edges = list(zip(all_edges[0], all_edges[1]))
    true_labels = []
    for e in all_edges:
        label = 1 if e in true_edges else 0
        true_labels.append(label)

    scores = est_precision[np.tril_indices(d, -1)]
    scores = np.abs(scores)
    scores = scores / np.sum(scores)
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    metrics_dict['auc'].append(roc_auc)
    metrics_dict['fpr'].append(fpr.tolist())
    metrics_dict['tpr'].append(tpr.tolist())


def append_to_all_metrics(all_metrics, metrics):
    all_metrics['auc'].append(metrics['auc'])
    all_metrics['fpr'].append(metrics['fpr'])
    all_metrics['tpr'].append(metrics['tpr'])


all_vnce1_metrics = {'auc': [], 'fpr': [], 'tpr': []}
all_vnce2_metrics = {'auc': [], 'fpr': [], 'tpr': []}
all_nce1_metrics = deepcopy(all_vnce2_metrics)
all_nce2_metrics = deepcopy(all_vnce2_metrics)
all_nce3_metrics = deepcopy(all_vnce2_metrics)

for outer_file in os.listdir(main_load_dir):
    load_dir = os.path.join(main_load_dir, outer_file, 'best')
    vnce_metrics1 = {'auc': [], 'fpr': [], 'tpr': []}
    vnce_metrics2 = {'auc': [], 'fpr': [], 'tpr': []}
    nce1_metrics = deepcopy(vnce_metrics2)
    nce2_metrics = deepcopy(vnce_metrics2)
    nce3_metrics = deepcopy(vnce_metrics2)

    sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
    sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]
    for i, file in enumerate(sorted_dirs):
        load = os.path.join(load_dir, file)
        config = pickle.load(open(os.path.join(load, 'config.p'), 'rb'))
        frac = float(config.frac_missing)
        d = config.d

        loaded = np.load(os.path.join(load, 'theta0_and_theta_true.npz'))
        loaded1 = np.load(os.path.join(load, 'vnce_results1.npz'))
        loaded2 = np.load(os.path.join(load, 'vnce_results2.npz'))
        loaded3 = np.load(os.path.join(load, 'nce_results1.npz'))
        loaded4 = np.load(os.path.join(load, 'nce_results2.npz'))
        loaded5 = np.load(os.path.join(load, 'nce_results3.npz'))

        theta_true = loaded['theta_true']
        vnce_theta1 = loaded1['vnce_thetas'][-1][-1]
        vnce_theta2 = loaded2['vnce_thetas'][-1][-1]
        nce_means_theta = loaded3['nce_thetas'][-1]
        nce_noise_theta = loaded4['nce_thetas'][-1]
        nce_rnd_theta = loaded5['nce_thetas'][-1]

        get_auc_fpr_tpr(vnce_metrics1, theta_true, vnce_theta1, d)
        get_auc_fpr_tpr(vnce_metrics2, theta_true, vnce_theta2, d)
        get_auc_fpr_tpr(nce1_metrics, theta_true, nce_means_theta, d)
        get_auc_fpr_tpr(nce2_metrics, theta_true, nce_noise_theta, d)
        get_auc_fpr_tpr(nce3_metrics, theta_true, nce_rnd_theta, d)

    append_to_all_metrics(all_vnce1_metrics, vnce_metrics1)
    append_to_all_metrics(all_vnce2_metrics, vnce_metrics2)
    append_to_all_metrics(all_nce1_metrics, nce1_metrics)
    append_to_all_metrics(all_nce2_metrics, nce2_metrics)
    append_to_all_metrics(all_nce3_metrics, nce3_metrics)


vnce1_fpr = np.array(all_vnce1_metrics['fpr'])
vnce2_fpr = np.array(all_vnce2_metrics['fpr'])
nce1_fpr = np.array(all_nce1_metrics['fpr'])
nce2_fpr = np.array(all_nce2_metrics['fpr'])
nce3_fpr = np.array(all_nce3_metrics['fpr'])

vnce1_tpr = np.array(all_vnce1_metrics['tpr'])
vnce2_tpr = np.array(all_vnce2_metrics['tpr'])
nce1_tpr = np.array(all_nce1_metrics['tpr'])
nce2_tpr = np.array(all_nce2_metrics['tpr'])
nce3_tpr = np.array(all_nce3_metrics['tpr'])

num_simulations = len(vnce1_fpr)
fpr_range = np.arange(0, 1.02, 0.02)
method_names = ['VNCE (true)', 'VNCE (approx)', 'NCE (means)', 'NCE (noise)', 'NCE (random)']
method_colours = ['black', 'blue', 'orange', 'green', 'red']

sns.set_style("darkgrid")
roc_fig, roc_axs = plt.subplots(5, 2, figsize=(5.7, 11), sharex=True, sharey=True)
roc_axs = roc_axs.ravel()

for frac_i in np.arange(10):
    ax = roc_axs[frac_i]
    for method_i, method in enumerate([[vnce1_fpr, vnce1_tpr], [vnce2_fpr, vnce2_tpr], [nce1_fpr, nce1_tpr], [nce2_fpr, nce2_tpr], [nce3_fpr, nce3_tpr]]):
        decile_curves = []
        for fpr in fpr_range:
            # loop through each simulation, and interpolate its true positive rate
            tprs_across_sims = []
            for sim_i in range(num_simulations):
                method_fprs = method[0][sim_i][frac_i]
                method_tprs = method[1][sim_i][frac_i]
                # get closest method_fprs to fpr
                fpr_index = take_closest(method_fprs, fpr)
                # interpolate tprs
                tpr1, tpr2 = method_tprs[fpr_index], method_tprs[fpr_index + 1]
                tpr = (tpr1 + tpr2) / 2
                tprs_across_sims.append(tpr)

            tpr_deciles = np.percentile(np.array(tprs_across_sims), [10, 50, 90])
            decile_curves.append(tpr_deciles)

        decile_curves = np.array(decile_curves)
        # ax.plot(fpr_range, decile_curves[:, 0], '--', label=method_names[method_i], color=method_colours[method_i], alpha=0.3)
        if method_i == 1:
            ax.plot(fpr_range, decile_curves[:, 1], '-.', label=method_names[method_i], color=method_colours[method_i])
        elif method_i == 2:
            ax.plot(fpr_range, decile_curves[:, 1], '--', label=method_names[method_i], color=method_colours[method_i])
        elif method_i == 3:
            pass
            # ax.plot(fpr_range, decile_curves[:, 1], '-.', label=method_names[method_i], color=method_colours[method_i])
        elif method_i == 4:
            pass
            # ax.plot(fpr_range, decile_curves[:, 1], label=method_names[method_i], color=method_colours[method_i], alpha=0.5)
        else:
            ax.plot(fpr_range, decile_curves[:, 1], label=method_names[method_i], color=method_colours[method_i])

        # ax.plot(fpr_range, decile_curves[:, 2], '--', label=method_names[method_i], color=method_colours[method_i], alpha=0.3)
        ax.set_title('{}% missing'.format(frac_i*10))
        ax.legend(loc='best')
        remove_duplicate_legends(ax)

roc_axs = roc_axs.reshape(5, 2)
for ax in roc_axs[:, 0]:
    ax.set_ylabel('True Positive Rate')
for ax in roc_axs[-1, :]:
    ax.set_xlabel('False Positive Rate')
roc_fig.tight_layout()
save_fig(roc_fig, save_dir, 'roc_curves')

vnce1_aucs = np.array(all_vnce1_metrics['auc'])
vnce2_aucs = np.array(all_vnce2_metrics['auc'])
nce1_aucs = np.array(all_nce1_metrics['auc'])
nce2_aucs = np.array(all_nce2_metrics['auc'])
nce3_aucs = np.array(all_nce3_metrics['auc'])

percentiles = [25, 50, 75]
vnce1_aucs_deciles = np.percentile(vnce1_aucs, percentiles, axis=0)
vnce2_aucs_deciles = np.percentile(vnce2_aucs, percentiles, axis=0)
nce1_aucs_deciles = np.percentile(nce1_aucs, percentiles, axis=0)
nce2_aucs_deciles = np.percentile(nce2_aucs, percentiles, axis=0)
nce3_aucs_deciles = np.percentile(nce3_aucs, percentiles, axis=0)

fracs = np.arange(10) / 10
fracs1 = fracs - 0.02
fracs2 = fracs - 0.01
fracs3 = fracs
fracs4 = fracs + 0.01
fracs5 = fracs + 0.02

def plot_auc(ax, fracs, deciles, label, colour):
    ax.errorbar(fracs, deciles[1], yerr=[deciles[1] - deciles[0], deciles[2] - deciles[1]], fmt='o',
                markersize=2, linestyle='None', label=label, color=colour, capsize=3, capthick=0.5)

aucs_fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
plot_auc(ax, fracs1, vnce1_aucs_deciles, 'VNCE (true)', 'black')
plot_auc(ax, fracs2, vnce2_aucs_deciles, 'VNCE (approx)', 'blue')
plot_auc(ax, fracs3, nce1_aucs_deciles, 'NCE (means)', 'orange')
plot_auc(ax, fracs4, nce2_aucs_deciles, 'NCE (noise)', 'green')
plot_auc(ax, fracs5, nce3_aucs_deciles, 'NCE (random)', 'red')

ax.set_xlabel('fraction missing')
ax.set_ylabel('AUC')
ax.legend(loc='lower left')
remove_duplicate_legends(ax)

save_fig(aucs_fig, save_dir, 'auc_curves')
