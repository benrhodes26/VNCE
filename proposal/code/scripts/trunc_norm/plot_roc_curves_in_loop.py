import os
import sys
CODE_DIRS = ['~/masters-project/ben-rhodes-masters-project/proposal/code', '/home/s1771906/code']
CODE_DIRS_2 = [d + '/neural_network' for d in CODE_DIRS] + CODE_DIRS
CODE_DIRS_2 = [os.path.expanduser(d) for d in CODE_DIRS_2]
for code_dir in CODE_DIRS_2:
    if code_dir not in sys.path:
        sys.path.append(code_dir)
#
# import graph_tool as gt
import matplotlib as matplotlib
import numpy as np
import pickle
import seaborn as sns

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
# from graph_tool.draw import graph_draw
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd
from sklearn.metrics import roc_curve, auc

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from plot import *
from project_statics import *
from utils import take_closest

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=8)
rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='plot relationship between fraction of training data missing and final mean-squared error for'
                                    'a truncated normal model trained with VNCE', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default=RESULTS + '/trunc-norm/')
parser.add_argument('--load_dir', type=str, default=EXPERIMENT_OUTPUTS + '/trunc_norm/')
# parser.add_argument('--exp_name', type=str, default='20d_reg_param/0/separated_results/3/', help='name of set of experiments this one belongs to')
parser.add_argument('--exp_name', type=str, default='20d_reg_param_cir/', help='name of set of experiments this one belongs to')

args = parser.parse_args()
main_load_dir = os.path.join(args.load_dir, args.exp_name)
main_load_dir = os.path.expanduser(main_load_dir)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.switch_backend('cairo')


def get_lower_diag_matrix(dist, theta, d):
    dist.theta = theta
    precision = dist.get_joint_pretruncated_params()[1]
    precision[np.diag_indices(d)] = 0  # ignore diagonals - only interested in off diagonal elements
    return precision

def get_auc_fpr_tpr(metrics_dict, data_dist, model, true_theta, est_theta, d):
    true_precision = get_lower_diag_matrix(data_dist, true_theta, d)
    est_precision = get_lower_diag_matrix(model, est_theta, d)

    true_precision[true_precision < 1e-5] = 0  # get rid of any numerical errors
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
    # scores = scores / np.sum(scores)

    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    metrics_dict['auc'].append(roc_auc)
    metrics_dict['fpr'].append(fpr.tolist())
    metrics_dict['tpr'].append(tpr.tolist())



def append_to_all_metrics(all_metrics, metrics):
    all_metrics['auc'].append(metrics['auc'])
    all_metrics['fpr'].append(metrics['fpr'])
    all_metrics['tpr'].append(metrics['tpr'])

def plot_auc(ax, fracs, deciles, label, colour, marker):
    ax.errorbar(fracs, deciles[1], yerr=[deciles[1] - deciles[0], deciles[2] - deciles[1]], fmt=marker,
                markersize=2, linestyle='None', label=label, color=colour, capsize=3, capthick=0.5)

'-----------------------------------------------------------------------------------------------------'
'------------------------------NUM METHODS AND FRACTIONS MISSING--------------------------------------'
'-----------------------------------------------------------------------------------------------------'
# fracs = np.arange(0, 10, 2) / 10
# fracs = np.array([0.2, 0.5])
# fracs = np.array([0.1, 0.3, 0.5])
fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
# fracs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
sorted_fracs = fracs
frac_range = np.arange(1)  # 10

# reg_params = [0.0, 0.00001, 0.0001, 0.001, 0.01]
# reg_params = [0.0, 0.00001, 0.0001, 0.001]
reg_params = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])

reg_markers = ['.', 's', '^', 'P', 'p']

fpr_range = np.arange(0, 1.02, 0.02)
# method_names = ['VNCE (lognormal)', 'NCE (means)', 'NCE (noise)', 'MLE (sampling)']
# method_names = ['VNCE (lognormal)', 'NCE (means)', 'MLE (sampling)']
# method_names = ['VNCE (lognormal)', 'MLE (sampling)']
method_names = ['VNCE (lognormal)', 'NCE (means)']
# method_names = ['VNCE (lognormal)']

# param_files = ['vnce_results3.npz', 'nce_results1.npz', 'nce_results2.npz', 'cd_results.npz']
# param_files = ['vnce_results3.npz', 'nce_results1.npz', 'cd_results.npz']
# param_files = ['vnce_results3.npz', 'cd_results.npz']
param_files = ['vnce_results3.npz', 'nce_results1.npz']
# param_files = ['vnce_results3.npz']
# param_files = ['nce_results1.npz', 'nce_results2.npz']
# param_files = ['vnce_results.npz']

# model_files = ['vnce_model3.p', 'nce_model1.p', 'cd_model.p']
model_files = ['vnce_model3.p', 'nce_model1.p']
# model_files = ['vnce_model3.p']
# model_files = ['vnce_model.p']

# method_colours = ['purple', 'orange', 'green', 'black']
# method_colours = ['purple', 'orange', 'black']
# method_colours = ['purple', 'black']
method_colours = ['purple', 'orange']
# method_colours = ['purple']
# method_colours = ['orange', 'green']

# method_linestyles = ['-', '--', '.']
method_linestyles = ['--', '-']
# method_linestyles = ['--']

num_methods = len(method_names)


aucs_fig, auc_ax = plt.subplots(1, 1, figsize=(6.5, 5))

for reg_marker, reg in zip(reg_markers, reg_params):
    '-----------------------------------------------------------------------------------------------------'
    '-------------------------------------PLOT ROC CURVES-------------------------------------------------'
    '-----------------------------------------------------------------------------------------------------'
    all_metrics = [{'auc': [], 'fpr': [], 'tpr': []} for i in range(num_methods)]
    for outer_file in os.listdir(main_load_dir):
        load_dir = os.path.join(main_load_dir, outer_file)
        metrics = [{'auc': [], 'fpr': [], 'tpr': []} for i in range(num_methods)]

        # sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
        sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]
        for i, file in enumerate(sorted_dirs):
            load = os.path.join(load_dir, file, "reg{}".format(reg))
            config = pickle.load(open(os.path.join(load, 'config.p'), 'rb'))
            frac = float(config.frac_missing)
            d = config.d

            theta_true = config.theta_true
            data_dist = config.data_dist

            for j in range(num_methods):
                loaded = np.load(os.path.join(load, param_files[j]))
                if param_files[j][0] == 'v':
                    theta = loaded['vnce_thetas'][-1][-1]
                elif param_files[j][0] == 'n':
                    theta = loaded['nce_thetas'][-1]
                elif param_files[j][0] == 'c':
                    theta = loaded['cd_thetas'][-1]

                model = pickle.load(open(os.path.join(load, model_files[j]), 'rb'))
                get_auc_fpr_tpr(metrics[j], data_dist, model, theta_true, theta, d)

        for j in range(num_methods):
            append_to_all_metrics(all_metrics[j], metrics[j])

    fprs = []
    tprs = []
    for j in range(num_methods):
        fprs.append(np.array(all_metrics[j]['fpr']))
        tprs.append(np.array(all_metrics[j]['tpr']))

    sns.set_style("darkgrid")
    roc_fig, roc_axs = plt.subplots(5, 2, figsize=(5.7, 11), sharex=True, sharey=True)
    roc_axs = roc_axs.ravel()

    num_simulations = len(fprs[0])
    for frac_i in frac_range:
        ax = roc_axs[frac_i]
        for method_i, method in enumerate(zip(fprs, tprs)):
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
            ax.plot(fpr_range, decile_curves[:, 1], method_linestyles[method_i], label=method_names[method_i], color=method_colours[method_i])
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

    '-----------------------------------------------------------------------------------------------------'
    '-------------------------------------PLOT AUC CURVES-------------------------------------------------'
    '-----------------------------------------------------------------------------------------------------'
    x_positions = []
    for i in np.linspace(-0.015, 0.015, num_methods):
        x_positions.append(fracs + i)

    percentiles = [25, 50, 75]
    # percentiles = [1, 50, 99]
    aucs = [np.array(metric['auc']) for metric in all_metrics]
    deciles = [np.percentile(auc, percentiles, axis=0) for auc in aucs]

    for j in range(num_methods):
        plot_auc(auc_ax, x_positions[j], deciles[j], method_names[j] + 'reg{}'.format(reg), method_colours[j], reg_marker)

auc_ax.set_xlabel('fraction missing')
auc_ax.set_ylabel('AUC')
auc_ax.legend(loc='lower left')
# remove_duplicate_legends(ax)

save_fig(aucs_fig, save_dir, 'auc_curves')
