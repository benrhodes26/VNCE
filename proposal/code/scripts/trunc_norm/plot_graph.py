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
rc('legend', fontsize=8)
# rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='plot relationship between fraction of training data missing and final mean-squared error for'
                                    'a truncated normal model trained with VNCE', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project-non-code/experiments/trunc-norm/')
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm/')
# parser.add_argument('--load_dir', type=str, default='~/masters-project-non-code/experimental-results/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='4d/cross-val/best', help='name of set of experiments this one belongs to')

args = parser.parse_args()
load_dir = os.path.join(args.load_dir, args.exp_name)
save_dir = os.path.join(args.save_dir, args.exp_name)
load_dir = os.path.expanduser(load_dir)
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

def convert_weights(weights):
    min = weights.min()
    max = weights.max()
    weights = (weights - min / (max - min))**5
    weights *= (max - min)
    weights += min
    return weights

# noinspection PyPep8Naming
def plot_graph(ax, theta, d):
    precision = get_lower_diag_matrix(d, theta)

    # to calculate edges, get indices of the d largest vals in the lower triangle of the precision matrix
    p_flat = precision[np.tril_indices(d, -1)]
    # p_flat.sort()
    # dth_largest = p_flat[-d]
    # precision[np.triu_indices(d)] = p_flat[0]  # hack to help us calculate edges easiliy
    # edges = np.where(precision >= dth_largest)

    # p_flat = np.abs(p_flat / np.max(p_flat))
    p_flat = convert_weights(np.abs(p_flat))
    edges = np.tril_indices(d, -1)

    G = gt.Graph(directed=False)
    G.add_edge_list([(v1, v2) for v1, v2 in zip(edges[0], edges[1])])

    # G.add_edge(v1, v2)
    graph_draw(G, vertex_text=G.vertex_index, pen_width=p_flat, mplfig=ax)


def plot_roc_curve(ax, true_theta, est_theta, d, method):
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
    fpr, tpr, thresholds = roc_curve(true_labels, scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label='{} (AUC={:.2f})'.format(method, roc_auc))

    return roc_auc


# loop through files, and plot graphs
vnce_aucs = {}
nce_means_aucs = {}
nce_noise_aucs = {}

sns.set_style("darkgrid")
graph_fig, graph_axs = plt.subplots(10, 3, figsize=(5.7, 19))
roc_fig, roc_axs = plt.subplots(5, 2, figsize=(5.7, 14.25))
roc_axs = roc_axs.ravel()

sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]

for i, file in enumerate(sorted_dirs):
    load = os.path.join(load_dir, file)
    config = pickle.load(open(os.path.join(load, 'config.p'), 'rb'))
    # reg_param = config.reg_param
    frac = float(config.frac_missing)
    d = config.d

    globals().update(np.load(os.path.join(load, 'theta0_and_theta_true.npz')))
    globals().update(np.load(os.path.join(load, 'vnce_results.npz')))
    globals().update(np.load(os.path.join(load, 'nce_filled_in_means_results.npz')))
    globals().update(np.load(os.path.join(load, 'nce_filled_in_noise_results.npz')))

    # print('reg param: {} \n frac missing: {}'.format(reg_param, frac))
    print('vnce mse: {}'.format(vnce_mse))
    print('nce (zeros) mse: {}'.format(nce_missing_mse))
    print('nce (noise) mse: {}'.format(nce_missing_mse_2))

    vnce_theta = vnce_thetas[-1][-1]
    nce_means_theta = nce_means_thetas[-1]
    nce_noise_theta = nce_noise_thetas[-1]

    axs = graph_axs[i]
    plot_graph(axs[0], vnce_theta, d)
    plot_graph(axs[1], nce_means_theta, d)
    plot_graph(axs[2], nce_noise_theta, d)
    titles = ['vnce', 'nce (means)', 'nce (noise)']
    for j, ax in enumerate(axs):
        ax.set_title(titles[j])
        ax.axis('off')

    ax = roc_axs[i]
    vnce_aucs[str(frac)] = plot_roc_curve(ax, theta_true, vnce_theta, d, 'VNCE')
    nce_means_aucs[str(frac)] = plot_roc_curve(ax, theta_true, nce_means_theta, d, 'NCE (means)')
    nce_noise_aucs[str(frac)] = plot_roc_curve(ax, theta_true, nce_noise_theta, d, 'NCE (noise)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve: {} missing'.format(frac))
    ax.legend(loc='best')

    # print('true_theta:', format(theta_true[config.reg_param_indices]))
    print("--------------{}------------------".format(frac))
    #print('vnce\n', vnce_theta[config.reg_param_indices])
    print('vnce\n', vnce_theta[d + 1:])
    #print(vnce_theta[:d + 1])
    #print('nce (means)\n', nce_means_theta[config.reg_param_indices])
    print('nce (means)\n', nce_means_theta[d + 1:])
    # print(nce_means_theta[:d + 1])
    # print('nce (noise)\n', nce_noise_theta[config.reg_param_indices])
    print('nce (noise)\n', nce_noise_theta[d + 1:])
#     print(nce_noise_theta[:d + 1])

save_fig(graph_fig, save_dir, 'undirected_graphs')
roc_fig.tight_layout()
save_fig(roc_fig, save_dir, 'roc_curves')

# Plot AUC for each fraction missing
fracs = np.arange(10) / 10
vnce_sorted_aucs = np.array([vnce_aucs[str(frac)] for frac in fracs])
nce_means_sorted_aucs = np.array([nce_means_aucs[str(frac)] for frac in fracs])
nce_noise_sorted_aucs = np.array([nce_noise_aucs[str(frac)] for frac in fracs])

aucs_fig, ax = plt.subplots(1, 1, figsize=(5.7, 5.7))
ax.plot(fracs, vnce_sorted_aucs, label='VNCE')
ax.plot(fracs, nce_means_sorted_aucs, label='NCE (means)')
ax.plot(fracs, nce_noise_sorted_aucs, label='NCE (noise)')

ax.set_xlabel('fraction missing')
ax.set_ylabel('AUC')
ax.legend(loc='best')
save_fig(aucs_fig, save_dir, 'auc_curves')
