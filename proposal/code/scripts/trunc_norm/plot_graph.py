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
# parser.add_argument('--load_dir', type=str, default='~/masters-project/ben-rhodes-masters-project/experimental-results/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='5d/reg0.01/', help='name of set of experiments this one belongs to')

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


# noinspection PyPep8Naming
def plot_graph(ax, theta, d):
    precision = get_lower_diag_matrix(d, theta)

    # to calculate edges, get indices of the d largest vals in the lower triangle of the precision matrix
    p_flat = precision[np.tril_indices(d, -1)]
    p_flat.sort()
    dth_largest = p_flat[-d]
    precision[np.triu_indices(d)] = p_flat[0]  # hack to help us calculate edges easiliy
    edges = np.where(precision >= dth_largest)

    G = gt.Graph(directed=False)
    for v1, v2 in zip(edges[0], edges[1]):
        G.add_edge(v1, v2)
    graph_draw(G, vertex_text=G.vertex_index, mplfig=ax)


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
    ax.plot(fpr, tpr, label='{} (AUC={})'.format(method, roc_auc))


# loop through files, and plot graphs
for i, file in enumerate(os.listdir(load_dir)):
    load = os.path.join(load_dir, file)
    config = pickle.load(open(os.path.join(load, 'config.p'), 'rb'))
    reg_param = config.reg_param
    frac = config.frac_missing
    d = config.d

    globals().update(np.load(os.path.join(load, 'theta0_and_theta_true.npz')))
    globals().update(np.load(os.path.join(load, 'vnce_results.npz')))
    globals().update(np.load(os.path.join(load, 'nce_filled_in_means_results.npz')))
    globals().update(np.load(os.path.join(load, 'nce_filled_in_noise_results.npz')))

    print('reg param: {} \n frac missing: {}'.format(reg_param, frac))
    print('vnce mse: {}'.format(vnce_mse))
    print('nce (zeros) mse: {}'.format(nce_missing_mse))
    print('nce (noise) mse: {}'.format(nce_missing_mse_2))

    vnce_theta = vnce_thetas[-1][-1]
    nce_means_theta = nce_means_thetas[-1]
    nce_noise_theta = nce_noise_thetas[-1]

    sns.set_style("darkgrid")
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2))
    axs = axs.ravel()
    plot_graph(axs[0], vnce_theta, d)
    plot_graph(axs[1], nce_means_theta, d)
    plot_graph(axs[2], nce_noise_theta, d)
    titles = ['vnce', 'nce (means)', 'nce (noise)']
    for j, ax in enumerate(axs):
        ax.set_title(titles[j])
        ax.axis('off')
    save_fig(fig, save_dir, 'fraction_missing:{}'.format(frac))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    plot_roc_curve(ax, theta_true, vnce_theta, d, 'VNCE')
    plot_roc_curve(ax, theta_true, nce_means_theta, d, 'NCE (means)')
    plot_roc_curve(ax, theta_true, nce_noise_theta, d, 'NCE (noise)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic curve')
    ax.legend(loc='best')
    save_fig(fig, save_dir, 'roc_curve_frac_miss:{}'.format(frac))

    # print('true_theta:', format(theta_true[config.reg_param_indices]))
    print('vnce\n', vnce_theta[config.reg_param_indices])
    print('nce (means)\n', nce_means_theta[config.reg_param_indices])
    print('nce (noise)\n', nce_noise_theta[config.reg_param_indices])
