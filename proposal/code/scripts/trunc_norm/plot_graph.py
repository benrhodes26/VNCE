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
parser.add_argument('--exp_name', type=str, default='5d-vlr0.1-nz=10-final/18/best/', help='name of set of experiments this one belongs to')

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

# def convert_weights(weights):
#     min = weights.min()
#     max = weights.max()
#     weights = (weights - min / (max - min))**5
#     weights *= (max - min)
#     weights += min
#     return weights

# noinspection PyPep8Naming
def plot_graph(ax, theta, d, v_size=15, font_size=12):
    precision = get_lower_diag_matrix(d, theta)

    # to calculate edges, get indices of the d largest vals in the lower triangle of the precision matrix
    p_flat = precision[np.tril_indices(d, -1)]
    p_flat.sort()
    dth_largest = p_flat[-d]
    precision[np.triu_indices(d)] = p_flat[0]  # hack to help us calculate edges easiliy
    edges = np.where(precision >= dth_largest)

    # p_flat = np.abs(p_flat / np.max(p_flat))
    # p_flat = convert_weights(np.abs(p_flat))
    # edges = np.tril_indices(d, -1)

    G = gt.Graph(directed=False)
    G.add_edge_list([(v1, v2) for v1, v2 in zip(edges[0], edges[1])])

    pos = gt.draw.sfdp_layout(G)
    x, y = gt.draw.ungroup_vector_property(pos, [0, 1])
    x.a = (x.a - x.a.min()) / (x.a.max() - x.a.min()) * 0.75 + 0.15
    y.a = (y.a - y.a.min()) / (y.a.max() - y.a.min()) * 0.75 + 0.15
    pos = gt.draw.group_vector_property([x, y])
    # G.add_edge(v1, v2)
    graph_draw(G,
               pos,
               output_size=(250, 250),
               vertex_fill_color='blue',
               vertex_size=v_size,
               vertex_text=G.vertex_index,
               vertex_font_size=font_size,
               mplfig=ax,
               fit_view=False)


sns.set_style("darkgrid")
graph_fig, graph_axs = plt.subplots(5, 3, figsize=(5.7, 9.5))
single_graph_fig, single_graph_axs = plt.subplots(1, 1, figsize=(4, 4))

sorted_fracs = sorted([float(f[4:]) for f in os.listdir(load_dir)])
sorted_dirs = ['frac' + str(frac) for frac in sorted_fracs]

for i, file in enumerate(sorted_dirs[::2]):
    load = os.path.join(load_dir, file)
    config = pickle.load(open(os.path.join(load, 'config.p'), 'rb'))
    # reg_param = config.reg_param
    frac = float(config.frac_missing)
    d = config.d

    loaded = np.load(os.path.join(load, 'theta0_and_theta_true.npz'))
    loaded1 = np.load(os.path.join(load, 'vnce_results1.npz'))
    loaded2 = np.load(os.path.join(load, 'vnce_results2.npz'))
    loaded3 = np.load(os.path.join(load, 'nce_results1.npz'))
    # loaded4 = np.load(os.path.join(load, 'nce_results2.npz'))
    # loaded5 = np.load(os.path.join(load, 'nce_results3.npz'))

    theta_true = loaded['theta_true']
    vnce_theta1 = loaded1['vnce_thetas'][-1][-1]
    vnce_theta2 = loaded2['vnce_thetas'][-1][-1]
    nce_means_theta = loaded3['nce_thetas'][-1]
    # nce_noise_theta = loaded4['nce_thetas'][-1]
    # nce_rnd_theta = loaded5['nce_thetas'][-1]

    axs = graph_axs[i]
    plot_graph(axs[0], vnce_theta1, d)
    plot_graph(axs[1], vnce_theta2, d)
    plot_graph(axs[2], nce_means_theta, d)

    if i == 0:
        plot_graph(single_graph_axs, vnce_theta1, d, v_size=25, font_size=16)

    # turn off axes
    for j, ax in enumerate(axs):
        ax.axis('off')

# add label to each row
rows = ['0%', '20%', '40%', '60%', '80%']
rows = rows[::-1]
pad = -12  # in points
for ax, row in zip(graph_axs[:, 0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                textcoords='offset points', size='large', ha='right', va='center', fontweight='heavy')
# add titles
titles = ['VNCE (true)', 'VNCE (approx)', 'NCE (means)']
for ax, title in zip(graph_axs[0], titles):
    ax.set_title(title, fontweight='heavy')
save_fig(graph_fig, save_dir, 'undirected_graphs')

single_graph_axs.axis('off')
save_fig(single_graph_fig, save_dir, 'ground truth graph')


