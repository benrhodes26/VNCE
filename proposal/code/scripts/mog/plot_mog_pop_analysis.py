import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code2_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/scripts/mog'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code2_dir not in sys.path:
    sys.path.append(code2_dir)

import numpy as np
import pickle
import seaborn as sns
from matplotlib import rc
from mog_population_analysis import *
from plot import *

rc('lines', linewidth=0.5)
rc('font', size=8)
rc('legend', fontsize=8)
rc('text', usetex=True)
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)

load_dir = '/disk/scratch/ben-rhodes-masters-project/experimental-results/mog/500_runs/sample_sizes'
save_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project-non-code/figs'

vnce_quantile_to_run_dict = pickle.load(open(os.path.join(load_dir, 'vnce_quantile_to_run_dict.p'), 'rb'))
nce_quantile_to_run_dict = pickle.load(open(os.path.join(load_dir, 'nce_quantile_to_run_dict.p'), 'rb'))
mle_quantile_to_run_dict = pickle.load(open(os.path.join(load_dir, 'mle_quantile_to_run_dict.p'), 'rb'))

sample_sizes = np.array([100, 500, 2500, 12500])
sns.set_style('darkgrid')
fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5), sharex=True, sharey=True)
plot1(ax, vnce_quantile_to_run_dict, sample_sizes, colour='r', label='VNCE', marker='o', markersize=3)
plot1(ax, nce_quantile_to_run_dict, sample_sizes, colour='b', label='NCE', marker='^', markersize=3)
plot1(ax, mle_quantile_to_run_dict, sample_sizes, colour='k', label='MLE', marker='s', markersize=3)
ax.grid('on')
ax.set_title(r'$\theta$')

fig.text(0.5, 0.02, r'$\log(10)$ sample size', ha='center')
fig.text(0.0, 0.5, r'$\log(10)$ MSE', va='center', rotation='vertical')
fig.tight_layout()
fig.subplots_adjust(wspace=0.1)
save_fig(fig, save_dir, 'mog-sample-size-against-theta-mse')

# PLOT 2: log_10 sample size against log_10 mean-squared error
fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5), sharex=True, sharey=True)
plot2(ax, vnce_quantile_to_run_dict, sample_sizes, colour='r', label='VNCE', marker='o', markersize=3)
plot2(ax, nce_quantile_to_run_dict, sample_sizes, colour='b', label='NCE', marker='^', markersize=3)
ax.grid('on')
ax.set_title(r'$c$')

# Reserve space for common axis labels
# axs[-1].set_xlabel('.', color=(0, 0, 0, 0))
# axs[-1].set_ylabel('.', color=(0, 0, 0, 0))
# plot common axis labels
fig.text(0.5, 0.02, r'$\log(10)$ sample size', ha='center')
fig.text(0.0, 0.5, r'$\log(10)$ MSE', va='center', rotation='vertical')
fig.tight_layout()
fig.subplots_adjust(wspace=0.1)
save_fig(fig, save_dir, 'mog-sample-size-against-scaling-param-mse')
