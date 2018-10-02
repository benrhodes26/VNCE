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

rc('lines', linewidth=1)
rc('font', size=14)
rc('legend', fontsize=12)
#   rc('text', usetex=True)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)

load_dir = '/disk/scratch/ben-rhodes-masters-project/experimental-results/mog/500_runs/sample_sizes'
save_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project-non-code/figs'

vnce_quantile_to_run_dict = pickle.load(open(os.path.join(load_dir, 'vnce_quantile_to_run_dict.p'), 'rb'))
nce_quantile_to_run_dict = pickle.load(open(os.path.join(load_dir, 'nce_quantile_to_run_dict.p'), 'rb'))
mle_quantile_to_run_dict = pickle.load(open(os.path.join(load_dir, 'mle_quantile_to_run_dict.p'), 'rb'))

sample_sizes = np.array([100, 500, 2500, 12500])
sns.set_style('darkgrid')
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.5), sharex=True, sharey=True)
axs = axs.ravel()
ax1 = axs[0]
plot1(ax1, vnce_quantile_to_run_dict, sample_sizes, colour='r', label='VNCE', marker='o', markersize=3)
plot1(ax1, nce_quantile_to_run_dict, sample_sizes, colour='b', label='NCE', marker='^', markersize=3)
plot1(ax1, mle_quantile_to_run_dict, sample_sizes, colour='k', label='MLE', marker='s', markersize=3)
ax1.grid('on')
ax1.set_title(r'$\theta$')

# PLOT 2: log_10 sample size against log_10 mean-squared error
ax2 = axs[1]
plot2(ax2, vnce_quantile_to_run_dict, sample_sizes, colour='r', label='VNCE', marker='o', markersize=3)
plot2(ax2, nce_quantile_to_run_dict, sample_sizes, colour='b', label='NCE', marker='^', markersize=3)
ax2.grid('on')
ax2.set_title(r'$c$')

# Reserve space for common axis labels
axs[-1].set_xlabel('.', color=(0, 0, 0, 0))
axs[-1].set_ylabel('.', color=(0, 0, 0, 0))
# plot common axis labels
fig.text(0.5, 0.02, r'$log(10)$ sample size', ha='center')
fig.text(0.0, 0.5, r'$log(10)$ MSE', va='center', rotation='vertical')
fig.tight_layout()
save_fig(fig, save_dir, 'mog-sample-size-against-mse')
