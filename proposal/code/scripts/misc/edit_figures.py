import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
code_dir_3 = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dirs = [code_dir, code_dir_2, code_dir_3]
for code_dir in code_dirs:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

from utils import save_fig, change_fig_fontsize

import numpy as np
import os
import pickle
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import rcParams as rcp

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=10)
rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

########################################### MOG POP ANALYSIS ######################################################
mog_pop_load_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/mog/500_runs/sample_sizes/'
mog_pop_save_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/experiments/mog/population_analysis/'
fig1 = pickle.load(open(os.path.join(mog_pop_load_dir, 'fig1.p'), 'rb'))
fig2 = pickle.load(open(os.path.join(mog_pop_load_dir, 'fig2.p'), 'rb'))
fig1.set_size_inches(2.8, 2.8)
fig2.set_size_inches(2.8, 2.8)
change_fig_fontsize(fig1, 8)
change_fig_fontsize(fig2, 8)
fig2.gca().grid()
save_fig(fig1, mog_pop_save_dir, 'sample-size-against-mse')
save_fig(fig2, mog_pop_save_dir, 'sample-size-against-mse-scaling-param.pdf')

########################################### RBM USPS SGD ######################################################
rbm_usps_sgd_load_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/experiments/rbm/usps-sgd/'
rbm_usps_sgd_save_dir = rbm_usps_sgd_load_dir
fig = pickle.load(open(os.path.join(rbm_usps_sgd_load_dir, 'usps-different_num_hiddens.p'), 'rb'))
fig.set_size_inches(5.7, 5.7)
save_fig(fig, rbm_usps_sgd_save_dir, 'usps-different_num_hiddens')

########################################### RBM USPS SGD ######################################################
