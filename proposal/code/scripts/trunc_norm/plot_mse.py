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

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
import matplotlib as matplotlib
from plot import *
from utils import take_closest

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=8)
rc('text', usetex=True)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)

parser = ArgumentParser(description='plot relationship between fraction of training data missing and final mean-squared error for'
                                    'a truncated normal model trained with VNCE', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project/ben-rhodes-masters-project/proposal/experiments/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='5d/', help='name of set of experiments this one belongs to')
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm/')
# parser.add_argument('--load_dir', type=str, default='~/masters-project/ben-rhodes-masters-project/experimental-results/trunc-norm/')

args = parser.parse_args()
load_dir = os.path.join(args.load_dir, args.exp_name)
load_dir = os.path.expanduser(load_dir)
save_dir = os.path.join(args.save_dir, args.exp_name)
save_dir = os.path.expanduser(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

frac_to_mse_dict = {}
nce_mse = None
# loop through files and get mses of vnce and nce with 0s
for i, file in enumerate(os.listdir(load_dir)):
    exp = os.path.join(load_dir, file)
    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
    frac = config.frac_missing

    loaded = np.load(os.path.join(exp, 'theta0_and_theta_true.npz'))
    vnce_mse = loaded['vnce_mse']
    nce_missing_mse = loaded['nce_missing_mse']  # missing data filled-in with zeros
    nce_missing_mse_2 = loaded['nce_missing_mse_2']  # missing data filled-in with zeros
    frac_to_mse_dict[str(frac)] = [vnce_mse, nce_missing_mse, nce_missing_mse_2]
    if file == 'frac0':
        nce_mse = loaded['nce_mse']

fracs = sorted([float(key) for key in frac_to_mse_dict.keys()])
vnce_mses = [frac_to_mse_dict[str(frac)][0] for frac in fracs]
nce_missing_mses = [frac_to_mse_dict[str(frac)][1] for frac in fracs]
nce_missing_mses_2 = [frac_to_mse_dict[str(frac)][2] for frac in fracs]

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(np.array(fracs), np.array(vnce_mses), label='VNCE')
ax.plot(np.array(fracs), np.array(nce_missing_mses), label='NCE (zeros fill in)')
ax.plot(np.array(fracs), np.array(nce_missing_mses_2), label='NCE (noise fill in)')
ax.plot((0, 1), (0.99782, 0.99782), label='Initial params')
ax.set_xlabel('fraction of data missing')
ax.set_ylabel('MSE')
ax.legend(loc='best')
save_fig(fig, save_dir, 'fraction_missing_vs_mse')
