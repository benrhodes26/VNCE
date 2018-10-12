import os
import sys
CODE_DIRS = ['~/masters-project/ben-rhodes-masters-project/proposal/code', '/home/s1771906/code']
CODE_DIRS_2 = [d + '/neural_network' for d in CODE_DIRS] + CODE_DIRS
CODE_DIRS_2 = [os.path.expanduser(d) for d in CODE_DIRS_2]
for code_dir in CODE_DIRS_2:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import numpy as np
import pickle

from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from plot import *
from project_statics import *
from shutil import copyfile
from utils import take_closest, mean_square_error

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from numpy import random as rnd

parser = ArgumentParser(description='Cross-validate l1-regularisation parameter for truncated normal experiments', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default=EXPERIMENT_OUTPUTS + '/trunc-norm/')
parser.add_argument('--exp_name', type=str, default='20d_reg_param_cir_n50/100', help='name of set of experiments this one belongs to')  # 5d-vlr0.1-nz=10-final
parser.add_argument('--load_dir', type=str, default=EXPERIMENT_OUTPUTS + '/trunc_norm/')
parser.add_argument('--separate_reg_param_results', dest='separate_reg_param_results', action='store_true', help='')
parser.add_argument('--no-separate_reg_param_results', dest='separate_reg_param_results', action='store_false')
parser.set_defaults(separate_reg_param_results=True)
args = parser.parse_args()


def make_nce_optimiser():

    nce_optimiser = NCEOptimiser(model=nce_missing_model,
                                 noise=noise,
                                 noise_samples=Y,
                                 regulariser=l1_reg,
                                 reg_param_indices=reg_param_indices,
                                 nu=nu)
    return nce_optimiser


def update_best_val_loss_if_larger(args, cross_val_dict, method, frac_missing, reg_param):
    val_loss = compute_val_loss(args, method)
    if val_loss > cross_val_dict[str(frac_missing)][1]:
        cross_val_dict[str(frac_missing)] = [reg_param, val_loss]


def compute_val_loss(args, method):
    # turn off regularisation before evaluating loss
    if method == 'vnce3':
        val_loss = compute_vnce_val_loss(args)
    elif method == 'nce1':
        args.nce_missing_optimiser.regulariser = None
        nce_val_data1 = args.X_val * (1 - args.val_missing_data_mask) + args.X_val_sample_mean * args.val_missing_data_mask
        val_loss = args.nce_missing_optimiser.compute_J(nce_val_data1)

    return val_loss


def compute_vnce_val_loss(args):
    args.vnce_loss3.model_regulariser = None

    num_latent_datasets = 100
    val_losses = np.zeros(num_latent_datasets, dtype='float64')
    for i in range(num_latent_datasets):
        # args.vnce_loss3.dp.reset_number_of_latent_samples(100)
        args.vnce_loss3.dp.sample_global_E()
        _, val_loss = args.vnce_loss3.compute_end_of_epoch_loss(print_loss=False)
        val_losses[i] = val_loss
        # prec = args.vnce_loss3.model.get_joint_pretruncated_params()[1]
        # print(prec)

    print("mean val loss:{} \n std:{}".format(np.mean(val_losses), np.std(val_losses)))
    return np.mean(val_losses)


def save_best_results(args, save_dir, load_dir, method, cross_val_dict):
    save = os.path.join(save_dir, 'frac{}'.format(str(args.frac_missing)))
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)

    if method == 'vnce3':
        copy_vnce(args, save, load_dir, cross_val_dict, '3')
    elif method == 'nce1':
        copy_nce(args, save, load_dir, cross_val_dict, '1')


def copy_vnce(args, save, load_dir, cross_val_dict, which):
    args.reg_param = cross_val_dict[str(args.frac_missing)][0]
    load = os.path.join(load_dir, 'frac{}/reg{}'.format(str(args.frac_missing), str(args.reg_param)))
    pickle.dump(args, open(os.path.join(save, "config.p"), "wb"))

    copyfile(os.path.join(load, 'vnce_model{}.p'.format(which)), os.path.join(save, 'vnce_model{}.p'.format(which)))
    copyfile(os.path.join(load, 'vnce_results{}.npz'.format(which)), os.path.join(save, 'vnce_results{}.npz'.format(which)))
    copyfile(os.path.join(load, 'theta0_and_theta_true.npz'), os.path.join(save, 'theta0_and_theta_true.npz'))
    copyfile(os.path.join(load, 'config.p'), os.path.join(save, 'config.p'))

    with open(os.path.join(save, "reg_params.txt"), 'w') as f:
        f.write("vnce{}: {}\n".format(which, args.reg_param))


def copy_nce(args, save, load_dir, best_nce, which):
    args.reg_param = best_nce[str(args.frac_missing)][0]
    load = os.path.join(load_dir, 'frac{}/reg{}'.format(str(args.frac_missing), str(args.reg_param)))

    copyfile(os.path.join(load, 'nce_model{}.p'.format(which)), os.path.join(save, 'nce_model{}.p'.format(which)))
    copyfile(os.path.join(load, 'nce_results{}.npz'.format(which)), os.path.join(save, 'nce_results{}.npz'.format(which)))

    with open(os.path.join(save, "reg_params.txt"), 'a') as f:
        f.write("nce{}: {}\n".format(which, args.reg_param))


def make_separated_copy(args, separated_dir, exp_dir, i, j, frac):
    if (i == 0) and args.separate_reg_param_results:
        dest = os.path.join(separated_dir, '{}/0/best/frac{}'.format(j, frac))
        if not os.path.exists(dest):
            os.makedirs(dest)
        for file in os.listdir(exp_dir):
            copyfile(os.path.join(exp_dir, file), os.path.join(dest, file))


load_dir = os.path.join(args.load_dir, args.exp_name)
load_dir = os.path.expanduser(load_dir)
save_dir = os.path.join(load_dir, 'best')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.separate_reg_param_results:
    separated_dir = os.path.join(load_dir, 'separated_results')

# methods = ['vnce']
methods = ['vnce3', 'nce1']
num_methods = len(methods)
# sorted_fracs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
sorted_fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
# sorted_fracs = np.array([0.4])
# reg_params = np.array([0, 0.00001, 0.0001, 0.001, 0.01])
# reg_params = np.array([0, 1, 10, 100, 1000, 10000])
reg_params = np.array([0.0, 1.0, 10.0, 100.0])
# reg_params = np.array([0, 0.00001, 0.0001, 0.001])

args.best_vnce_lognormal = {str(f):(0, -np.inf) for f in sorted_fracs}  # miss_frac: (reg_param, val_loss)
args.best_nce_means = {str(f):(0, -np.inf) for f in sorted_fracs}  # miss_frac: (reg_param, val_loss)
cross_val_dicts = [args.best_vnce_lognormal, args.best_nce_means]

# For each frac_missing and method, loop over all reg_params and keep the best (measured by validation loss)
for frac in sorted_fracs:
    for i, tup in enumerate(zip(methods, cross_val_dicts)):
        method, cross_val_dict = tup
        for j, reg in enumerate(reg_params):
            print('frac{}, method{} reg{}'.format(frac, method, reg))
            exp_dir = os.path.join(load_dir, 'frac{}/reg{}'.format(frac, reg))
            config = pickle.load(open(os.path.join(exp_dir, 'config.p'), 'rb'))
            # print("frac missing: ", config.frac_missing)
            update_best_val_loss_if_larger(config, cross_val_dict, method, frac, reg)
            make_separated_copy(args, separated_dir, exp_dir, i, j, frac)

        save_best_results(config, save_dir, load_dir, method, cross_val_dict)
