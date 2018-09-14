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

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from numpy import random as rnd
from scipy.io import loadmat
from scipy.optimize import fsolve
from scipy.special import erfcx
from shutil import copyfile
from time import gmtime, strftime

from distribution import MissingDataProductOfTruncNormsPosterior, UnivariateTruncNormTruePosteriors, MissingDataProductOfTruncNormNoise
from fully_observed_models import UnnormalisedTruncNorm
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from nce_optimiser import NCEOptimiser
from plot import *
from regularisers import L1Regulariser
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, ScipyMinimiseEmStep, ExactEStep, MonteCarloVnceLoss

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments

# parser.add_argument('--save_dir', type=str, default='/home/ben/masters-project-non-code/experimental-results/trunc-norm',
#                     help='Path to directory where model will be saved')
parser.add_argument('--save_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm',
                    help='Path to directory where model will be saved')
parser.add_argument('--exp_name', type=str, default='5d-vlr0.1-nz=10-final', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default='cross-val', help='name of this exact experiment')

# Data arguments
parser.add_argument('--n', type=int, default=1000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=10, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=int, default=1, help='ratio of noise to data samples in NCE')
parser.add_argument('--load_data', dest='load_data', action='store_true', help='load 100d data generated in matlab')
parser.add_argument('--no-load_data', dest='load_data', action='store_false')
parser.set_defaults(load_data=False)

# Model arguments
parser.add_argument('--d', type=int, default=5, help='dimension of visibles for synthetic dataset')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers if using a neural network to parametrise variational distribution')
parser.add_argument('--hidden_dim', type=int, default=100, help='dimension of hidden layer in neural network')
parser.add_argument('--activation_layer', type=object, default=TanhLayer(), help='type of non-linearity in neural network')

# Latent NCE optimisation arguments
parser.add_argument('--opt_method', type=str, default='SGD', help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter', type=int, default=5, help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=0, help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_epochs1', type=int, default=500, help='Maximum number of loops through the dataset during training')
parser.add_argument('--max_num_epochs2', type=int, default=500, help='Maximum number of loops through the dataset during training')
parser.add_argument('--model_learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used to train the model')
parser.add_argument('--var_learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used to train the variational dist')
parser.add_argument('--batch_size', type=int, default=100, help='if opt_method=SGD, this is the size of a minibatch')
parser.add_argument('--num_batch_per_em_step', type=int, default=1, help='if opt_method=SGD, this is the number of batches per EM step')
parser.add_argument('--track_loss', dest='track_loss', action='store_true', help='track VNCE loss in E & M steps')
parser.add_argument('--no-track_loss', dest='track_loss', action='store_false')
parser.set_defaults(track_loss=True)

# nce optimisation arguments
parser.add_argument('--nce_opt_method', type=str, default='SGD', help='nce optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter_nce', type=int, default=500, help='number of iterations inside scipy.minimize')
parser.add_argument('--nce_missing_num_epochs', type=int, default=250, help='if nce_opt_method=SGD, this is the number of passes through data set')
parser.add_argument('--nce_learn_rate', type=float, default=0.1, help='if nce_opt_method=SGD, this is the learning rate used')
parser.add_argument('--nce_batch_size', type=int, default=100, help='if nce_opt_method=SGD, this is the size of a minibatch')

# Other arguments
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=False)
parser.add_argument('--random_seed', type=int, default=1083463236, help='seed for np.random.RandomState')

args = parser.parse_args()
args.rng = rnd.RandomState(args.random_seed)
if args.opt_method == 'SGD':
    args.num_em_steps_per_epoch = args.n / (args.batch_size * args.num_batch_per_em_step)
else:
    args.num_em_steps_per_epoch = 1
args.max_num_em_steps1 = args.max_num_epochs1 * args.num_em_steps_per_epoch * 2
args.max_num_em_steps2 = args.max_num_epochs2 * args.num_em_steps_per_epoch


def calculate_mse(args):
    args.init_mse = get_mse(args, args.theta0[1:], 'Initial')
    args.vnce_mse1 = get_mse(args, args.vnce_loss1.model.theta[1:], 'Missing Data VNCE1')
    args.vnce_mse2 = get_mse(args, args.vnce_loss2.model.theta[1:], 'Missing Data VNCE2')
    args.nce_mse1 = get_mse(args, args.nce_missing_model.theta[1:], 'filled-in means NCE')
    args.nce_mse2 = get_mse(args, args.nce_missing_model_2.theta[1:], 'filled-in noise NCE')
    args.nce_mse3 = get_mse(args, args.nce_missing_model_3.theta[1:], 'filled-in random NCE')

    print('vnce1 final scaling parameter: {}'.format(args.vnce_loss1.model.theta[0]))
    print('vnce2 final scaling parameter: {}'.format(args.vnce_loss2.model.theta[0]))
    print('nce filled in with means final scaling parameter: {}'.format(args.nce_missing_model.theta[0]))
    print('nce filled in with noise final scaling parameter: {}'.format(args.nce_missing_model_2.theta[0]))
    print('nce filled in with noise final scaling parameter: {}'.format(args.nce_missing_model_3.theta[0]))


def get_mse(args, theta, method_name):
    mse = mean_square_error(args.theta_true[1:], theta)
    print('{} Mean Squared Error: {}'.format(method_name, mse))
    return mse


def update_best_loss(best_vnce1, frac_missing, vnce_val_loss1):
    if vnce_val_loss1 > best_vnce1[str(frac_missing)][1]:
        best_vnce1[str(frac_missing)][1] = vnce_val_loss1


def cross_validate(args, frac_missing, i, reg_param):
    best_vnce1 = args.best_vnce_true
    best_vnce2 = args.best_vnce_approx
    best_nce1 = args.best_nce_means
    best_nce2 = args.best_nce_noise
    best_nce3 = args.best_nce_rnd

    _, vnce_val_loss1 = args.vnce_loss1.compute_end_of_epoch_loss()
    _, vnce_val_loss2 = args.vnce_loss2.compute_end_of_epoch_loss()

    nce_val_data1 = args.X_val * (1 - args.val_missing_data_mask) + args.X_val_sample_mean * args.val_missing_data_mask
    nce_val_loss1 = args.nce_missing_optimiser.compute_J(nce_val_data1)

    num_val_noise = int(args.nu * len(args.X_val))
    nce_val_data2 = args.X_val * (1 - args.val_missing_data_mask) + args.Y[:num_val_noise] * args.val_missing_data_mask
    nce_val_loss2 = args.nce_missing_optimiser_2.compute_J(nce_val_data2)

    rnd_fill = rnd.uniform(0, 3, args.X_val.shape)
    nce_val_data3 = args.X_val * (1 - args.val_missing_data_mask) + rnd_fill * args.val_missing_data_mask
    nce_val_loss3 = args.nce_missing_optimiser_3.compute_J(nce_val_data3)

    if i == 0:
        best_vnce1[str(frac_missing)] = [reg_param, vnce_val_loss1]
        best_vnce2[str(frac_missing)] = [reg_param, vnce_val_loss2]
        best_nce1[str(frac_missing)] = [reg_param, nce_val_loss1]
        best_nce2[str(frac_missing)] = [reg_param, nce_val_loss2]
        best_nce3[str(frac_missing)] = [reg_param, nce_val_loss3]

    else:
        update_best_loss(best_vnce1, frac_missing, vnce_val_loss1)
        update_best_loss(best_vnce2, frac_missing, vnce_val_loss2)
        update_best_loss(best_nce1, frac_missing, nce_val_loss1)
        update_best_loss(best_nce2, frac_missing, nce_val_loss2)
        update_best_loss(best_nce3, frac_missing, nce_val_loss3)


def plot_and_save_vnce(save, vnce_optimiser, vnce_loss, vnce_mse, which):

    vnce_thetas, vnce_alphas, vnce_losses, vnce_times = vnce_optimiser.get_flattened_result_arrays(
        flatten_params=False)
    m_step_ids, e_step_ids, m_step_start_ids, e_step_start_ids = vnce_optimiser.get_m_and_e_step_ids()
    vnce_train_losses = np.array(vnce_optimiser.train_losses)
    vnce_val_losses = np.array(vnce_optimiser.val_losses)

    vnce_plot = plot_vnce_loss(vnce_times, vnce_train_losses, vnce_val_losses)
    save_fig(vnce_plot, save, 'vnce_loss{}'.format(which))

    np.savez(os.path.join(save, "vnce_results{}".format(which)),
             vnce_mse=vnce_mse,
             vnce_thetas=vnce_thetas,
             vnce_times=vnce_times,
             vnce_losses=vnce_losses,
             vnce_val_losses=vnce_val_losses,
             m_step_ids=m_step_ids,
             e_step_ids=e_step_ids,
             m_step_start_ids=m_step_start_ids,
             e_step_start_ids=e_step_start_ids)

    pickle.dump(vnce_loss.model, open(os.path.join(save, "vnce_model{}.p".format(which)), "wb"))
    pickle.dump(vnce_loss.noise, open(os.path.join(save, "noise{}.p".format(which)), "wb"))
    pickle.dump(vnce_loss.variational_noise, open(os.path.join(save, "var_dist{}.p".format(which)), "wb"))
    pickle.dump(vnce_alphas, open(os.path.join(save, "alphas{}.p".format(which)), "wb"))


def plot_nce_results(args, save):
    nce_plot, axs = plt.subplots(1, 3, figsize=(5.5, 5))
    axs = axs.ravel()
    axs[0].plot(args.nce_missing_optimiser.times, args.nce_missing_optimiser.Js, c='r', label='NCE filled-in means (train)')
    axs[1].plot(args.nce_missing_optimiser_2.times, args.nce_missing_optimiser_2.Js, c='g', label='NCE filled-in noise (train)')
    axs[2].plot(args.nce_missing_optimiser_3.times, args.nce_missing_optimiser_3.Js, c='b', label='NCE filled-in random (train)')
    nce_plot.legend()
    save_fig(nce_plot, save, 'nce_losses')


def save_nce(save, nce_optimiser, nce_model, nce_mse, which):
    np.savez(os.path.join(save, "nce_results{}".format(which)),
             nce_mse=nce_mse,
             nce_thetas=nce_optimiser.thetas,
             nce_times=nce_optimiser.times,
             nce_losses=nce_optimiser.Js)
    pickle.dump(nce_model, open(os.path.join(save, "nce_model{}.p".format(which)), "wb"))


def plot_and_save_results(save_dir, args):
    save = os.path.join(save_dir, 'frac{}/reg{}'.format(str(args.frac_missing), str(args.reg_param)))
    if not os.path.exists(save):
        os.makedirs(save)

    plot_and_save_vnce(save, args.vnce_optimiser1, args.vnce_loss1, args.vnce_mse1, '1')
    plot_and_save_vnce(save, args.vnce_optimiser2, args.vnce_loss2, args.vnce_mse2, '2')

    plot_nce_results(args, save)
    save_nce(save, args.nce_missing_optimiser, args.nce_missing_model, args.nce_mse1, "1")
    save_nce(save, args.nce_missing_optimiser_2, args.nce_missing_model_2, args.nce_mse2, "2")
    save_nce(save, args.nce_missing_optimiser_3, args.nce_missing_model_3, args.nce_mse3, "3")

    with open(os.path.join(save, "config.txt"), 'w') as f:
        for key, value in vars(args).items():
            f.write("{}: {}\n".format(key, value))
    pickle.dump(args, open(os.path.join(save, "config.p"), "wb"))
    pickle.dump(args.data_dist, open(os.path.join(save, "data_dist.p"), "wb"))
    np.savez(os.path.join(save, "theta0_and_theta_true"),
             theta0=args.theta0,
             theta_true=args.theta_true,
             init_mse=args.init_mse)
    np.savez(os.path.join(save, "data"),
             X_train=args.vnce_loss1.train_data,
             X_train_mask=args.vnce_loss1.train_miss_mask,
             X_val=args.vnce_loss1.val_data,
             X_val_mask=args.vnce_loss1.val_miss_mask,
             Y=args.vnce_loss1.Y)

def copy_vnce(args, save, save_dir, best_vnce, which):
    args.reg_param = best_vnce[str(args.frac_missing)][0]
    load = os.path.join(save_dir, 'frac{}/reg{}'.format(str(args.frac_missing), str(args.reg_param)))
    pickle.dump(args, open(os.path.join(save, "config.p"), "wb"))

    copyfile(os.path.join(load, 'vnce_model{}.p'.format(which)), os.path.join(save, 'vnce_model{}.p'.format(which)))
    copyfile(os.path.join(load, 'vnce_results{}.npz'.format(which)), os.path.join(save, 'vnce_results{}.npz'.format(which)))
    copyfile(os.path.join(load, 'theta0_and_theta_true.npz'), os.path.join(save, 'theta0_and_theta_true.npz'))
    copyfile(os.path.join(load, 'config.p'), os.path.join(save, 'config.p'))

    with open(os.path.join(save, "reg_params.txt"), 'a') as f:
        f.write("vnce{}: {}\n".format(which, args.reg_param))


def copy_nce(args, save, save_dir, best_nce, which):
    args.reg_param = best_nce[str(args.frac_missing)][0]
    load = os.path.join(save_dir, 'frac{}/reg{}'.format(str(args.frac_missing), str(args.reg_param)))

    copyfile(os.path.join(load, 'nce_model{}.p'.format(which)), os.path.join(save, 'nce_model{}.p'.format(which)))
    copyfile(os.path.join(load, 'nce_results{}.npz'.format(which)), os.path.join(save, 'nce_results{}.npz'.format(which)))

    with open(os.path.join(save, "reg_params.txt"), 'a') as f:
        f.write("nce{}: {}\n".format(which, args.reg_param))


def save_best_results(args, save_dir):
    best_vnce1 = args.best_vnce_true
    best_vnce2 = args.best_vnce_approx
    best_nce1 = args.best_nce_means
    best_nce2 = args.best_nce_noise
    best_nce3 = args.best_nce_rnd
    for frac_missing in args.frac_range:

        args.frac_missing = frac_missing
        save = os.path.join(save_dir, 'best/frac{}'.format(str(args.frac_missing)))
        if not os.path.exists(save):
            os.makedirs(save)

        copy_vnce(args, save, save_dir, best_vnce1, '1')
        copy_vnce(args, save, save_dir, best_vnce2, '2')

        copy_nce(args, save, save_dir, best_nce1, '1')
        copy_nce(args, save, save_dir, best_nce2, '2')
        copy_nce(args, save, save_dir, best_nce3, '3')


def compose_layers(args):
    """Return the layers for a multi-layer neural network according to the specified hyperparameters"""

    first_layer = [AffineLayer(args.d, args.hidden_dim, args.weights_init, args.biases_init),
                   args.activation_layer]

    hidden_layers = []
    for i in range(args.num_layers - 1):
        hidden_layers.append(AffineLayer(args.hidden_dim, args.hidden_dim, args.weights_init, args.biases_init))
        hidden_layers.append(args.activation_layer)

    final_layer = [AffineLayer(args.hidden_dim, args.d * 2, args.weights_init, args.biases_init)]

    return first_layer + hidden_layers + final_layer


def estimate_trunc_norm_params(sample_mean, sample_var):
    """Return the estimated mean and std of the pre-truncated normals that, after truncation, fits each dimension of the data """
    d = len(sample_mean)

    def trunc_norm_param_residuals(params):
        mean = params[:d]
        chol = params[d:]

        alpha = -mean * chol
        erf_term = 1 / erfcx(alpha / 2**0.5)
        std = 1 / chol
        var = std**2
        const = (2 / np.pi)**0.5

        trunc_mean = mean + const * erf_term * std
        trunc_var = var * (1 + const * alpha * erf_term - (const * erf_term)**2)

        mean_res = sample_mean - trunc_mean  # (d, )
        var_res = sample_var - trunc_var  # (d, )
        return np.concatenate((mean_res, var_res))

    guess = np.ones(2 * d)
    sol = fsolve(trunc_norm_param_residuals, guess)

    # sol = res.x
    trunc_mean = sol[:d]
    trunc_chol = sol[d:]
    return trunc_mean, trunc_chol


def load_data(args):
    tnorm_data = loadmat('/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project-non-code/data/trunc_norm/trandn_result.mat')
    res = tnorm_data['total_result'].T

    args.d = 100
    args.n = 2000
    args.X_train = res[:args.n]
    args.X_val = res[args.n:]

    # parameters that generated to the loaded data
    args.true_mean = np.ones(args.d)
    a = np.diag(np.ones(args.d))
    b = np.diag(np.ones(args.d - 1), 1) * 0.1
    c = np.diag(np.ones(args.d - 1), -1) * 0.1
    args.true_prec = a + b + c
    args.true_prec[0, -1] = 0.1
    args.true_prec[-1, 0] = 0.1
    args.data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=args.true_mean,
                                                      precision=args.true_prec, rng=args.rng)
    args.theta_true = deepcopy(args.data_dist.theta)


def generate_data(args):
    rng = args.rng

    # initialise ground-truth parameters for data generating distribution
    args.true_mean = np.ones(args.d)
    # args.true_prec = np.zeros((args.d, args.d))
    # args.true_prec[:3, :3] = rnd.uniform(0.1, 0.5, (3, 3))
    # args.true_prec[3:6, 3:6] = rnd.uniform(0.1, 0.5, (3, 3))
    # args.true_prec[6:9, 6:9] = rnd.uniform(0.1, 0.5, (3, 3))
    # args.true_prec[3, 2] = rnd.uniform(0.1, 0.5)
    # args.true_prec[6, 2] = rnd.uniform(0.1, 0.5)
    # args.true_prec[6, 3] = rnd.uniform(0.1, 0.5)
    # args.true_prec[np.triu_indices(args.d, 1)] = 0
    # args.true_prec += args.true_prec.T
    # args.true_prec[np.diag_indices(args.d)] = 1
    a = np.diag(np.ones(args.d))
    # b = np.diag(np.ones(args.d - 1), 1) * 0.4
    # c = np.diag(np.ones(args.d - 1), -1) * 0.4
    b = np.diag(rnd.uniform(0.3, 0.5, args.d - 1), 1)
    # c = np.diag(rnd.uniform(0.1, 0.5, args.d - 1), -1)
    args.true_prec = a + b + b.T
    args.true_prec[0, -1] = rnd.uniform(0.3, 0.5)
    args.true_prec[-1, 0] = deepcopy(args.true_prec[0, -1])
    print('true parameters: \n mean: {} \n precision: {}'.format(args.true_mean, args.true_prec))

    args.data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=args.true_mean, precision=args.true_prec, rng=rng)
    args.theta_true = deepcopy(args.data_dist.theta)

    # make synthetic data
    args.X_train = args.data_dist.sample(args.n)
    args.X_val = args.data_dist.sample(int(args.n / 5))


def generate_mask(args):
    args.train_missing_data_mask = args.rng.uniform(0, 1, args.X_train.shape) < args.frac_missing
    args.val_missing_data_mask = args.train_missing_data_mask[:len(args.X_val)]

    # calculate the sample mean and variance of non-missing data
    observed_train_data = args.X_train * (1 - args.train_missing_data_mask)
    masked_train = np.ma.masked_equal(observed_train_data, 0)
    observed_val_data = args.X_val * (1 - args.val_missing_data_mask)
    masked_val = np.ma.masked_equal(observed_val_data, 0)

    args.X_train_sample_mean = np.array(masked_train.mean(0))
    args.X_train_sample_diag_var = np.array(masked_train.var(0))  # ignore covariances
    args.X_val_sample_mean = np.array(masked_val.mean(0))
    args.X_val_sample_diag_var = np.array(masked_val.var(0))  # ignore covariances
    # print('sample mean: {} \n sample var: {}'.format(args.X_train_sample_mean, args.X_train_sample_diag_var))


def make_noise(args):
    # estimate (from the synthetic data) the parameters of a factorial truncated normal for the noise distribution
    noise_mean, noise_chol = estimate_trunc_norm_params(args.X_train_sample_mean, args.X_train_sample_diag_var)
    print('estimated noise parameters: \n mean: {} \n chol: {}'.format(noise_mean, noise_chol))

    # make noise distribution and noise samples for vnce
    args.noise = MissingDataProductOfTruncNormNoise(mean=noise_mean, chol=noise_chol, rng=args.rng)
    args.Y = args.noise.sample(int(args.n * args.nu))
    args.noise_miss_mask = np.repeat(args.train_missing_data_mask, args.nu, axis=0)
    args.noise_val_miss_mask = np.repeat(args.val_missing_data_mask, args.nu, axis=0)
    observed_noise = args.Y * (1 - args.noise_miss_mask)
    masked_noise = np.ma.masked_equal(observed_noise, 0)
    args.Y_means = np.array(masked_noise.mean(0))


def make_model(args):
    # initialise the model p(x, z)
    args.scale0 = np.array([0.])
    # args.mean0 = np.zeros(args.d).astype(float)
    args.mean0 = args.X_train_sample_mean
    args.prec0 = np.identity(args.d).astype(float)
    # args.prec0 = rnd.uniform(-0.5, 0.5, (args.d, args.d))
    # args.prec0[np.diag_indices(args.d)] = 1
    args.model = MissingDataUnnormalisedTruncNorm(scaling_param=args.scale0, mean=args.mean0, precision=args.prec0, rng=args.rng)
    args.theta0 = deepcopy(args.model.theta)


def make_var_dist(args):
    args.var_dist = UnivariateTruncNormTruePosteriors(mean=args.mean0, precision=args.prec0, rng=args.rng)


def make_approx_var_dist(args):
    args.weights_init = UniformInit(-0.00001, 0.00001, rng=args.rng)
    args.biases_init = ConstantInit(0)
    layers = [AffineLayer(args.d, 2 * args.d, args.weights_init, args.biases_init)]
    # layers = compose_layers(args)
    nn = MultipleLayerModel(layers)
    args.approx_var_dist = MissingDataProductOfTruncNormsPosterior(nn=nn, data_dim=args.d, rng=args.rng)


def make_regulariser(args):
    # initiliase L1 regulariser
    args.l1_reg = L1Regulariser(args.reg_param)
    # we only regularise the off-diagonal elements of the precision. It is slightly tricky to calculate the indices of these
    # elements within theta (the model's parameters). This is because the precision matrix is symmetric, and hence theta
    # only contains the lower triangular elements, which are flattened and concatened with the rest of the model's parameters.
    lprecision_diag_inds = [sum(np.arange(args.d + 1)[:i]) - 1 for i in range(1, args.d + 2)][1:]
    off_diag_inds = [i for i in range(max(lprecision_diag_inds)) if i not in lprecision_diag_inds]
    args.reg_param_indices = 1 + args.d + np.array(off_diag_inds)


def make_vnce_loss_functions(args):
    """Returns two VNCE loss functions, the first uses a true posterior, the second uses an approximate one."""
    make_noise(args)
    make_model(args)
    make_var_dist(args)
    make_approx_var_dist(args)
    make_regulariser(args)

    # create loss functions
    use_sgd = (args.opt_method == 'SGD')
    args.vnce_loss1 = MonteCarloVnceLoss(model=args.model,
                                         train_data=args.X_train,
                                         val_data=args.X_val,
                                         noise=args.noise,
                                         noise_samples=args.Y,
                                         variational_noise=args.var_dist,
                                         noise_to_data_ratio=args.nu,
                                         num_latent_per_data=args.nz,
                                         regulariser=args.l1_reg,
                                         reg_param_indices=args.reg_param_indices,
                                         use_neural_model=False,
                                         use_neural_variational_noise=False,
                                         train_missing_data_mask=args.train_missing_data_mask,
                                         val_missing_data_mask=args.val_missing_data_mask,
                                         noise_miss_mask=args.noise_miss_mask,
                                         X_means=args.X_train_sample_mean,
                                         Y_means=args.Y_means,
                                         use_minibatches=use_sgd,
                                         batch_size=args.batch_size,
                                         separate_terms=args.separate_terms,
                                         rng=args.rng)

    args.vnce_loss2 = MonteCarloVnceLoss(model=deepcopy(args.model),
                                         train_data=args.X_train,
                                         val_data=args.X_val,
                                         noise=args.noise,
                                         noise_samples=args.Y,
                                         variational_noise=args.approx_var_dist,
                                         noise_to_data_ratio=args.nu,
                                         num_latent_per_data=args.nz,
                                         regulariser=args.l1_reg,
                                         reg_param_indices=args.reg_param_indices,
                                         use_neural_model=False,
                                         use_neural_variational_noise=True,
                                         train_missing_data_mask=args.train_missing_data_mask,
                                         val_missing_data_mask=args.val_missing_data_mask,
                                         noise_miss_mask=args.noise_miss_mask,
                                         X_means=args.X_train_sample_mean,
                                         Y_means=args.Y_means,
                                         use_minibatches=use_sgd,
                                         batch_size=args.batch_size,
                                         use_reparam_trick=True,
                                         separate_terms=args.separate_terms,
                                         rng=args.rng)


def make_vnce_optimisers(args):
    m_step = SgdEmStep(do_m_step=True,
                       learning_rate=args.model_learn_rate,
                       num_batches_per_em_step=args.num_batch_per_em_step,
                       noise_to_data_ratio=args.nu,
                       track_loss=args.track_loss,
                       rng=args.rng)
    e_step1 = ExactEStep(track_loss=True)
    e_step2 = SgdEmStep(do_m_step=False,
                        learning_rate=args.var_learn_rate,
                        num_batches_per_em_step=args.num_batch_per_em_step,
                        noise_to_data_ratio=args.nu,
                        track_loss=args.track_loss,
                        rng=args.rng)

    args.vnce_optimiser1 = VemOptimiser(m_step=m_step, e_step=e_step1, num_em_steps_per_save=args.num_em_steps_per_epoch)
    args.vnce_optimiser2 = VemOptimiser(m_step=deepcopy(m_step), e_step=e_step2, num_em_steps_per_save=args.num_em_steps_per_epoch)


def train(args):

    make_vnce_loss_functions(args)
    make_vnce_optimisers(args)

    args.vnce_optimiser1.fit(loss_function=args.vnce_loss1,
                             theta0=deepcopy(args.vnce_loss1.model.theta),
                             alpha0=deepcopy(args.vnce_loss1.get_alpha()),
                             stop_threshold=args.stop_threshold,
                             max_num_em_steps=args.max_num_em_steps1)

    args.vnce_optimiser2.fit(loss_function=args.vnce_loss2,
                             theta0=deepcopy(args.vnce_loss2.model.theta),
                             alpha0=deepcopy(args.vnce_loss2.get_alpha()),
                             stop_threshold=args.stop_threshold,
                             max_num_em_steps=args.max_num_em_steps2)

    args.nce_missing_model = UnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.nce_missing_optimiser = NCEOptimiser(model=args.nce_missing_model,
                                              noise=args.noise,
                                              noise_samples=args.Y,
                                              regulariser=args.l1_reg,
                                              reg_param_indices=args.reg_param_indices,
                                              nu=args.nu)
    args.filled_in_with_mean_data = args.X_train * (1 - args.train_missing_data_mask) + args.X_train_sample_mean * args.train_missing_data_mask
    args.nce_missing_optimiser.fit(X=args.filled_in_with_mean_data,
                                   theta0=deepcopy(args.theta0),
                                   opt_method=args.nce_opt_method,
                                   maxiter=args.maxiter_nce,
                                   learning_rate=args.nce_learn_rate,
                                   batch_size=args.nce_batch_size,
                                   num_epochs=args.nce_missing_num_epochs)

    args.nce_missing_model_2 = UnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.nce_missing_optimiser_2 = NCEOptimiser(model=args.nce_missing_model_2,
                                                noise=args.noise,
                                                noise_samples=args.Y,
                                                regulariser=args.l1_reg,
                                                reg_param_indices=args.reg_param_indices,
                                                nu=args.nu)
    args.filled_in_with_noise_data = args.X_train * (1 - args.train_missing_data_mask) + args.Y * args.train_missing_data_mask
    args.nce_missing_optimiser_2.fit(X=args.filled_in_with_noise_data,
                                     theta0=deepcopy(args.theta0),
                                     opt_method=args.nce_opt_method,
                                     maxiter=args.maxiter_nce,
                                     learning_rate=args.nce_learn_rate,
                                     batch_size=args.nce_batch_size,
                                     num_epochs=args.nce_missing_num_epochs)

    args.nce_missing_model_3 = UnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.nce_missing_optimiser_3 = NCEOptimiser(model=args.nce_missing_model_3,
                                                noise=args.noise,
                                                noise_samples=args.Y,
                                                regulariser=args.l1_reg,
                                                reg_param_indices=args.reg_param_indices,
                                                nu=args.nu)
    rnd_fill = rnd.uniform(0, 3, args.X_train.shape)
    args.filled_in_with_rnd_data = args.X_train * (1 - args.train_missing_data_mask) + rnd_fill * args.train_missing_data_mask
    args.nce_missing_optimiser_3.fit(X=args.filled_in_with_rnd_data,
                                     theta0=deepcopy(args.theta0),
                                     opt_method=args.nce_opt_method,
                                     maxiter=args.maxiter_nce,
                                     learning_rate=args.nce_learn_rate,
                                     batch_size=args.nce_batch_size,
                                     num_epochs=args.nce_missing_num_epochs)


def main(args):
    if args.load_data:
        load_data(args)
    else:
        generate_data(args)

    for i in range(100):
        save_dir = os.path.join(args.save_dir, args.exp_name, str(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # todo: throw away all empty data!

        # for each value of fraction missing, search over regularisation param and pick best using cross-validation
        args.best_vnce_true = {}
        args.best_vnce_approx = {}
        args.best_nce_means = {}
        args.best_nce_noise = {}
        args.best_nce_rnd = {}

        args.frac_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # args.frac_range = [0.5]
        for frac_missing in args.frac_range:
            args.frac_missing = frac_missing
            generate_mask(args)
            # for i, reg_param in enumerate([0, 0.00001, 0.0001, 0.001]):
            for j, reg_param in enumerate([0]):
                args.reg_param = reg_param
                train(args)
                calculate_mse(args)
                cross_validate(args, frac_missing, j, reg_param)
                plot_and_save_results(save_dir, args)

        save_best_results(args, save_dir)


if __name__ == "__main__":
    main(args)
