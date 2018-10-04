import os
import sys
CODE_DIRS = ['~/masters-project/ben-rhodes-masters-project/proposal/code', '/home/s1771906/code']
CODE_DIRS_2 = [d + '/neural_network' for d in CODE_DIRS] + CODE_DIRS
CODE_DIRS_2 = [os.path.expanduser(d) for d in CODE_DIRS_2]
for code_dir in CODE_DIRS_2:
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

from contrastive_divergence_optimiser import CDOptimiser
from data_provider import DataProvider
from distribution import MissingDataProductOfTruncNormsPosterior, UnivariateTruncNormTruePosteriors,\
    MissingDataProductOfTruncNormNoise, MissingDataLogNormalPosterior
from fully_observed_models import UnnormalisedTruncNorm
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm, MissingDataUnnormalisedTruncNormSymmetric
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from nce_optimiser import NCEOptimiser
from plot import *
from project_statics import *
from regularisers import L1Regulariser
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, ScipyMinimiseEmStep, ExactEStep, MonteCarloVnceLoss, ScipyOptimiser

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experiments for learning the parameters of a truncated normal from incomplete data with VNCE',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--save_dir', type=str, default=EXPERIMENT_OUTPUTS +'/trunc_norm', help='Path to directory where model will be saved')
parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')  # lognormal_50d_nu10
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')
parser.add_argument('--data_path', type=str, default=EXPERIMENT_OUTPUTS + '/trunc_norm/data', help='Number of datapoints')

# data arguments
parser.add_argument('--sample_size', type=int, default=1000, help='Number of datapoints')
parser.add_argument('--d', type=int, default=20, help='dimension of visibles for synthetic dataset')
parser.add_argument('--frac_missing', type=float, default=0.8, help='percentage of data missing completely at random')
parser.add_argument('--prec_type', type=str, default='circular', help='type of ground truth precision')
parser.add_argument('--nz', type=int, default=5, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=int, default=10, help='ratio of noise to data samples in NCE')

# Model / var_dist arguments

parser.add_argument('--num_layers', type=int, default=2, help='number of layers if using a neural network to parametrise variational distribution')
parser.add_argument('--hidden_dim', type=int, default=100, help='dimension of hidden layer in neural network')
parser.add_argument('--activation_layer', type=object, default=TanhLayer(), help='type of non-linearity in neural network')

# Latent NCE optimisation arguments
parser.add_argument('--reg_param', type=float, default=1000, help='l1 regularisation parameter')
parser.add_argument('--opt_method1', type=str, default='BFGS', help='optimisation method.')
parser.add_argument('--opt_method2', type=str, default='SGD', help='optimisation method.')
parser.add_argument('--opt_method3', type=str, default='BFGS', help='optimisation method.')
parser.add_argument('--maxiter1', type=int, default=0, help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--maxiter2', type=int, default=0, help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--maxiter3', type=int, default=5, help='number of iterations performed by L-BFGS-B optimiser')
parser.add_argument('--stop_threshold', type=float, default=0, help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_epochs1', type=int, default=0, help='Maximum number of loops through the dataset during training')  # 500
parser.add_argument('--max_num_epochs2', type=int, default=0, help='Maximum number of loops through the dataset during training')  # 500
parser.add_argument('--max_num_epochs3', type=int, default=10, help='Maximum number of loops through the dataset during training')  # 500
parser.add_argument('--model_learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used to train the model')
parser.add_argument('--var_learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used to train the variational dist')
parser.add_argument('--batch_size', type=int, default=50, help='if opt_method=SGD, this is the size of a minibatch')
parser.add_argument('--num_batch_per_em_step', type=int, default=1, help='if opt_method=SGD, this is the number of batches per EM step')
parser.add_argument('--track_loss', dest='track_loss', action='store_true', help='track VNCE loss in E & M steps')
parser.add_argument('--no-track_loss', dest='track_loss', action='store_false')
parser.set_defaults(track_loss=True)

# nce optimisation arguments
parser.add_argument('--nce_reg_param', type=float, default=0.01, help='l1 regularisation parameter')
parser.add_argument('--nce_opt_method', type=str, default='BFGS', help='nce optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter_nce1', type=int, default=100, help='number of iterations inside scipy.minimize')
parser.add_argument('--maxiter_nce2', type=int, default=0, help='number of iterations inside scipy.minimize')
parser.add_argument('--maxiter_nce3', type=int, default=0, help='number of iterations inside scipy.minimize')
parser.add_argument('--nce_missing_num_epochs1', type=int, default=100, help='if nce_opt_method=SGD, this is the number of passes through data set')  # 250
parser.add_argument('--nce_missing_num_epochs2', type=int, default=0, help='if nce_opt_method=SGD, this is the number of passes through data set')  # 250
parser.add_argument('--nce_missing_num_epochs3', type=int, default=0, help='if nce_opt_method=SGD, this is the number of passes through data set')  # 250
parser.add_argument('--nce_learn_rate', type=float, default=0.03, help='if nce_opt_method=SGD, this is the learning rate used')
parser.add_argument('--nce_batch_size', type=int, default=50, help='if nce_opt_method=SGD, this is the size of a minibatch')

# MLE optimisation arguments
parser.add_argument('--cd_num_gibbs_steps', type=int, default=10, help='Gibbs MCMC thinning factor')
parser.add_argument('--cd_learn_rate', type=float, default=0.01, help='learning rate for sampling-based MLE')
parser.add_argument('--cd_batch_size', type=int, default=50, help='batch size for sampling-based MLE')
parser.add_argument('--cd_num_epochs', type=int, default=100, help='number of epochs for sampling-based MLE')
parser.add_argument('--cd_nz', type=int, default=5, help='use nz samples for *data* expectation '
                                                         'and nz*n samples for model expectations when using sampling-based MLE')

# Other arguments
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=False)
parser.add_argument('--use_numeric_stable_approx_second_term', dest='use_numeric_stable_approx_second_term', action='store_true', help='')
parser.add_argument('--no-use_numeric_stable_approx_second_term', dest='use_numeric_stable_approx_second_term', action='store_false')
parser.set_defaults(use_numeric_stable_approx_second_term=False)
# parser.add_argument('--random_seed', type=int, default=1083463236, help='seed for np.random.RandomState')
parser.add_argument('--random_seed', type=int, default=20781, help='seed for np.random.RandomState')
parser.add_argument('--sim_num', type=int, default=100, help='id for this simulation')

args = parser.parse_args()
args.rng = rnd.RandomState(args.random_seed)


def calculate_mse(args):
    args.init_mse = get_mse(args, args.theta0[1:], 'Initial')
    args.vnce_mse1 = get_mse(args, args.vnce_loss1.model.theta[1:], 'Missing Data VNCE1')
    args.vnce_mse2 = get_mse(args, args.vnce_loss2.model.theta[1:], 'Missing Data VNCE2')
    args.vnce_mse3 = get_mse(args, args.vnce_loss3.model.theta[1:], 'Missing Data VNCE3')
    args.nce_mse1 = get_mse(args, args.nce_missing_model.theta[1:], 'filled-in means NCE')
    args.nce_mse2 = get_mse(args, args.nce_missing_model_2.theta[1:], 'filled-in noise NCE')
    args.nce_mse3 = get_mse(args, args.nce_missing_model_3.theta[1:], 'filled-in random NCE')
    args.cd_mse = get_mse(args, args.cd_model.theta[1:], 'MLE (sampling)')

    print('vnce1 final scaling parameter: {}'.format(args.vnce_loss1.model.theta[0]))
    print('vnce2 final scaling parameter: {}'.format(args.vnce_loss2.model.theta[0]))
    print('vnce3 final scaling parameter: {}'.format(args.vnce_loss3.model.theta[0]))
    print('nce filled in with means final scaling parameter: {}'.format(args.nce_missing_model.theta[0]))
    print('nce filled in with noise final scaling parameter: {}'.format(args.nce_missing_model_2.theta[0]))
    print('nce filled in with noise final scaling parameter: {}'.format(args.nce_missing_model_3.theta[0]))


def get_mse(args, theta, method_name):
    mse = mean_square_error(args.theta_true[1:], theta)
    print('{} Mean Squared Error: {}'.format(method_name, mse))
    return mse


def plot_and_save_vnce(save, vnce_optimiser, vnce_loss, vnce_mse, which):
    if repr(vnce_optimiser) == 'ScipyOptimiser':
        vnce_params, vnce_losses, vnce_times = vnce_optimiser.get_flattened_result_arrays()
        len_theta = len(vnce_loss.model.theta)
        vnce_thetas = vnce_params[:, :len_theta]
        vnce_alphas = vnce_params[:, len_theta:]
        m_step_ids, e_step_ids, m_step_start_ids, e_step_start_ids = None, None, None, None
    else:
        vnce_thetas, vnce_alphas, vnce_losses, vnce_times = vnce_optimiser.get_flattened_result_arrays(flatten_params=False)
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
    pickle.dump(vnce_loss.variational_dist, open(os.path.join(save, "var_dist{}.p".format(which)), "wb"))
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

def save_cd(save, cd_optimiser, cd_model, cd_mse):
    np.savez(os.path.join(save, "cd_results"),
             cd_mse=cd_mse,
             cd_thetas=cd_optimiser.thetas,
             cd_times=cd_optimiser.times)
    pickle.dump(cd_model, open(os.path.join(save, "cd_model.p"), "wb"))


def plot_and_save_results(save_dir, args):
    plot_and_save_vnce(save_dir, args.vnce_optimiser1, args.vnce_loss1, args.vnce_mse1, '1')
    plot_and_save_vnce(save_dir, args.vnce_optimiser2, args.vnce_loss2, args.vnce_mse2, '2')
    plot_and_save_vnce(save_dir, args.vnce_optimiser3, args.vnce_loss3, args.vnce_mse3, '3')

    plot_nce_results(args, save_dir)
    save_nce(save_dir, args.nce_missing_optimiser, args.nce_missing_model, args.nce_mse1, "1")
    save_nce(save_dir, args.nce_missing_optimiser_2, args.nce_missing_model_2, args.nce_mse2, "2")
    save_nce(save_dir, args.nce_missing_optimiser_3, args.nce_missing_model_3, args.nce_mse3, "3")

    save_cd(save_dir, args.cd_optimiser, args.cd_model, args.cd_mse)

    with open(os.path.join(save_dir, "config.txt"), 'w') as f:
        for key, value in vars(args).items():
            f.write("{}: {}\n".format(key, value))
    pickle.dump(args, open(os.path.join(save_dir, "config.p"), "wb"))
    pickle.dump(args.data_dist, open(os.path.join(save_dir, "data_dist.p"), "wb"))
    np.savez(os.path.join(save_dir, "theta0_and_theta_true"),
             theta0=args.theta0,
             theta_true=args.theta_true,
             init_mse=args.init_mse)
    np.savez(os.path.join(save_dir, "data"),
             X_train=args.data_provider1.train_data,
             X_train_mask=args.data_provider1.train_miss_mask,
             X_val=args.data_provider1.val_data,
             X_val_mask=args.data_provider1.val_miss_mask,
             Y=args.data_provider1.Y)


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
    """Return the estimated mean and std of the pre-truncated normals that, after truncation, fits each dimension of the data"""
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


def get_num_em_steps_per_epoch(args, opt_method, num_vnce='1'):
    if opt_method == 'SGD':
        num_em_steps_per_epoch = int(args.n / (args.batch_size * args.num_batch_per_em_step))
    else:
        num_em_steps_per_epoch = 2

    if num_vnce == '1':
        args.num_em_steps_per_epoch1 = num_em_steps_per_epoch
    if num_vnce == '2':
        args.num_em_steps_per_epoch2 = num_em_steps_per_epoch
    if num_vnce == '3':
        args.num_em_steps_per_epoch3 = num_em_steps_per_epoch

    return num_em_steps_per_epoch


def load_data(args):
    exact_data_path = os.path.join(args.data_path, "dim{}/{}/{}/frac{}/data.p".format(args.d, args.prec_type, args.sim_num, args.frac_missing))
    data_args = pickle.load(open(exact_data_path, 'rb'))
    vars(args).update(vars(data_args))
    pickle.dump(args, open(os.path.join(args.save_dir, "config.p"), "wb"))

    # set the number of EM steps to train for
    args.max_num_em_steps1 = args.max_num_epochs1 * get_num_em_steps_per_epoch(args, args.opt_method1, '1') * 2
    args.max_num_em_steps2 = args.max_num_epochs2 * get_num_em_steps_per_epoch(args, args.opt_method2, '2')
    args.max_num_em_steps3 = args.max_num_epochs3 * get_num_em_steps_per_epoch(args, args.opt_method3, '3')


def make_noise(args):
    # estimate (from the synthetic data) the parameters of a factorial truncated normal for the noise distribution
    args.noise_mean, args.noise_chol = estimate_trunc_norm_params(args.X_train_sample_mean, args.X_train_sample_diag_var)

    # make noise distribution and noise samples for vnce
    args.noise = MissingDataProductOfTruncNormNoise(mean=args.noise_mean, chol=args.noise_chol, rng=args.rng)
    args.Y = args.noise.sample(int(args.n * args.nu))
    args.noise_miss_mask = np.repeat(args.train_missing_data_mask, args.nu, axis=0)
    args.noise_val_miss_mask = np.repeat(args.val_missing_data_mask, args.nu, axis=0)
    observed_noise = args.Y * (1 - args.noise_miss_mask)
    masked_noise = np.ma.masked_equal(observed_noise, 0)
    args.Y_means = np.array(masked_noise.mean(0))


def make_model(args):
    # initialise the model p(x, z)
    args.scale0 = np.array([0.])
    # args.scale0 = np.array([-args.d])
    args.mean0 = np.zeros(args.d).astype(float)
    # args.prec0 = np.identity(args.d).astype(float)
    # args.mean0 = args.noise_mean
    args.prec0 = np.identity(args.d).astype(float)
    # args.prec0[np.diag_indices(args.d)] = args.noise_chol**2

    # args.prec0 = rnd.uniform(-0.5, 0.5, (args.d, args.d))
    # args.prec0[np.diag_indices(args.d)] = 1
    # args.model = MissingDataUnnormalisedTruncNorm(scaling_param=args.scale0, mean=args.mean0, precision=args.prec0, rng=args.rng)
    args.model = MissingDataUnnormalisedTruncNormSymmetric(scaling_param=args.scale0, mean=args.mean0, precision=args.prec0, rng=args.rng)
    args.theta0 = deepcopy(args.model.theta)


def make_var_dists(args):
    make_true_univariate_var_dist(args)
    make_approx_univariate_var_dist(args)
    make_approx_lognormal_var_dist(args)


def make_true_univariate_var_dist(args):
    args.var_dist = UnivariateTruncNormTruePosteriors(mean=args.mean0, precision=args.prec0, rng=args.rng)


def make_approx_univariate_var_dist(args):
    args.weights_init = UniformInit(-0.00001, 0.00001, rng=args.rng)
    args.biases_init = ConstantInit(0)
    layers = [AffineLayer(args.d, 2 * args.d, args.weights_init, args.biases_init)]
    # layers = compose_layers(args)
    nn = MultipleLayerModel(layers)
    args.approx_var_dist = MissingDataProductOfTruncNormsPosterior(nn=nn, data_dim=args.d, rng=args.rng)


def make_approx_lognormal_var_dist(args):
    var_mean0 = np.zeros(args.d).astype(float)
    var_prec0 = np.identity(args.d).astype(float)
    args.lognormal_var_dist = MissingDataLogNormalPosterior(mean=var_mean0, precision=var_prec0)


def make_regulariser(args):
    # initiliase L1 regulariser
    args.l1_reg = L1Regulariser(args.reg_param)
    args.nce_l1_reg = L1Regulariser(args.nce_reg_param)
    # we only regularise the off-diagonal elements of the precision. It is slightly tricky to calculate the indices of these
    # elements within theta (the model's parameters). This is because the precision matrix is symmetric, and hence theta
    # only contains the lower triangular elements, which are flattened and concatened with the rest of the model's parameters.
    lprecision_diag_inds = [sum(np.arange(args.d + 1)[:i]) - 1 for i in range(1, args.d + 2)][1:]
    off_diag_inds = [i for i in range(max(lprecision_diag_inds)) if i not in lprecision_diag_inds]
    args.reg_param_indices = 1 + args.d + np.array(off_diag_inds).astype(int)


def make_data_providers(args):
    args.use_sgd1 = (args.opt_method1 == 'SGD')
    args.use_sgd2 = (args.opt_method2 == 'SGD')
    args.use_sgd3 = (args.opt_method3 == 'SGD')
    args.data_provider1 = DataProvider(train_data=args.X_train,
                                       val_data=args.X_val,
                                       noise_samples=args.Y,
                                       noise_to_data_ratio=args.nu,
                                       num_latent_per_data=args.nz,
                                       variational_dist=args.var_dist,
                                       train_missing_data_mask=args.train_missing_data_mask,
                                       val_missing_data_mask=args.val_missing_data_mask,
                                       noise_miss_mask=args.noise_miss_mask,
                                       use_cdi=True,
                                       X_means=args.X_train_sample_mean,
                                       Y_means=args.Y_means,
                                       use_minibatches=args.use_sgd1,
                                       batch_size=args.batch_size,
                                       use_reparam_trick=False,
                                       rng=args.rng)

    args.data_provider2 = DataProvider(train_data=args.X_train,
                                       val_data=args.X_val,
                                       noise_samples=args.Y,
                                       noise_to_data_ratio=args.nu,
                                       num_latent_per_data=args.nz,
                                       variational_dist=args.approx_var_dist,
                                       train_missing_data_mask=args.train_missing_data_mask,
                                       val_missing_data_mask=args.val_missing_data_mask,
                                       noise_miss_mask=args.noise_miss_mask,
                                       use_cdi=True,
                                       X_means=args.X_train_sample_mean,
                                       Y_means=args.Y_means,
                                       use_minibatches=args.use_sgd2,
                                       batch_size=args.batch_size,
                                       use_reparam_trick=True,
                                       rng=args.rng)

    args.data_provider3 = DataProvider(train_data=args.X_train,
                                       val_data=args.X_val,
                                       noise_samples=args.Y,
                                       noise_to_data_ratio=args.nu,
                                       num_latent_per_data=args.nz,
                                       variational_dist=args.lognormal_var_dist,
                                       train_missing_data_mask=args.train_missing_data_mask,
                                       val_missing_data_mask=args.val_missing_data_mask,
                                       noise_miss_mask=args.noise_miss_mask,
                                       use_cdi=False,
                                       use_minibatches=args.use_sgd3,
                                       batch_size=args.batch_size,
                                       use_reparam_trick=True,
                                       rng=args.rng)


def make_vnce_loss_functions(args):
    """Returns two VNCE loss functions, the first uses a true posterior, the second uses an approximate one."""
    make_noise(args)
    make_model(args)
    make_var_dists(args)
    make_regulariser(args)
    make_data_providers(args)

    # create loss functions
    args.vnce_loss1 = MonteCarloVnceLoss(data_provider=args.data_provider1,
                                         model=args.model,
                                         noise=args.noise,
                                         variational_dist=args.var_dist,
                                         noise_to_data_ratio=args.nu,
                                         use_neural_model=False,
                                         use_neural_variational_dist=False,
                                         regulariser=args.l1_reg,
                                         reg_param_indices=args.reg_param_indices,
                                         use_minibatches=args.use_sgd1,
                                         separate_terms=args.separate_terms,
                                         use_numeric_stable_approx_second_term=args.use_numeric_stable_approx_second_term,
                                         rng=args.rng)

    args.vnce_loss2 = MonteCarloVnceLoss(data_provider=args.data_provider2,
                                         model=deepcopy(args.model),
                                         noise=args.noise,
                                         variational_dist=args.approx_var_dist,
                                         noise_to_data_ratio=args.nu,
                                         use_neural_model=False,
                                         use_neural_variational_dist=True,
                                         regulariser=args.l1_reg,
                                         reg_param_indices=args.reg_param_indices,
                                         use_minibatches=args.use_sgd2,
                                         use_reparam_trick=True,
                                         separate_terms=args.separate_terms,
                                         use_numeric_stable_approx_second_term=args.use_numeric_stable_approx_second_term,
                                         rng=args.rng)

    args.vnce_loss3 = MonteCarloVnceLoss(data_provider=args.data_provider3,
                                         model=deepcopy(args.model),
                                         noise=args.noise,
                                         variational_dist=args.lognormal_var_dist,
                                         noise_to_data_ratio=args.nu,
                                         use_neural_model=False,
                                         use_neural_variational_dist=args.use_sgd3,
                                         regulariser=args.l1_reg,
                                         reg_param_indices=args.reg_param_indices,
                                         use_minibatches=False,
                                         use_reparam_trick=True,
                                         separate_terms=args.separate_terms,
                                         use_numeric_stable_approx_second_term=args.use_numeric_stable_approx_second_term,
                                         rng=args.rng)


def make_vnce_optimisers(args):
    m_step1 = make_e_or_m_step(True, args.model_learn_rate, args.opt_method1, args.maxiter1, args.theta_inds)
    m_step2 = make_e_or_m_step(True, args.model_learn_rate, args.opt_method2, args.maxiter2, args.theta_inds)
    m_step3 = make_e_or_m_step(True, args.model_learn_rate, args.opt_method3, args.maxiter3, args.theta_inds)
    e_step1 = ExactEStep(track_loss=True)
    e_step2 = make_e_or_m_step(False, args.var_learn_rate, args.opt_method2, args.maxiter2)
    args.alpha_inds = np.arange(len(args.lognormal_var_dist.alpha))
    e_step3 = make_e_or_m_step(False, args.var_learn_rate, args.opt_method3, args.maxiter3, args.alpha_inds)

    args.vnce_optimiser1 = VemOptimiser(m_step=m_step1, e_step=e_step1, num_em_steps_per_save=args.num_em_steps_per_epoch1)
    args.vnce_optimiser2 = VemOptimiser(m_step=m_step2, e_step=e_step2, num_em_steps_per_save=args.num_em_steps_per_epoch2)
    args.vnce_optimiser3 = VemOptimiser(m_step=m_step3, e_step=e_step3, num_em_steps_per_save=args.num_em_steps_per_epoch3, save_dir=args.save_dir)
    # args.vnce_optimiser3 = ScipyOptimiser(opt_method=args.opt_method3, max_iter=args.maxiter3)


def make_e_or_m_step(do_m_step, learn_rate, opt_method, max_iter, inds=None):
    if opt_method == 'SGD':
        e_or_m_step = SgdEmStep(do_m_step=do_m_step,
                                learning_rate=learn_rate,
                                num_batches_per_em_step=args.num_batch_per_em_step,
                                inds=inds,
                                track_loss=args.track_loss)
    else:
        e_or_m_step = ScipyMinimiseEmStep(do_m_step=do_m_step,
                                          optimisation_method=opt_method,
                                          max_iter=max_iter,
                                          inds=inds)
    return e_or_m_step


def train(args):
    make_vnce_loss_functions(args)
    make_vnce_optimisers(args)

    try:
        args.vnce_optimiser1.fit(loss_function=args.vnce_loss1,
                                 theta0=deepcopy(args.vnce_loss1.model.theta),
                                 alpha0=deepcopy(args.vnce_loss1.get_alpha()),
                                 stop_threshold=args.stop_threshold,
                                 max_num_em_steps=args.max_num_em_steps1)
    except KeyboardInterrupt:
        pass
    try:
        args.vnce_optimiser2.fit(loss_function=args.vnce_loss2,
                                 theta0=deepcopy(args.vnce_loss2.model.theta),
                                 alpha0=deepcopy(args.vnce_loss2.get_alpha()),
                                 stop_threshold=args.stop_threshold,
                                 max_num_em_steps=args.max_num_em_steps2)
    except KeyboardInterrupt:
        pass

    # try:
    #     args.vnce_optimiser3.fit(loss_function=args.vnce_loss3)
    try:
        args.vnce_optimiser3.fit(loss_function=args.vnce_loss3,
                                 theta0=deepcopy(args.vnce_loss3.model.theta),
                                 alpha0=deepcopy(args.vnce_loss3.get_alpha()),
                                 stop_threshold=args.stop_threshold,
                                 max_num_em_steps=args.max_num_em_steps3)
    except KeyboardInterrupt:
        pass
    args.nce_missing_model = UnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.nce_missing_optimiser = NCEOptimiser(model=args.nce_missing_model,
                                              noise=args.noise,
                                              noise_samples=args.Y,
                                              regulariser=args.nce_l1_reg,
                                              reg_param_indices=args.reg_param_indices,
                                              nu=args.nu,
                                              save_dir=args.save_dir)
    args.filled_in_with_mean_data = args.X_train * (1 - args.train_missing_data_mask) + args.X_train_sample_mean * args.train_missing_data_mask
    try:
        args.nce_missing_optimiser.fit(X=args.filled_in_with_mean_data,
                                       theta0=deepcopy(args.theta0),
                                       opt_method=args.nce_opt_method,
                                       maxiter=args.maxiter_nce1,
                                       learning_rate=args.nce_learn_rate,
                                       batch_size=args.nce_batch_size,
                                       num_epochs=args.nce_missing_num_epochs1,
                                       inds=args.theta_inds)
    except KeyboardInterrupt:
        pass

    args.nce_missing_model_2 = UnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.nce_missing_optimiser_2 = NCEOptimiser(model=args.nce_missing_model_2,
                                                noise=args.noise,
                                                noise_samples=args.Y,
                                                regulariser=args.nce_l1_reg,
                                                reg_param_indices=args.reg_param_indices,
                                                nu=args.nu,
                                                save_dir=args.save_dir)
    noise_data = args.Y.reshape(args.nu, args.n, args.d)[0, :, :]
    args.filled_in_with_noise_data = args.X_train * (1 - args.train_missing_data_mask) + noise_data * args.train_missing_data_mask
    try:
        args.nce_missing_optimiser_2.fit(X=args.filled_in_with_noise_data,
                                         theta0=deepcopy(args.theta0),
                                         opt_method=args.nce_opt_method,
                                         maxiter=args.maxiter_nce2,
                                         learning_rate=args.nce_learn_rate,
                                         batch_size=args.nce_batch_size,
                                         num_epochs=args.nce_missing_num_epochs2,
                                         inds=args.theta_inds)
    except KeyboardInterrupt:
        pass

    args.nce_missing_model_3 = UnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.nce_missing_optimiser_3 = NCEOptimiser(model=args.nce_missing_model_3,
                                                noise=args.noise,
                                                noise_samples=args.Y,
                                                regulariser=args.nce_l1_reg,
                                                reg_param_indices=args.reg_param_indices,
                                                nu=args.nu,
                                                save_dir=args.save_dir)
    rnd_fill = args.rng.uniform(0, 3, args.X_train.shape)
    args.filled_in_with_rnd_data = args.X_train * (1 - args.train_missing_data_mask) + rnd_fill * args.train_missing_data_mask
    try:
        args.nce_missing_optimiser_3.fit(X=args.filled_in_with_rnd_data,
                                         theta0=deepcopy(args.theta0),
                                         opt_method=args.nce_opt_method,
                                         maxiter=args.maxiter_nce3,
                                         learning_rate=args.nce_learn_rate,
                                         batch_size=args.nce_batch_size,
                                         num_epochs=args.nce_missing_num_epochs3,
                                         inds=args.theta_inds)
    except KeyboardInterrupt:
        pass

    args.cd_model = MissingDataUnnormalisedTruncNorm(deepcopy(args.scale0), deepcopy(args.mean0), precision=deepcopy(args.prec0), rng=args.rng)
    args.cd_optimiser = CDOptimiser(model=args.cd_model, rng=args.rng, save_dir=args.save_dir)
    try:
        args.cd_optimiser.fit(X=args.X_train * (1 - args.train_missing_data_mask),
                              theta0=deepcopy(args.theta0),
                              num_gibbs_steps=args.cd_num_gibbs_steps,
                              learning_rate=args.cd_learn_rate,
                              batch_size=args.cd_batch_size,
                              num_epochs=args.cd_num_epochs,
                              nz=args.cd_nz,
                              inds=args.theta_inds)
    except:
        pass


def main(args):
    len_theta = int(0.5 * args.d * (args.d + 3)) + 1  # only optimise scaling param and precision
    args.theta_inds = np.concatenate((np.array([0]), np.arange(args.d + 1, len_theta))).astype(int)

    exp_dir = os.path.join(args.exp_name, str(args.sim_num), 'frac{}/reg{}'.format(str(args.frac_missing), str(args.reg_param)))
    args.save_dir = os.path.join(args.save_dir, exp_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    load_data(args)
    train(args)
    calculate_mse(args)
    plot_and_save_results(args.save_dir, args)


if __name__ == "__main__":
    main(args)
