import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
code_dir_3 = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dir_4 = '/home/ben/ben-rhodes-masters-project/proposal/code/neural_network'
code_dirs = [code_dir, code_dir_2, code_dir_3, code_dir_4]
for code_dir in code_dirs:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import numpy as np
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from itertools import combinations
from numpy import random as rnd
from scipy.optimize import root, minimize, fsolve
from scipy.special import erfcx
from scipy.stats import norm, multivariate_normal
from time import gmtime, strftime

from distribution import MissingDataProductOfTruncNormsPosterior, MissingDataProductOfTruncNormNoise, MissingDataUniformNoise
from fully_observed_models import UnnormalisedTruncNorm
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from nce_optimiser import NCEOptimiser
from plot import *
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, ScipyMinimiseEmStep, MonteCarloVnceLoss

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--data_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data/',
                    help='Path to directory where data is loaded and saved')
# parser.add_argument('--save_dir', type=str, default='/home/ben/ben-rhodes-masters-project/experimental_results/trunc_norm',
#                     help='Path to directory where model will be saved')
parser.add_argument('--save_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm',
                    help='Path to directory where model will be saved')
parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default='d=2-frac=0.1', help='name of this exact experiment')

# Data arguments
parser.add_argument('--which_dataset', default='synthetic', help='options: usps and synthetic')
parser.add_argument('--n', type=int, default=10000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=int, default=1, help='ratio of noise to data samples in NCE')
parser.add_argument('--frac_missing', type=float, default=0, help='fraction of data missing at random')

# Model arguments
parser.add_argument('--theta0_path', type=str, default=None, help='path to pre-trained weights')
parser.add_argument('--d', type=int, default=2, help='dimension of visibles for synthetic dataset')
parser.add_argument('--num_layers', type=int, default=2, help='dimension of visibles for synthetic dataset')
parser.add_argument('--hidden_dim', type=int, default=100, help='dimension of visibles for synthetic dataset')
parser.add_argument('--activation_layer', type=object, default=TanhLayer(), help='dimension of visibles for synthetic dataset')

# Latent NCE optimisation arguments
parser.add_argument('--opt_method', type=str, default='SGD', help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter', type=int, default=5, help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=0, help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_epochs', type=int, default=100, help='Maximum number of loops through the dataset during training')
parser.add_argument('--model_learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used to train the model')
parser.add_argument('--var_learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used to train the variational dist')
parser.add_argument('--batch_size', type=int, default=100, help='if opt_method=SGD, this is the size of a minibatch')
parser.add_argument('--num_batch_per_em_step', type=int, default=100, help='if opt_method=SGD, this is the number of batches per EM step')
parser.add_argument('--track_loss', dest='track_loss', action='store_true', help='track VNCE loss in E & M steps')
parser.add_argument('--no-track_loss', dest='track_loss', action='store_false')
parser.set_defaults(track_loss=True)

# nce optimisation arguments
parser.add_argument('--nce_opt_method', type=str, default='SGD', help='nce optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter_nce', type=int, default=50, help='number of iterations inside scipy.minimize')
parser.add_argument('--nce_num_epochs', type=int, default=50, help='if nce_opt_method=SGD, this is the number of passes through data set')
parser.add_argument('--nce_missing_num_epochs', type=int, default=50, help='if nce_opt_method=SGD, this is the number of passes through data set')
parser.add_argument('--nce_learn_rate', type=float, default=0.1, help='if nce_opt_method=SGD, this is the learning rate used')
parser.add_argument('--nce_batch_size', type=int, default=100, help='if nce_opt_method=SGD, this is the size of a minibatch')

# Other arguments
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=True)
parser.add_argument('--random_seed', type=int, default=1083463236, help='seed for np.random.RandomState')

args = parser.parse_args()
args.rng = rnd.RandomState(args.random_seed)
if args.opt_method == 'SGD':
    args.num_em_steps_per_epoch = args.n / (args.batch_size * args.num_batch_per_em_step)
else:
    args.num_em_steps_per_epoch = 1
args.max_num_em_steps = args.max_num_epochs * args.num_em_steps_per_epoch
save_dir = os.path.join(args.save_dir, args.exp_name, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def compose_layers(args):
    """Return the layers for a multi-layer neural network according to the specified hyperparameters"""

    first_layer = [AffineLayer(args.d, args.hidden_dim, args.weights_init, args.biases_init),
                   args.activation_layer]

    hidden_layers = []
    for i in range(args.num_layers - 2):
        hidden_layers.append(AffineLayer(args.hidden_dim, args.hidden_dim, args.weights_init, args.biases_init))
        hidden_layers.append(args.activation_layer)

    final_layer = [AffineLayer(args.hidden_dim, args.d * 2, args.weights_init, args.biases_init)]

    return first_layer + hidden_layers + final_layer


def estimate_trunc_norm_params(sample_mean, sample_var):
    """Return the estimated mean and std of the factorial *non*-truncated normal that, after truncation, fits the data """
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


def generate_data(args):
    rng = args.rng

    # initialise ground-truth parameters for data generating distribution
    # args.true_mean = np.ones(args.d)
    # args.true_chol = np.tril(np.ones((args.d, args.d))) * 0.3
    args.true_mean = np.ones(args.d)
    # todo: get precision!
    print('true parameters: \n mean: {} \n chol: {}'.format(args.true_mean, args.true_chol))

    args.data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=args.true_mean, chol=args.true_chol, rng=rng)
    args.theta_true = deepcopy(args.data_dist.theta)

    # make synthetic data
    args.X_train = args.data_dist.sample(args.n)
    args.X_val = args.data_dist.sample(int(args.n / 5))

    # make masks (which are used for simulating missing data)
    args.train_missing_data_mask = args.rng.uniform(0, 1, args.X_train.shape) < args.frac_missing
    args.val_missing_data_mask = args.rng.uniform(0, 1, args.X_val.shape) < args.frac_missing

    # calculate the sample mean and variance of non-missing data
    observed_data = args.X_train * (1 - args.train_missing_data_mask)
    masked = np.ma.masked_equal(observed_data, 0)
    args.X_train_sample_mean = np.array(masked.mean(0))
    args.X_train_sample_diag_var = np.array(masked.var(0))  # ignore covariances
    # print('sample mean: {} \n sample var: {}'.format(args.X_train_sample_mean, args.X_train_sample_diag_var))


def make_vnce_loss_function(args):
    """ """
    rng = args.rng

    # estimate (from the synthetic data) the parameters of a factorial truncated normal for the noise distribution
    noise_mean, noise_chol = estimate_trunc_norm_params(args.X_train_sample_mean, args.X_train_sample_diag_var)
    # noise_mean = np.ones(args.d, dtype=float) * 5
    # noise_chol = np.ones(args.d, dtype=float) * 0.1
    print('estimated noise parameters: \n mean: {} \n chol: {}'.format(noise_mean, noise_chol))

    # make noise distribution and noise samples for vnce
    args.noise = MissingDataProductOfTruncNormNoise(mean=noise_mean, chol=noise_chol, rng=rng)
    # args.noise = MissingDataUniformNoise(np.zeros(args.d), np.ones(args.d) * 100)
    args.Y = args.noise.sample(int(args.n * args.nu))
    args.noise_miss_mask = np.repeat(args.train_missing_data_mask, args.nu, axis=0)
    assert args.Y.shape == args.noise_miss_mask.shape, 'noise samples do not have the same shape as the missing data mask'

    # initialise the model p(x, z)
    # todo: may need to initialise this scaling parameter more judiciously when d >> 0
    args.scale0 = np.array([0.])
    args.mean0 = np.zeros(args.d).astype(float)
    args.chol0 = np.identity(args.d).astype(float)  # cholesky of precision
    # args.mean0 = deepcopy(noise_mean)
    # args.chol0 = deepcopy(np.diag(noise_chol))  # cholesky of precision
    # args.mean0 = args.true_mean
    # args.chol0 = args.true_chol
    model = MissingDataUnnormalisedTruncNorm(scaling_param=args.scale0, mean=args.mean0, chol=args.chol0, rng=rng)
    args.theta0 = deepcopy(model.theta)

    # create variational distribution, which uses a multi-layer neural network
    args.weights_init = UniformInit(-0.001, 0.001, rng=rng)
    args.biases_init = ConstantInit(0)
    # args.biases_init = ConstantVectorInit(np.concatenate((noise_mean, np.log(noise_chol))))
    # args.biases_init = ConstantVectorInit(np.concatenate((args.true_mean, np.diag(args.true_chol))))
    layers = [AffineLayer(args.d, 2 * args.d, args.weights_init, args.biases_init)]
    # layers = compose_layers(args)
    nn = MultipleLayerModel(layers)
    var_dist = MissingDataProductOfTruncNormsPosterior(nn=nn, data_dim=args.d, rng=rng)

    # create loss function
    use_sgd = (args.opt_method == 'SGD')
    vnce_loss_function = MonteCarloVnceLoss(model=model,
                                            train_data=args.X_train,
                                            val_data=args.X_val,
                                            noise=args.noise,
                                            noise_samples=args.Y,
                                            variational_noise=var_dist,
                                            noise_to_data_ratio=args.nu,
                                            num_latent_per_data=args.nz,
                                            use_neural_model=False,
                                            use_neural_variational_noise=True,
                                            train_missing_data_mask=args.train_missing_data_mask,
                                            val_missing_data_mask=args.val_missing_data_mask,
                                            noise_miss_mask=args.noise_miss_mask,
                                            use_minibatches=use_sgd,
                                            batch_size=args.batch_size,
                                            use_reparam_trick=True,
                                            separate_terms=args.separate_terms,
                                            rng=rng)
    # vnce_loss_function = MonteCarloVnceLoss(model=model,
    #                                         train_data=args.X_train,
    #                                         val_data=args.X_val,
    #                                         noise=args.noise,
    #                                         noise_samples=args.Y,
    #                                         variational_noise=var_dist,
    #                                         noise_to_data_ratio=args.nu,
    #                                         num_latent_per_data=args.nz,
    #                                         use_neural_model=False,
    #                                         use_neural_variational_noise=True,
    #                                         use_minibatches=use_sgd,
    #                                         batch_size=args.batch_size,
    #                                         use_reparam_trick=True,
    #                                         separate_terms=args.separate_terms,
    #                                         drop_data_frac=args.frac_missing,
    #                                         rng=rng)
    return vnce_loss_function


def make_vnce_optimiser(args):
    if args.opt_method == 'SGD':
        m_step = SgdEmStep(do_m_step=True,
                           learning_rate=args.model_learn_rate,
                           num_batches_per_em_step=args.num_batch_per_em_step,
                           noise_to_data_ratio=args.nu,
                           track_loss=args.track_loss,
                           rng=args.rng)
    else:
        m_step = ScipyMinimiseEmStep(do_m_step=True,
                                     optimisation_method=args.opt_method,
                                     max_iter=args.maxiter)
    e_step = SgdEmStep(do_m_step=False,
                       learning_rate=args.var_learn_rate,
                       num_batches_per_em_step=args.num_batch_per_em_step,
                       noise_to_data_ratio=args.nu,
                       track_loss=args.track_loss,
                       rng=args.rng)

    vnce_optimiser = VemOptimiser(m_step=m_step, e_step=e_step, num_em_steps_per_save=args.num_em_steps_per_epoch)

    return vnce_optimiser


def train(args):
    args.vnce_loss = make_vnce_loss_function(args)
    args.vnce_optimiser = make_vnce_optimiser(args)
    args.vnce_optimiser.fit(loss_function=args.vnce_loss,
                            theta0=deepcopy(args.vnce_loss.model.theta),
                            alpha0=deepcopy(args.vnce_loss.variational_noise.nn.params),
                            stop_threshold=args.stop_threshold,
                            max_num_em_steps=args.max_num_em_steps)

    args.nce_model = UnnormalisedTruncNorm(args.scale0, args.mean0, args.chol0, rng=args.rng)
    args.nce_optimiser = NCEOptimiser(model=args.nce_model, noise=args.noise, noise_samples=args.Y, nu=args.nu)
    args.nce_optimiser.fit(X=args.X_train,
                           theta0=args.theta0,
                           opt_method=args.nce_opt_method,
                           maxiter=args.maxiter_nce,
                           learning_rate=args.nce_learn_rate,
                           batch_size=args.nce_batch_size,
                           num_epochs=args.nce_num_epochs)

    args.nce_missing_model = UnnormalisedTruncNorm(args.scale0, args.mean0, args.chol0, rng=args.rng)
    args.nce_missing_optimiser = NCEOptimiser(model=args.nce_missing_model, noise=args.noise, noise_samples=args.Y, nu=args.nu)
    args.nce_missing_optimiser.fit(X=args.X_train * (1 - args.train_missing_data_mask),
                                   theta0=args.theta0,
                                   opt_method=args.nce_opt_method,
                                   maxiter=args.maxiter_nce,
                                   learning_rate=args.nce_learn_rate,
                                   batch_size=args.nce_batch_size,
                                   num_epochs=args.nce_missing_num_epochs)


def print_results(args):
    args.init_mse = get_mse(args, args.theta0[1:], 'Initial')
    args.vnce_mse = get_mse(args, args.vnce_loss.model.theta[1:], 'Missing Data VNCE')
    args.nce_mse = get_mse(args, args.nce_model.theta[1:], 'NCE')
    args.nce_missing_mse = get_mse(args, args.nce_missing_model.theta[1:], 'Missing Data NCE')
    print('final vnce scaling parameter: {}'.format(args.vnce_loss.model.theta[0]))
    print('final nce scaling parameter: {}'.format(args.nce_model.theta[0]))
    print('final missing data nce scaling parameter: {}'.format(args.nce_missing_model.theta[0]))


def get_mse(args, theta, method_name):
    mse = mean_square_error(args.theta_true[1:], theta)
    print('{} Mean Squared Error: {}'.format(method_name, mse))
    # print('true theta: {}'.format(args.theta_true[1:]))
    # print('{} theta: {}'.format(method_name, theta))
    return mse


def plot_and_save_results(save_dir, args):
    vnce_thetas, vnce_alphas, vnce_losses, vnce_times = args.vnce_optimiser.get_flattened_result_arrays(flatten_params=False)
    m_step_ids, e_step_ids, m_step_start_ids, e_step_start_ids = args.vnce_optimiser.get_m_and_e_step_ids()

    vnce_train_losses = np.array(args.vnce_optimiser.train_losses)
    vnce_val_losses = np.array(args.vnce_optimiser.val_losses)

    vnce_plot = plot_vnce_loss(vnce_times, vnce_train_losses, vnce_val_losses)
    save_fig(vnce_plot, save_dir, 'vnce_loss')

    nce_plot, axs = plt.subplots(1, 2, figsize=(5.5, 5))
    axs = axs.ravel()
    axs[0].plot(args.nce_optimiser.times, args.nce_optimiser.Js, c='b', label='NCE (train)')
    axs[1].plot(args.nce_missing_optimiser.times, args.nce_missing_optimiser.Js, c='r', label='NCE missing data (train)')
    nce_plot.legend()
    save_fig(nce_plot, save_dir, 'nce_loss')

    with open(os.path.join(save_dir, "config.txt"), 'w') as f:
        for key, value in vars(args).items():
            f.write("{}: {}\n".format(key, value))

    pickle.dump(args, open(os.path.join(save_dir, "config.p"), "wb"))
    pickle.dump(args.data_dist, open(os.path.join(save_dir, "data_dist.p"), "wb"))
    pickle.dump(args.vnce_loss.model, open(os.path.join(save_dir, "vnce_model.p"), "wb"))
    pickle.dump(args.nce_model, open(os.path.join(save_dir, "nce_model.p"), "wb"))
    pickle.dump(args.vnce_loss.noise, open(os.path.join(save_dir, "noise.p"), "wb"))
    pickle.dump(args.vnce_loss.variational_noise, open(os.path.join(save_dir, "var_dist.p"), "wb"))
    pickle.dump(args.vnce_optimiser.thetas, open(os.path.join(save_dir, "vnce_thetas.p"), "wb"))
    pickle.dump(args.vnce_optimiser.alphas, open(os.path.join(save_dir, "vnce_alphas.p"), "wb"))

    np.savez(os.path.join(save_dir, "theta0_and_theta_true"),
             theta0=args.theta0,
             theta_true=args.theta_true,
             vnce_mse=args.vnce_mse,
             nce_mse=args.nce_mse,
             nce_missing_mse=args.nce_missing_mse)
    np.savez(os.path.join(save_dir, "data"),
             X_train=args.vnce_loss.train_data,
             X_train_mask=args.vnce_loss.train_miss_mask,
             X_val=args.vnce_loss.val_data,
             X_val_mask=args.vnce_loss.val_miss_mask,
             Y=args.vnce_loss.Y)
    np.savez(os.path.join(save_dir, "vnce_results"),
             vnce_times=vnce_times,
             vnce_losses=vnce_losses,
             vnce_val_losses=vnce_val_losses,
             m_step_ids=m_step_ids,
             e_step_ids=e_step_ids,
             m_step_start_ids=m_step_start_ids,
             e_step_start_ids=e_step_start_ids)
    np.savez(os.path.join(save_dir, "nce_results"),
             nce_thetas=args.nce_optimiser.thetas,
             nce_times=args.nce_optimiser.times,
             nce_losses=args.nce_optimiser.Js)
    np.savez(os.path.join(save_dir, "nce_missing_data_results"),
             nce_thetas=args.nce_missing_optimiser.thetas,
             nce_times=args.nce_missing_optimiser.times,
             nce_losses=args.nce_missing_optimiser.Js)


def main(args, save_dir):
    generate_data(args)
    train(args)
    print_results(args)
    plot_and_save_results(save_dir, args)


if __name__ == "__main__":
    main(args, save_dir)
