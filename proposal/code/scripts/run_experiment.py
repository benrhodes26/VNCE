"""Training script for experimental comparison of learning an RBM using
two methods: latent nce and contrastive divergence.
"""
import os
import sys
cur_dir = os.path.split(os.getcwd())[0]
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import numpy as np
import os
import pdb
import pickle

# my code
from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_nce_optimiser import LatentNCEOptimiser
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from utils import get_true_weights, plot_rbm_parameters, plot_log_likelihood_training_curves,\
    rescale_times, average_log_likelihood

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd
from time import gmtime, strftime

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent '
                        'nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--save_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/'
                                                    'ben-rhodes-masters-project/experimental-results',
                    help='Path to directory where model will be saved')
parser.add_argument('--name', type=str, default=START_TIME, help='name of experiment file')
# Data arguments
parser.add_argument('--n', type=int, default=10000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=float, default=1.0, help='ratio of noise to data samples in NCE')
parser.add_argument('--num_gibbs_steps', type=int, default=1000,
                    help='number of gibbs steps used to generate as synthetic dataset')
# Model arguments
parser.add_argument('--d', type=int, default=20, help='dimension of visibles')
parser.add_argument('--m', type=int, default=15, help='dimension of hiddens')
parser.add_argument('--true_sd', type=float, default=1.0, help='standard deviation of gaussian rv for true weights')

# Latent NCE optimisation arguments
parser.add_argument('--noise', type=str, default='marginal',
                    help='type of noise distribution for latent NCE. Currently, this'
                         'can be either marginals or chow-liu')
parser.add_argument('--opt_method', type=str, default='L-BFGS-B',
                    help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--ftol', type=float, default=2.220446049250313e-09,
                    help='Tolerance used as stopping criterion in scipy.minimize')
parser.add_argument('--maxiter', type=int, default=20,
                    help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=1e-09,
                    help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_em_steps', type=int, default=20,
                    help='Maximum number of EM steps to perform')
parser.add_argument('--learn_rate', type=float, default=0.01,
                    help='if opt_method=SGD, this is the learning rate used')
parser.add_argument('--batch_size', type=int, default=10,
                    help='if opt_method=SGD, this is the size of a minibatch')

# Contrastive divergence optimisation arguments
parser.add_argument('--cd_num_steps', type=int, default=1, help='number of gibbs steps used to sample from '
                                                                'model during learning with CD')
parser.add_argument('--cd_learn_rate', type=float, default=0.01,
                    help='Initial learning rate for contrastive divergence')
parser.add_argument('--cd_batch_size', type=int, default=10, help='number of datapoints used per gradient update')
parser.add_argument('--cd_num_epochs', type=int, default=100, help='number of passes through data set')

# Lnce optimisation arguments
parser.add_argument('--maxiter_nce', type=int, default=100, help='number of passes through data set')

# Other arguments
parser.add_argument('--time_step_size_lnce', type=int, default=1, help='How often we calculate log-likelihoods. '
                    'Every time-step is one iteration/epoch of the lnce optimiser.')

args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir, args.name)
os.mkdir(SAVE_DIR)

"""
==========================================================================================================
                                            SETUP
==========================================================================================================
"""

# For reproducibility
rng = rnd.RandomState(1083463236)

n, nz, nu = args.n, args.nz, args.nu
d, m = args.d, args.m
true_sd = args.true_sd

# todo: get realistic weights learnt from downsampled mnist digits
# generate weights of RBM that we want to learn
true_theta = rng.randn(d+1, m+1) * true_sd
true_theta[0, 0] = 0

# generate synthetic dataset
true_data_dist = RestrictedBoltzmannMachine(true_theta, rng=rng)
X, Z = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
X_mean = np.mean(X, axis=0)

# random initial weights, that depend on the data
theta0 = np.asarray(
    rng.uniform(
        low=-4 * np.sqrt(6. / (d + m)),
        high=4 * np.sqrt(6. / (d + m)),
        size=(d+1, m+1)
    ))
theta0[1:, 0] = np.log(X_mean / (1 - X_mean))  # visible biases
theta0[0, 0] = -m * np.log(2) + np.sum(np.log(1 - X_mean))  # scaling parameter
theta0[0, 1:] = 0  # hidden biases

# initialise models
model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
cd_model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
init_model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)  # rbm with initial parameters, for comparison

# initialise noise distributions for latent NCE
noise_dist = None
if args.noise == 'marginal':
    noise_dist = MultivariateBernoulliNoise(X_mean, rng=rng)
elif args.noise == 'chow_liu':
    noise_dist = ChowLiuTree(X, rng=rng)

# initialise variational distribution, which is the exact posterior in this case
var_dist = RBMLatentPosterior(theta0, rng=rng)

# initialise optimisers
optimiser = LatentNCEOptimiser(model, noise_dist, var_dist, n, nu=nu, latent_samples_per_datapoint=nz, rng=rng)
cd_optimiser = CDOptimiser(cd_model, rng=rng)
nce_optimiser = NCEOptimiser(nce_model, noise_dist, n, nu=nu)

# calculate normalisation constant of true model
true_data_dist.reset_norm_const()
true_norm_const = true_data_dist.norm_const
true_theta[0, 0] = -np.log(true_norm_const)


"""
==========================================================================================================
                                            OPTIMISATION
==========================================================================================================
"""

# perform latent nce optimisation
print('starting latent nce optimisation...')
_ = optimiser.fit_using_analytic_q(X=X,
                                   theta0=model.theta,
                                   opt_method=args.opt_method,
                                   ftol=args.ftol,
                                   maxiter=args.maxiter,
                                   stop_threshold=args.stop_threshold,
                                   max_num_em_steps=args.max_num_em_steps,
                                   learning_rate=args.learn_rate,
                                   batch_size=args.batch_size,
                                   disp=False,
                                   plot=False)
print('finished!')
optimal_J1 = optimiser.evaluate_J1_at_param(theta=true_theta.reshape(-1), X=X)
print('J1(true_theta) = {}'.format(optimal_J1))
print('starting cd optimisation...')
# perform contrastive divergence optimisation
_ = cd_optimiser.fit(X=X,
                     theta0=theta0.reshape(-1),
                     num_gibbs_steps=args.cd_num_steps,
                     learning_rate=args.cd_learn_rate,
                     batch_size=args.cd_batch_size,
                     num_epochs=args.cd_num_epochs)
print('finished!')
print('starting nce optimisation...')
_ = nce_optimiser.fit(X=X, theta0=theta0.reshape(-1), maxiter=args.maxiter_nce)

"""
==========================================================================================================
                                        METRICS & PLOTTING
==========================================================================================================
"""

# throw away some of the intermediate results obtained during optimisation, since
# it can be expensive to compute the log-likelihood if we don't need granular plots
optimiser.reduce_optimisation_results(args.time_step_size_lnce)

# rescale timings to have the same resolution as lnce timings
cd_optimiser.times, cd_time_ids = rescale_times(deepcopy(optimiser.times),
                                                deepcopy(cd_optimiser.times))
cd_optimiser.thetas = cd_optimiser.thetas[cd_time_ids]
nce_optimiser.times, nce_time_ids = rescale_times(deepcopy(optimiser.times),
                                                  deepcopy(nce_optimiser.times))
nce_optimiser.thetas = nce_optimiser.thetas[nce_time_ids]
nce_optimiser.Js = nce_optimiser.Js[nce_time_ids]

# calculate average log-likelihood at each iteration for both models
print('calculating log-likelihoods...')
av_log_like_lnce = optimiser.av_log_like_for_each_iter(X)
av_log_like_cd = cd_optimiser.av_log_like_for_each_iter(X)
av_log_like_nce = nce_optimiser.av_log_like_for_each_iter(X)
print('finished!')

# calculate average log-likelihood of initial and true models
true_log_like = average_log_likelihood(true_data_dist, X)
init_log_like = average_log_likelihood(init_model, X)
noise_log_like = np.mean(np.log(noise_dist(X)))

training_curves = [[optimiser.times, av_log_like_lnce, 'lnce'],
                   [cd_optimiser.times, av_log_like_cd, 'cd'],
                   [nce_optimiser.times, av_log_like_nce, 'nce']]
static_lines = [[true_log_like, 'true'],
                [init_log_like, 'initial'],
                [noise_log_like, 'noise']]

# plot log-likelihood during training
like_training_plot = plot_log_likelihood_training_curves(training_curves, static_lines)
like_training_plot.savefig('{}/likelihood-optimisation-curve.pdf'.format(SAVE_DIR))

# plot J (NCE objective function) and J1 (lower bound to NCE objective) during training
J_plot = optimiser.plot_loss_curve(optimal_J1=optimal_J1)
ax = J_plot.gca()
ax.plot(nce_optimiser.times, nce_optimiser.Js, label='J')
J_plot.savefig('{}/J-optimisation-curve.pdf'.format(SAVE_DIR))

# plot rbm weights for each model (including ground truth and random initialisation)
params = [true_data_dist.theta, model.theta, cd_model.theta, init_model.theta]
titles = ['True parameters', 'Latent NCE parameters',
          'Contrastive divergence parameters', 'Randomly initialised parameters']
rbm_weights_plot = plot_rbm_parameters(params, titles, d, m)
rbm_weights_plot.savefig('{}/rbm-weight-visualisation.pdf'.format(SAVE_DIR))

# check that latent nce has produced an approximately normalised model
model.reset_norm_const()
print('We hope that the normalisation constant of the learnt model is 1. In reality it is: {} '.format(
    model.norm_const))

END_TIME = strftime('%Y%m%d-%H%M', gmtime())

"""
==========================================================================================================
                                            SAVE TO FILE
==========================================================================================================
"""
# save everything
config = {'start time': START_TIME,
          'end_time': END_TIME,
          'n': n,
          'nz': nz,
          'nu': nu,
          'd': d,
          'm': m,
          'true_sd': true_sd,
          'num_gibbs_steps': args.num_gibbs_steps,
          'noise': args.noise,
          'opt_method': args.opt_method,
          'ftol': args.ftol,
          'maxiter': args.maxiter,
          'stop_threshold': args.stop_threshold,
          'max_num_em_steps': args.max_num_em_steps,
          'learn_rate': args.learn_rate,
          'batch_size': args.batch_size,
          'cd_num_steps': args.cd_num_steps,
          'cd_learn_rate': args.cd_learn_rate,
          'cd_batch_size': args.cd_batch_size,
          'cd_num_epochs': args.cd_num_epochs,
          'random_seed': 1083463236}

with open(os.path.join(SAVE_DIR, "config.txt"), 'w') as f:
    for key, value in config.items():
        f.write("{}: {}\n".format(key, value))

pickle.dump(config, open(os.path.join(SAVE_DIR, "config.p"), "wb"))
pickle.dump(like_training_plot, open(os.path.join(SAVE_DIR, "likelihood_training_plot.p"), "wb"))
pickle.dump(J_plot, open(os.path.join(SAVE_DIR, "J_plot.p"), "wb"))
pickle.dump(rbm_weights_plot, open(os.path.join(SAVE_DIR, "rbm_weights_plot.p"), "wb"))
pickle.dump(optimiser, open(os.path.join(SAVE_DIR, "lnce_optimiser.p"), "wb"))
pickle.dump(cd_optimiser, open(os.path.join(SAVE_DIR, "cd_optimiser.p"), "wb"))
pickle.dump(nce_optimiser, open(os.path.join(SAVE_DIR, "nce_optimiser.p"), "wb"))

np.savez(os.path.join(SAVE_DIR, "data"), X=X, Y=optimiser.Y)
np.savez(os.path.join(SAVE_DIR, "true_weights_and_likelihood"), true_theta=true_theta, ll=true_log_like)
np.savez(os.path.join(SAVE_DIR, "init_theta_and_likelihood"), theta0=theta0, ll=init_log_like)
np.savez(os.path.join(SAVE_DIR, "lnce_results"), params=optimiser.thetas, J1s=optimiser.J1s,
         times=optimiser.times, ll=av_log_like_lnce)
np.savez(os.path.join(SAVE_DIR, "cd_results"), params=cd_optimiser.thetas, times=cd_optimiser.times, ll=av_log_like_cd)
np.savez(os.path.join(SAVE_DIR, "nce_results"), params=nce_optimiser.thetas, times=nce_optimiser.times,
         ll=av_log_like_nce)
