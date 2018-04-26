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
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from utils import *
from vnce_optimisers import VNCEOptimiser, VNCEOptimiserWithoutImportanceSampling

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd
from time import gmtime, strftime

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--save_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/'
                                                    'ben-rhodes-masters-project/experimental-results/rbm',
                    help='Path to directory where model will be saved')
parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')
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
parser.add_argument('--optimiser', type=str, default='VNCEOptimiser',
                    help='optimiser class to use. See latent_nce_optimiser.py for options')
parser.add_argument('--noise', type=str, default='marginal',
                    help='type of noise distribution for latent NCE. Currently, this can be either marginals or chow-liu')
parser.add_argument('--opt_method', type=str, default='L-BFGS-B',
                    help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--ftol', type=float, default=2.220446049250313e-09,
                    help='Tolerance used as stopping criterion in scipy.minimize')
parser.add_argument('--pos_var_threshold', type=float, default=0.01,
                    help='if ratio: (model_posterior/variational_posterior) exceeds this threshold during the M-step'
                         'of optimisation, then terminate the M-step')
parser.add_argument('--ratio_variance_method_1', dest='ratio_variance_method_1', action='store_true')
parser.add_argument('--no-ratio_variance_method_1', dest='ratio_variance_method_1', action='store_false')
parser.set_defaults(ratio_variance_method_1=True)
parser.add_argument('--maxiter', type=int, default=5,
                    help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=1e-09,
                    help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_em_steps', type=int, default=500,
                    help='Maximum number of EM steps to perform')
parser.add_argument('--learn_rate', type=float, default=0.01,
                    help='if opt_method=SGD, this is the learning rate used')
parser.add_argument('--batch_size', type=int, default=10,
                    help='if opt_method=SGD, this is the size of a minibatch')


# Contrastive divergence optimisation arguments
parser.add_argument('--cd_num_steps', type=int, default=1, help='number of gibbs steps used to sample from '
                                                                'model during learning with CD')
parser.add_argument('--cd_learn_rate', type=float, default=0.01, help='Initial learning rate for contrastive divergence')
parser.add_argument('--cd_batch_size', type=int, default=100, help='number of datapoints used per gradient update')
parser.add_argument('--cd_num_epochs', type=int, default=2000, help='number of passes through data set')

# nce optimisation arguments
parser.add_argument('--maxiter_nce', type=int, default=500, help='number of passes through data set')

# Other arguments
parser.add_argument('--compute_log_like', dest='compute_log_like', action='store_true')
parser.add_argument('--no-compute_log_like', dest='compute_log_like', action='store_false')
parser.set_defaults(compute_log_like=True)
parser.add_argument('--num_log_like_steps', type=int, default=20, help='Number of time-steps for which we calculate log-likelihoods')
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=True)

args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir, args.exp_name, args.name)
os.makedirs(SAVE_DIR)

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
true_theta = rng.randn(d + 1, m + 1) * true_sd
true_theta[0, 0] = 0

# generate synthetic dataset
true_data_dist = RestrictedBoltzmannMachine(true_theta, rng=rng)
X, Z = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
X_mean = np.mean(X, axis=0)

# initialise noise distributions for VNCE & sample from it
noise_dist = None
if args.noise == 'marginal':
    noise_dist = MultivariateBernoulliNoise(X_mean, rng=rng)
elif args.noise == 'chow_liu':
    noise_dist = ChowLiuTree(X, rng=rng)
Y = noise_dist.sample(int(n * nu))  # generate noise

# random initial weights, that depend on the data
theta0 = np.asarray(
    rng.uniform(
        low=-4 * np.sqrt(6. / (d + m)),
        high=4 * np.sqrt(6. / (d + m)),
        size=(d + 1, m + 1)
    ))
theta0[1:, 0] = np.log(X_mean / (1 - X_mean))  # visible biases
theta0[0, 0] = -m * np.log(2) + np.sum(np.log(1 - X_mean))  # scaling parameter
theta0[0, 1:] = 0  # hidden biases

# initialise models
model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
cd_model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
init_model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)  # rbm with initial parameters, for comparison

# initialise variational distribution, which is the exact posterior in this case
var_dist = RBMLatentPosterior(theta0, rng=rng)

# initialise optimisers
if args.optimiser == 'VNCEOptimiser':
    optimiser = VNCEOptimiser(model=model, noise=noise_dist, noise_samples=Y, variational_dist=var_dist,
                              sample_size=n, nu=nu, latent_samples_per_datapoint=nz, rng=rng,
                              pos_var_threshold=args.pos_var_threshold, ratio_variance_method_1=args.ratio_variance_method_1)
if args.optimiser == 'VNCEOptimiserWithoutImportanceSampling':
    optimiser = VNCEOptimiserWithoutImportanceSampling(model=model, noise=noise_dist, noise_samples=Y, variational_dist=var_dist,
                                                       sample_size=n, nu=nu, latent_samples_per_datapoint=nz, rng=rng)

cd_optimiser = CDOptimiser(cd_model, rng=rng)
nce_optimiser = NCEOptimiser(model=nce_model, noise=noise_dist, noise_samples=Y, sample_size=n, nu=nu)

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
_ = optimiser.fit(X=X,
                  theta0=model.theta,
                  opt_method=args.opt_method,
                  ftol=args.ftol,
                  maxiter=args.maxiter,
                  stop_threshold=args.stop_threshold,
                  max_num_em_steps=args.max_num_em_steps,
                  learning_rate=args.learn_rate,
                  batch_size=args.batch_size,
                  disp=False,
                  plot=False,
                  separate_terms=args.separate_terms,
                  save_dir=SAVE_DIR)
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

if args.separate_terms:
    Js_for_lnce_thetas = get_Js_for_vnce_thetas(X, nce_optimiser, optimiser, separate_terms=args.separate_terms)
    J_plot, Js_for_lnce_thetas = create_J_diff_plot(optimiser, Js_for_lnce_thetas)
    J_plot.savefig('{}/J-optimisation-curve.pdf'.format(SAVE_DIR))

# reduce results, since calculating log-likelihood is expensive
reduced_lnce_times, reduced_lnce_thetas = optimiser.reduce_optimisation_results(args.num_log_like_steps)
reduced_cd_times, reduced_cd_thetas = cd_optimiser.reduce_optimisation_results(args.num_log_like_steps)
cd_optimiser.times, cd_optimiser.thetas = reduced_cd_times, reduced_cd_thetas
reduced_nce_times, reduced_nce_thetas = nce_optimiser.reduce_optimisation_results(args.num_log_like_steps)

av_log_like_lnce = None
av_log_like_cd = None
av_log_like_nce = None
true_log_like = None
init_log_like = None
like_training_plot = None
if args.compute_log_like:
    # calculate average log-likelihood at each iteration for both models
    print('calculating log-likelihoods...')
    av_log_like_lnce = optimiser.av_log_like_for_each_iter(X, thetas=reduced_lnce_thetas)
    av_log_like_cd = cd_optimiser.av_log_like_for_each_iter(X, thetas=reduced_cd_thetas)
    av_log_like_nce = nce_optimiser.av_log_like_for_each_iter(X, thetas=reduced_nce_thetas)
    print('finished!')

    # calculate average log-likelihood of initial and true models
    true_log_like = average_log_likelihood(true_data_dist, X)
    init_log_like = average_log_likelihood(init_model, X)
    noise_log_like = np.mean(np.log(noise_dist(X)))

    training_curves = [[reduced_lnce_times, av_log_like_lnce, 'lnce'],
                       [reduced_cd_times, av_log_like_cd, 'cd'],
                       [reduced_nce_times, av_log_like_nce, 'nce']]
    static_lines = [[true_log_like, 'true'],
                    [init_log_like, 'initial'],
                    [noise_log_like, 'noise']]

    # plot log-likelihood during training
    like_training_plot = plot_log_likelihood_training_curves(training_curves, static_lines)
    like_training_plot.savefig('{}/likelihood-optimisation-curve.pdf'.format(SAVE_DIR))

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
config = vars(args)
config.update({'start_time': START_TIME, 'end_time': END_TIME, 'random_seed': 1083463236})
with open(os.path.join(SAVE_DIR, "config.txt"), 'w') as f:
    for key, value in config.items():
        f.write("{}: {}\n".format(key, value))

pickle.dump(config, open(os.path.join(SAVE_DIR, "config.p"), "wb"))
pickle.dump(like_training_plot, open(os.path.join(SAVE_DIR, "likelihood_training_plot.p"), "wb"))
pickle.dump(J_plot, open(os.path.join(SAVE_DIR, "J_plot.p"), "wb"))
pickle.dump(rbm_weights_plot, open(os.path.join(SAVE_DIR, "rbm_weights_plot.p"), "wb"))
pickle.dump(optimiser, open(os.path.join(SAVE_DIR, "vnce_optimiser.p"), "wb"))
pickle.dump(cd_optimiser, open(os.path.join(SAVE_DIR, "cd_optimiser.p"), "wb"))
pickle.dump(nce_optimiser, open(os.path.join(SAVE_DIR, "nce_optimiser.p"), "wb"))

np.savez(os.path.join(SAVE_DIR, "data"), X=X, Y=optimiser.Y)
np.savez(os.path.join(SAVE_DIR, "true_weights_and_likelihood"), true_theta=true_theta, ll=true_log_like)
np.savez(os.path.join(SAVE_DIR, "init_theta_and_likelihood"), theta0=theta0, ll=init_log_like)
np.savez(os.path.join(SAVE_DIR, "vnce_results"), params=optimiser.thetas, times=optimiser.times, J1s=optimiser.J1s,
         reduced_params=reduced_lnce_thetas, reduced_times=reduced_lnce_times, ll=av_log_like_lnce, E_step_ids=optimiser.E_step_ids)
np.savez(os.path.join(SAVE_DIR, "cd_results"), params=cd_optimiser.thetas, times=cd_optimiser.times,
         reduced_params=reduced_cd_thetas, reduced_times=reduced_cd_times, ll=av_log_like_cd)
np.savez(os.path.join(SAVE_DIR, "nce_results"), params=nce_optimiser.thetas, times=nce_optimiser.times,
         Js=nce_optimiser.Js, Js_for_lnce_thetas=Js_for_lnce_thetas, reduced_params=reduced_nce_thetas,
         reduced_times=reduced_nce_times, ll=av_log_like_lnce)
