import numpy as np
import os
import pdb
import pickle

# my code
from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from latent_nce_optimiser import LatentNCEOptimiser
from latent_variable_model import RestrictedBoltzmannMachine
from utils import get_true_weights, plot_rbm_parameters, plot_rbm_log_likelihood_training_curves,\
    rescale_cd_times, average_log_likelihood

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd
from time import gmtime, strftime

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

# For reproducibility
rng = rnd.RandomState(1083463236)
parser = ArgumentParser(description='Experimental comparison of training an RBM using latent '
                        'nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# read/write arguments
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
parser.add_argument('--init_visible_biases_match_marginals', dest='init_visible_bias', action='store_true',
                    help='initialise the ith visible bias to be log(p_i/(1-p_i)), where p_i is the '
                         'probability of the ith visible unit being equal to 1, as estimated from data')
parser.add_argument('--no-init_visible_biases_match_marginals', dest='init_visible_bias', action='store_false',
                    help='see help documentation of --init_visible_biases_match_marginals')
parser.set_defaults(init_visible_bias=True)

# Latent NCE optimisation arguments
parser.add_argument('--noise', type=str, default='marginal',
                    help='type of noise distribution for latent NCE. Currently, this'
                         'can be either marginals or chow-liu')
parser.add_argument('--ftol', type=float, default=1e-5,
                    help='Tolerance used as stopping criterion in scipy.minimize')
parser.add_argument('--maxiter', type=int, default=10,
                    help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=1e-5,
                    help='Tolerance used as stopping criterion in scipy.minimize')
parser.add_argument('--max_num_em_steps', type=int, default=20,
                    help='Maximum number of EM steps to perform')

# Contrastive divergence optimisation arguments
parser.add_argument('--cd_num_steps', type=int, default=1, help='number of gibbs steps used to sample from '
                                                                'model during learning with CD')
parser.add_argument('--cd_learn_rate', type=float, default=0.01,
                    help='Initial learning rate for contrastive divergence')
parser.add_argument('--cd_batch_size', type=int, default=10, help='number of datapoints used per gradient update')
parser.add_argument('--cd_num_epochs', type=int, default=100, help='number of passes through data set')
args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir, args.name)
os.mkdir(SAVE_DIR)

n, nz, nu = args.n, args.nz, args.nu
d, m = args.d, args.m

# todo: get realistic weights learnt from downsampled mnist digits
# generate weights of RBM that we want to learn
true_W = rnd.randn(d+1, m+1)
true_W[0, 0] = 0
theta0 = rnd.uniform(-0.1, 0.1, (d+1, m+1))  # random initialisation of weights
theta0[0, 0] = -(d+m)*np.log(2)

# generate synthetic dataset
true_data_dist = RestrictedBoltzmannMachine(true_W)
X, Z = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
X_mean = np.mean(X, axis=0)
if args.init_visible_bias:
    theta0[1:, 0] = np.log(X_mean / (1 - X_mean))
    theta0[0, 0] = -m*np.log(2) + np.sum(np.log(1 - X_mean))

# initialise models
model = RestrictedBoltzmannMachine(theta0)
cd_model = RestrictedBoltzmannMachine(theta0)
init_model = RestrictedBoltzmannMachine(theta0)  # rbm with initial parameters, for comparison

# initialise noise distributions for latent NCE
noise_dist = None
if args.noise == 'marginal':
    noise_dist = MultivariateBernoulliNoise(X_mean)
if args.noise == 'chow_liu':
    noise_dist = ChowLiuTree(X)

# initialise variational distribution, which is the exact posterior in this case
var_dist = RBMLatentPosterior(theta0)

# initialise optimisers
optimiser = LatentNCEOptimiser(model, noise_dist, var_dist, n, nu=nu, latent_samples_per_datapoint=nz)
cd_optimiser = CDOptimiser(cd_model)

# calculate normalisation constant of true model
true_data_dist.reset_norm_const()
true_norm_const = true_data_dist.norm_const
true_W[0, 0] = -np.log(true_norm_const)

# perform latent nce optimisation
lnce_thetas, J1s, lnce_times = \
    optimiser.fit_using_analytic_q(X,
                                   theta0=model.theta,
                                   ftol=args.ftol,
                                   maxiter=args.maxiter,
                                   stop_threshold=args.stop_threshold,
                                   max_num_em_steps=args.max_num_em_steps,
                                   disp=True,
                                   plot=False)

# perform contrastive divergence optimisation
cd_thetas, cd_times = cd_optimiser.fit(X,
                                       theta0=theta0.reshape(-1),
                                       num_gibbs_steps=args.cd_num_steps,
                                       learning_rate=args.cd_learn_rate,
                                       batch_size=args.cd_batch_size,
                                       num_epochs=args.cd_num_epochs)

# rescale cd timings so they are comparable to lnce timings
cd_times, cd_time_ids = rescale_cd_times(lnce_times, cd_times)
cd_thetas = cd_thetas[cd_time_ids]
cd_optimiser.times, cd_optimiser.thetas = cd_times, cd_thetas

# calculate average log-likelihood at each iteration for both models
av_log_like_lnce = optimiser.av_log_like_for_each_iter(X)
av_log_like_cd = cd_optimiser.av_log_like_for_each_iter(X)
# calculate average log-likelihood of initial and true models
true_log_likelihood = average_log_likelihood(true_data_dist, X)
init_log_likelihood = average_log_likelihood(init_model, X)

# plot log-likelihood during training
likelihood_training_plot = plot_rbm_log_likelihood_training_curves(av_log_like_lnce,
                                                                   av_log_like_cd,
                                                                   lnce_times,
                                                                   cd_times,
                                                                   init_ll=init_log_likelihood,
                                                                   true_ll=true_log_likelihood,
                                                                   end=None)
likelihood_training_plot.savefig('{}/likelihood-optimisation-curve.pdf'.format(SAVE_DIR))

# plot J1 (lower bound to NCE objective) during training
J1_plot = optimiser.plot_loss_curve(end=None, true_theta=true_W.reshape(-1), X=X)
J1_plot.savefig('{}/J1-optimisation-curve.pdf'.format(SAVE_DIR))

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

# save everything, in case we need it later
hyperparam_dict = {'n': n, 'nz': nz, 'nu': nu, 'd': d, 'm': m,
                   'num_gibbs_steps': args.num_gibbs_steps, 'noise': args.noise,
                   'ftol': args.ftol, 'maxiter': args.maxiter, 'stop_threshold': args.stop_threshold,
                   'cd_num_steps': args.cd_num_steps, 'cd_learn_rate': args.cd_learn_rate,
                   'cd_batch_size': args.cd_batch_size, 'cd_num_epochs': args.cd_num_epochs,
                   'random_seed': 1083463236}
pickle.dump(likelihood_training_plot, open(os.path.join(SAVE_DIR, "likelihood_training_plot"), "wb"))
pickle.dump(J1_plot, open(os.path.join(SAVE_DIR, "J1_plot"), "wb"))
pickle.dump(rbm_weights_plot, open(os.path.join(SAVE_DIR, "rbm_weights_plot"), "wb"))
pickle.dump(optimiser, open(os.path.join(SAVE_DIR, "lnce_optimiser"), "wb"))
pickle.dump(cd_optimiser, open(os.path.join(SAVE_DIR, "cd_optimiser"), "wb"))

np.savez(os.path.join(SAVE_DIR, "data"), X=X, Y=optimiser.Y)
np.savez(os.path.join(SAVE_DIR, "true_W"), true_W=true_W, ll=true_log_likelihood)
np.savez(os.path.join(SAVE_DIR, "theta0"), theta0=theta0, ll=init_log_likelihood)
np.savez(os.path.join(SAVE_DIR, "lnce_learning"), params=lnce_thetas, J1s=J1s, times=lnce_times, ll=av_log_like_lnce)
np.savez(os.path.join(SAVE_DIR, "cd_learning"), params=cd_thetas, times=cd_times, ll=av_log_like_cd)
