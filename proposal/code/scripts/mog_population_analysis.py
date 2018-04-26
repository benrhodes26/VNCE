# todo: docstring description of script here

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np
import pdb

# my code
from distribution import PolynomialSigmoidBernoulli, GaussianNoise
from gaussian_mixture_analytic_expectations import *
from fully_observed_models import SumOfTwoUnnormalisedGaussians
from latent_variable_model import MixtureOfTwoUnnormalisedGaussians
from mle_optimiser import MLEOptimiser
from nce_optimiser import NCEOptimiser
from utils import *
from vnce_optimisers import VNCEOptimiserWithAnalyticExpectations, VNCEOptimiser

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from itertools import product
from matplotlib import pyplot as plt
from numpy import random as rnd

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--save_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/'
                                                    'ben-rhodes-masters-project/experimental-results',
                    help='Path to directory where model will be saved')
parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')

# Data arguments
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=float, default=1.0, help='ratio of noise to data samples in NCE')

# Model arguments
parser.add_argument('--d', type=int, default=20, help='dimension of visibles')
parser.add_argument('--m', type=int, default=15, help='dimension of hiddens')
parser.add_argument('--sigma1', type=float, default=4.0, help='fixed standard deviation of gaussian in unnormalised MoG')

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

# parameters of simulation
parser.add_argument('--num_random_inits', type=int, default=5,
                    help='number of random initialisations of theta (to avoid local minima)')
parser.add_argument('--num_runs', type=int, default=500,
                    help='number of runs of the simulation (we sample a new data set each run)')
parser.add_argument('--sigma0_low', type=int, default=1,
                    help='number of runs of the simulation (we sample a new data set each run)')
parser.add_argument('--sigma0_high', type=int, default=9,
                    help='number of runs of the simulation (we sample a new data set each run)')
args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir, args.exp_name, args.name)
os.makedirs(SAVE_DIR)

DEFAULT_SEED = 909637236
rng = rnd.RandomState(DEFAULT_SEED)

"""
==========================================================================================================
                                            SETUP
==========================================================================================================
"""

samples_sizes = np.array([100, 500, 2500, 12500])

# E(r(x, z))
E1 = E_r
# E(log(psi_1(x, z)))
E2 = E_log_psi_1
# E((psi_1(x, z) - 1) / psi_1(x, z))
E3 = E_psi_1_ratio_times_grad_log_theta
# E(grad_theta(log(phi(u,z)) r(u, z))
E4 = E_r_times_grad_log_theta
# gradient_alpha(E(log(psi_1(x, z)))
E5 = grad_wrt_alpha_of_E_log_psi_1

model = MixtureOfTwoUnnormalisedGaussians(np.array([0, 0]), sigma1=args.sigma1, rng=rng)
nce_model = SumOfTwoUnnormalisedGaussians(np.array([0, 0]), sigma1=sigma1, rng=rng)  # for comparison
# todo: need the normalised sum for mle!
mle_model = SumOfTwoUnnormalisedGaussians(np.array([0]), sigma1=sigma1, rng=rng)  # for comparison
var_dist = PolynomialSigmoidBernoulli(alpha=np.array([0, 0, 0]), rng=rng)


def get_final_mses(runs):
    mses = np.array([mean_square_error(estimate=run[0][-1], true_value=run[2], plot=False) for run in runs])
    return mses


def get_final_mses_no_scaling_param(runs):
    mses = np.array([mean_square_error(estimate=run[0][-1][-1], true_value=run[2][-1], plot=False) for run in runs])
    return mses


def get_final_mses_only_scaling_param(runs):
    mses = np.array([mean_square_error(estimate=run[0][-1][0], true_value=run[2][0], plot=False) for run in runs])
    return mses


def sort_by_mse(runs):
    final_mses = get_final_mses(runs)
    sorted_runs_ids = np.argsort(final_mses)
    sorted_runs = runs[sorted_runs_ids]
    return sorted_runs


def save_deciles(quantile_to_run_dict, sorted_runs):
    """Get 1st, 5th & 9th deciles"""
    lowest_decile_run = sorted_runs[int(len(sorted_runs) / 10)]
    median_run = sorted_runs[int(len(vnce_runs_for_sample) / 2)]
    highest_decile_run = sorted_runs[int(9 * len(sorted_runs) / 10)]

    quantile_to_run_dict['0.1'] = quantile_to_run_dict.get('0.1', []).append(lowest_decile_run)
    quantile_to_run_dict['0.5'] = quantile_to_run_dict.get('0.5', []).append(median_run)
    quantile_to_run_dict['0.9'] = quantile_to_run_dict.get('0.9', []).append(highest_decile_run)

"""
==========================================================================================================
                                         EXPERIMENT
==========================================================================================================
"""
'''
For each value of n in the range (200, 1000, 5000, 25000), we run 500 simulations, drawing a new ground-truth
sigma0 from the interval (1, 9) each time, which we then use to sample a synthetic data set. For each of the
500 simulations, we run the optimisation with 5 different random initialisations, and keep the best result of the 5
(i.e the final theta has lowest mean-square error) to avoid local optima.
'''

# map each sample size to a list of tuples (parameters_for_run, times_for_run, ground_truth_thetas_for_run),
# where the list is equal in length to num_runs. Each tuple corresponds to the parameters/times/ground_truths for a particular run.
vnce_runs = {'100': [], '500': [], '2500': [], '12500': []}
nce_runs = {'100': [], '500': [], '2500': [], '12500': []}
mle_runs = {'100': [], '500': [], '2500': [], '12500': []}
true_thetas = np.zeros(args.num_runs)
max_run_time = 0
for i in range(args.num_runs):

    # true value of parameters
    sigma0 = rng.uniform(args.sigma0_low, args.sigma0_high)
    true_c = 0.5*np.log(2*np.pi) + np.log(args.sigma1 + sigma0)
    true_theta = np.array([true_c, np.log(sigma0)])
    true_thetas[i] = deepcopy(true_theta)

    for j, sample_size in enumerate(samples_sizes):

        # generate data
        true_data_dist = MixtureOfTwoUnnormalisedGaussians(theta=true_theta, sigma1=args.sigma1, rng=rng)
        X = true_data_dist.sample(sample_size)

        # generate noise
        noise = GaussianNoise(mean=0, cov=sigma0**2, rng=rng)
        Y = noise.sample(int(sample_size*args.nu))

        # optimise models for multiple random initialisations and take the best (to avoid bad local optima)
        vnce_rnd_init_runs = []
        nce_rnd_init_runs = []
        mle_rnd_init_runs = []
        for k in range(args.num_random_inits):

            # random initial values for optimisation
            theta0 = np.array([0, np.log(rnd.uniform(1, 9))])
            alpha0 = np.array([0, 0, 0])

            # construct optimisers
            optimiser = VNCEOptimiserWithAnalyticExpectations(model=model, noise=noise, noise_samples=Y, variational_dist=var_dist, sample_size=n,
                                                              E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, nu=args.nu, latent_samples_per_datapoint=args.nz, eps=10**-20, rng=rng)
            nce_optimiser = NCEOptimiser(model=nce_model, noise=noise, noise_samples=Y, nu=args.nu, eps=10**-20)
            mle_optimiser = MLEOptimiser(model=mle_model)

            # run vnce optimisation
            optimiser.fit(X, theta0=theta0, alpha0=alpha0, opt_method=args.opt_method, ftol=args.ftol, maxiter=args.maxiter,
                          stop_threshold=args.stop_threshold, learning_rate=args.learning_rate, batch_size=args.batch_size,
                          max_num_em_steps=args.max_num_em_steps, separate_terms=args.separate_terms, disp=False)

            # run nce optimisation
            nce_optimiser.fit(X, theta0=theta0, disp=False, ftol=args.ftol, maxiter=args.maxiter, separate_terms=args.separate_terms)

            # maximum likelihood optimisation
            mle_optimiser.fit(X, theta0=theta0, disp=False, ftol=args.ftol, maxiter=args.maxiter)

            # save all optimisation results to temp store
            vnce_rnd_init_runs.append((optimiser.thetas, optimiser.times, true_theta))
            nce_rnd_init_runs.append((nce_optimiser.thetas, nce_optimiser.times, true_theta))
            mle_rnd_init_runs.append((mle_model.thetas, mle_optimiser.times, true_theta))

            max_run_time = max(max_run_time, optimiser.times[-1], nce_optimiser.times[-1], mle_optimiser.times[-1])

        # keep only the best randomly initialised model
        vnce_runs[str(n)].append(vnce_rnd_init_runs[np.argmin(get_final_mses(vnce_rnd_init_runs))])
        nce_runs[str(n)].append(nce_rnd_init_runs[np.argmin(get_final_mses(nce_rnd_init_runs))])
        mle_runs[str(n)].append(mle_rnd_init_runs[np.argmin(get_final_mses(mle_rnd_init_runs))])

"""
==========================================================================================================
                                           METRICS & PLOTTING
==========================================================================================================
"""
# For each sample size (200, 1000, 5000, 25000), we find the 0.1, 0.5 & 0.9 quantiles of the mean-squared
# error of the parameter estimates out of a population of 500 runs.
vnce_quantile_to_run_dict = {}
nce_quantile_to_run_dict = {}
mle_quantile_to_run_dict = {}
# get indices of the runs corresponding to 0.1, 0.5 & 0.9 deciles
for j, sample_size in enumerate(samples_sizes):

    vnce_runs_for_sample = vnce_runs[str(sample_size)]
    nce_runs_for_sample = nce_runs[str(sample_size)]
    mle_runs_for_sample = mle_runs[str(sample_size)]

    # sort the runs by mean-squared error of final parameter estimate
    sorted_vnce_runs_for_sample = sort_by_mse(vnce_runs_for_sample)
    sorted_nce_runs_for_sample = sort_by_mse(nce_runs_for_sample)
    sorted_mle_runs_for_sample = sort_by_mse(mle_runs_for_sample)

    # save the 0.1, 0.5 & 0.9 quantiles
    save_deciles(vnce_quantile_to_run_dict, sorted_vnce_runs_for_sample)
    save_deciles(nce_quantile_to_run_dict, sorted_nce_runs_for_sample)
    save_deciles(mle_quantile_to_run_dict, sorted_mle_runs_for_sample)


def plot1(ax, quant_to_run_dict, sample_sizes, colour, label):
    for quantile, runs in quant_to_run_dict.items():
        final_mses_no_scaling_param = get_final_mses_no_scaling_param(runs)
        if quantile == '0.5':
            ax.plot(np.log10(sample_sizes), np.log10(final_mses_no_scaling_param), c=colour, label=label)
        else:
            ax.plot(np.log10(sample_sizes), np.log10(final_mses_no_scaling_param), c=colour, alpha=0.3)


def plot2(ax, quant_to_run_dict, sample_sizes, colour, label):
    for quantile, runs in quant_to_run_dict.items():
        final_mses_scaling_param = get_final_mses_only_scaling_param(runs)
        if quantile == '0.5':
            ax.plot(np.log10(sample_sizes), np.log10(final_mses_scaling_param), c=colour, label=label)
        else:
            ax.plot(np.log10(sample_sizes), np.log10(final_mses_scaling_param), c=colour, alpha=0.3)

# PLOT 1: log_10 sample size against log_10 mean-squared error
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
plot1(ax1, vnce_quantile_to_run_dict, sample_sizes, colour='r', label='VNCE')
plot1(ax1, nce_quantile_to_run_dict, sample_sizes, colour='b', label='NCE')
plot1(ax1, mle_quantile_to_run_dict, sample_sizes, colour='k', label='MLE')

# PLOT 2: log_10 sample size against log_10 mean-squared error
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
plot2(ax2, vnce_quantile_to_run_dict, sample_sizes, colour='r', label='VNCE')
plot2(ax2, nce_quantile_to_run_dict, sample_sizes, colour='b', label='NCE')


# Now we generate 'trade-off' curves such as those in fig1b of http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf
def make_tradeoff_curves(times, runs):
    median_min_mse = np.zeros(len(times))
    all_upper_bounds = []  # tuples of (simulation_id, sample_size) of upper bounds I want to plot
    for i, t in enumerate(times):
        save_upper_bounds = i % 100 == 0
        upper_bounds = []
        min_mses = []
        for j in range(args.num_runs):

            mses_for_diff_sample_sizes = []
            for sample_size in samples_sizes:
                run = runs[str(sample_size)][j]  # triple (thetas, times, ground_truth_thetas)
                time_id = takeClosest(run[1], t)
                mse = mean_square_error(estimate=run[0][time_id], true_value=run[2][-1], plot=False)
                mses_for_diff_sample_sizes.append(mse)

            # take min over all sample sizes
            min_mse, min_mse_id = np.min(mses_for_diff_sample_sizes), np.argmin(mses_for_diff_sample_sizes)
            min_mses.append(min_mse)

            if save_upper_bounds:
                upper_bounds.append(samples_sizes[min_mse_id])  # save best sample size in terms of mse at this time, t

        # take median over all simulations
        arg_median_min_mse = np.argsort(min_mses)[int(len(min_mses) / 2)]
        median_min_mse[i] = min_mses[arg_median_min_mse]

        if save_upper_bounds:
            median_min_sample_size = upper_bounds[arg_median_min_mse]
            all_upper_bounds.append(
                (arg_median_min_mse, median_min_sample_size))  # (simulation_id, sample_size) of upper bound to plot

    upper_bound_curves = get_upper_bound_curves(all_upper_bounds, runs)

    return median_min_mse, upper_bound_curves   # (len(times), ) , list of 5 [times, mses] for different upper bounds


def get_upper_bound_curves(all_upper_bounds, runs):
    upper_bound_runs = []
    for sim_id, sample_size in all_upper_bounds:
        run = runs[str(sample_size)][sim_id]
        run_length = len(run[0])
        run_mses = [mean_square_error(estimate=run[0][i], true_value=run[2][i], plot=False) for i in range(run_length)]
        upper_bound_runs.append((run[1], run_mses))

    return upper_bound_runs  # list of 5 [times, mses] for different upper bounds


def plot3(ax, times, median_min_mses, label, colour, upper_bounds=None):
    ax.plot(times, median_min_mses, label=label, c=colour)
    if upper_bounds is not None:
        for t, mses in upper_bounds:
            ax.plot(t, mses, c=colour, alpha=0.3)

times = 10**(np.linspace(-1, np.log10(max_run_time), num=500))
vnce_median_min_mses, vnce_upper_bounds = make_tradeoff_curves(times, vnce_runs)
nce_median_min_mses, nce_upper_bounds = make_tradeoff_curves(times, nce_runs)
mle_median_min_mses, mle_upper_bounds = make_tradeoff_curves(times, mle_runs)

# PLOT 3: trade-off curve for VNCE with upper bound curves
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))
plot3(ax3, times, vnce_median_min_mses, label='VNCE', colour='r', upper_bounds=vnce_upper_bounds)

# PLOT 4: trade-off curve for all estimation methods
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 7))
plot3(ax4, times, vnce_median_min_mses, label='VNCE', colour='r', upper_bounds=vnce_upper_bounds)
plot3(ax4, times, nce_median_min_mses, label='NCE', colour='b', upper_bounds=nce_upper_bounds)
plot3(ax4, times, mle_median_min_mses, label='MLE', colour='k', upper_bounds=mle_upper_bounds)

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

# save arrays containing the parameters
