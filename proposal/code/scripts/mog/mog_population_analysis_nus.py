# todo: docstring description of script here

import os
import sys
nb_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np
import pickle

# my code
from distribution import PolynomialSigmoidBernoulli, GaussianNoise
from gaussian_mixture_analytic_expectations import *
from fully_observed_models import SumOfTwoUnnormalisedGaussians, SumOfTwoNormalisedGaussians
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
from time import strftime, gmtime

START_TIME = strftime('%Y%m%d-%H%M', gmtime())
parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--save_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/mog',
                    help='Path to directory where model will be saved')
parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')

# Data arguments
parser.add_argument('--n', type=int, default=10000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')

# Model arguments
parser.add_argument('--sigma1', type=float, default=4.0, help='fixed standard deviation of gaussian in unnormalised MoG')

# Latent NCE optimisation arguments
parser.add_argument('--opt_method', type=str, default='CG',
                    help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--ftol', type=float, default=2.220446049250313e-09,
                    help='Tolerance used as stopping criterion in scipy.minimize')
parser.add_argument('--maxiter', type=int, default=15000,
                    help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=1e-09,
                    help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_em_steps', type=int, default=100,
                    help='Maximum number of EM steps to perform')
parser.add_argument('--learn_rate', type=float, default=0.01,
                    help='if opt_method=SGD, this is the learning rate used')
parser.add_argument('--batch_size', type=int, default=10,
                    help='if opt_method=SGD, this is the size of a minibatch')
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=False)

# NCE optimisation arguments
parser.add_argument('--nce_maxiter', type=int, default=15000, help='max number of iterations for NCE')
parser.add_argument('--mle_maxiter', type=int, default=15000, help='max number of iterations for MLE')

# parameters of simulation
parser.add_argument('--num_random_inits', type=int, default=5,
                    help='number of random initialisations of theta (to avoid local minima)')
parser.add_argument('--num_runs', type=int, default=500,
                    help='number of runs of the simulation (we sample a new data set each run)')
parser.add_argument('--sigma0_low', type=int, default=2,
                    help='number of runs of the simulation (we sample a new data set each run)')
parser.add_argument('--sigma0_high', type=int, default=6,
                    help='number of runs of the simulation (we sample a new data set each run)')

args = parser.parse_args()


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
    median_run = sorted_runs[int(len(sorted_runs) / 2)]
    highest_decile_run = sorted_runs[int(9 * len(sorted_runs) / 10)]

    quantile_to_run_dict['0.1'] = quantile_to_run_dict.get('0.1', []) + [lowest_decile_run]
    quantile_to_run_dict['0.5'] = quantile_to_run_dict.get('0.5', []) + [median_run]
    quantile_to_run_dict['0.9'] = quantile_to_run_dict.get('0.9', []) + [highest_decile_run]


def make_quantile_to_run_dict(runs, nus):
    """Returns quantile_to_run_dicts: a mapping from a quantile (e.g the median'0.5') to a list of triples
    (thetas, times, true_theta), where the final estimate thetas[-1] has the median MSE of all runs
    for the given estimation method. The list contains one run (i.e triple) for every nu val.
    """

    quantile_to_run_dict = {}
    for j, nu in enumerate(nus):
        runs_for_nu = runs[str(nu)]

        # sort the runs by mean-squared error of final parameter estimate
        sorted_runs_for_nu = sort_by_mse(np.array(runs_for_nu))

        # save the 0.1, 0.5 & 0.9 quantiles
        save_deciles(quantile_to_run_dict, sorted_runs_for_nu)

    return quantile_to_run_dict


def make_tradeoff_curves(times, runs, nus):
    median_min_mse = np.zeros(len(times))
    all_upper_bounds = []  # tuples of (simulation_id, nu) of upper bounds I want to plot
    for i, t in enumerate(times):
        save_upper_bounds = i % 100 == 0
        upper_bounds = []
        min_mses = []
        for j in range(args.num_runs):

            mses_for_diff_nus = []
            for nu in nus:
                run = runs[str(nu)][j]  # triple (thetas, times, ground_truth_thetas)
                time_id = take_closest(run[1], t)
                mse = mean_square_error(estimate=run[0][time_id][-1], true_value=run[2][-1], plot=False)
                mses_for_diff_nus.append(mse)

            # take min over all nu vals
            min_mse, min_mse_id = np.min(mses_for_diff_nus), np.argmin(mses_for_diff_nus)
            min_mses.append(min_mse)

            if save_upper_bounds:
                upper_bounds.append(nus[min_mse_id])  # save best nu val in terms of mse at this time, t

        # take median over all simulations
        arg_median_min_mse = np.argsort(min_mses)[int(len(min_mses) / 2)]
        median_min_mse[i] = min_mses[arg_median_min_mse]

        if save_upper_bounds:
            median_min_nu_val = upper_bounds[arg_median_min_mse]
            all_upper_bounds.append(
                (arg_median_min_mse, median_min_nu_val))  # (simulation_id, nu) of upper bound to plot

    upper_bound_curves = get_upper_bound_curves(all_upper_bounds, runs)

    return median_min_mse, upper_bound_curves   # (len(times), ) , list of 5 [times, mses] for different upper bounds


def get_upper_bound_curves(all_upper_bounds, runs):
    upper_bound_runs = []
    for sim_id, nu in all_upper_bounds:
        run = runs[str(nu)][sim_id]
        run_length = len(run[0])
        run_mses = [mean_square_error(estimate=run[0][i][-1], true_value=run[2][-1], plot=False) for i in range(run_length)]
        upper_bound_runs.append((run[1], run_mses))

    return upper_bound_runs  # list of 5 [times, mses] for different upper bounds


def unroll_runs(runs):
    unrolled_dict = {}
    for key, val in runs.items():
        for i, run in enumerate(val):
            unrolled_dict['nu:{}_run:{}_thetas'.format(key, i)] = run[0]
            unrolled_dict['nu:{}_run:{}_times'.format(key, i)] = run[1]
            unrolled_dict['nu:{}_run:{}_true_theta'.format(key, i)] = run[2]

    return unrolled_dict


# noinspection PyShadowingNames
def run_experiment(args, rng):
    """For each value of nu in the range (1, 5, 10), we run 500 simulations, drawing a new ground-truth
    sigma0 from the interval (args.low_sigma0, args.high_sigma0) each time, which we then use to sample a synthetic data set.
    For each of the 500 simulations, we run the optimisation with 5 different random initialisations, and keep the best
    result of the 5 (i.e the final theta has lowest mean-square error) to avoid local optima.
    """
    nus = np.array([1, 5, 10])

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
    nce_model = SumOfTwoUnnormalisedGaussians(np.array([0, 0]), sigma1=args.sigma1, rng=rng)  # for comparison
    mle_model = SumOfTwoNormalisedGaussians(np.array([0]), sigma1=args.sigma1, rng=rng)  # for comparison
    var_dist = PolynomialSigmoidBernoulli(alpha=np.array([0, 0, 0]), rng=rng)

    # map each nu val to a list of tuples (parameters_for_run, times_for_run, ground_truth_theta_for_run),
    # where the list is equal in length to num_runs. Each tuple corresponds to the parameters/times/ground_truth for a particular run.
    vnce_runs = {str(i): [] for i in nus}
    nce_runs = {str(i): [] for i in nus}
    mle_runs = {str(i): [] for i in nus}
    max_run_time = 0
    for i in range(args.num_runs):

        # true value of parameters
        sigma0 = rng.uniform(args.sigma0_low, args.sigma0_high)
        true_c = 0.5 * np.log(2 * np.pi) + np.log(args.sigma1 + sigma0)
        true_theta = np.array([true_c, np.log(sigma0)])

        for j, nu in enumerate(nus):

            # generate data
            true_data_dist = MixtureOfTwoUnnormalisedGaussians(theta=true_theta, sigma1=args.sigma1, rng=rng)
            X = true_data_dist.sample(args.n)

            # generate noise
            noise = GaussianNoise(mean=0, cov=sigma0 ** 2, rng=rng)
            Y = noise.sample(int(args.n * nu))

            # optimise models for multiple random initialisations and take the best (to avoid bad local optima)
            vnce_rnd_init_runs = []
            nce_rnd_init_runs = []
            mle_rnd_init_runs = []
            for k in range(args.num_random_inits):
                # random initial values for optimisation
                theta0 = np.array([0, np.log(rnd.uniform(1, 9))])
                alpha_2 = 0.5*((1/theta0[1]**2) + (1/args.sigma1**2))
                alpha0 = np.array([0, 0, alpha_2])

                # construct optimisers
                optimiser = VNCEOptimiserWithAnalyticExpectations(model=model, noise=noise, noise_samples=Y,
                                                                  variational_dist=var_dist, E1=E1, E2=E2, E3=E3, E4=E4,
                                                                  E5=E5, nu=nu, eps=10 ** -20, rng=rng)
                nce_optimiser = NCEOptimiser(model=nce_model, noise=noise, noise_samples=Y, nu=nu, eps=10 ** -20)
                mle_optimiser = MLEOptimiser(model=mle_model)

                # run vnce optimisation
                optimiser.fit(X, theta0=theta0, alpha0=alpha0, opt_method=args.opt_method, ftol=args.ftol,
                              maxiter=args.maxiter, stop_threshold=args.stop_threshold, learning_rate=args.learn_rate,
                              batch_size=args.batch_size, max_num_em_steps=args.max_num_em_steps, separate_terms=args.separate_terms, disp=False)

                # run nce optimisation
                nce_optimiser.fit(X, theta0=theta0, disp=False, ftol=args.ftol, maxiter=args.nce_maxiter, separate_terms=args.separate_terms)

                # maximum likelihood optimisation
                mle_optimiser.fit(X, theta0=theta0[1], disp=False, ftol=args.ftol, maxiter=args.mle_maxiter)

                # convert stdev & normalising parameters back from log-domain
                exp_true_theta = np.exp(deepcopy(true_theta))
                optimiser_thetas = np.exp(deepcopy(optimiser.thetas))
                nce_optimiser_thetas = np.exp(deepcopy(nce_optimiser.thetas))
                mle_optimiser_thetas = np.exp(deepcopy(mle_optimiser.thetas))

                # get timings for the M steps (ignoring E steps)
                optimiser_times = deepcopy(optimiser.times[optimiser.M_step_ids])

                # save all optimisation results to temp store
                vnce_rnd_init_runs.append((optimiser_thetas, optimiser_times, exp_true_theta))
                nce_rnd_init_runs.append((nce_optimiser_thetas, deepcopy(nce_optimiser.times), exp_true_theta))
                mle_rnd_init_runs.append((mle_optimiser_thetas, deepcopy(mle_optimiser.times), np.array([exp_true_theta[1]])))  # ignore scaling param

                max_run_time = max(max_run_time, deepcopy(optimiser.times[-1]), deepcopy(nce_optimiser.times[-1]), deepcopy(mle_optimiser.times[-1]))

            # keep only the best randomly initialised model
            vnce_runs[str(nu)].append(vnce_rnd_init_runs[np.argmin(get_final_mses(vnce_rnd_init_runs))])
            nce_runs[str(nu)].append(nce_rnd_init_runs[np.argmin(get_final_mses(nce_rnd_init_runs))])
            mle_runs[str(nu)].append(mle_rnd_init_runs[np.argmin(get_final_mses(mle_rnd_init_runs))])

    return nus, vnce_runs, nce_runs, mle_runs, max_run_time


def plot1(ax, quant_to_run_dict, nus, colour, label):
    for quantile, runs in quant_to_run_dict.items():
        final_mses_no_scaling_param = get_final_mses_no_scaling_param(runs)
        if quantile == '0.5':
            ax.plot(nus, np.log10(final_mses_no_scaling_param), c=colour, linewidth=1.5, label=label, marker='o')
        else:
            ax.plot(nus, np.log10(final_mses_no_scaling_param), c=colour, alpha=0.5, marker='o', linestyle='--')
    ax.set_xlabel('ratio of noise to data samples')
    ax.set_ylabel('log 10 mean squared error')
    ax.grid()
    ax.legend()


def plot2(ax, quant_to_run_dict, nus, colour, label):
    for quantile, runs in quant_to_run_dict.items():
        final_mses_scaling_param = get_final_mses_only_scaling_param(runs)
        if quantile == '0.5':
            ax.plot(nus, np.log10(final_mses_scaling_param), c=colour, linewidth=1.5, label=label, marker='o')
        else:
            ax.plot(nus, np.log10(final_mses_scaling_param), c=colour, alpha=0.5, marker='o', linestyle='--')
    ax.set_xlabel('ratio of noise to data samples')
    ax.set_ylabel('log 10 mean squared error normalising parameter')
    ax.grid()
    ax.legend()


def plot3(ax, times, median_min_mses, label, colour, upper_bounds=None):
    ax.plot(np.log10(times), np.log10(median_min_mses), label=label, c=colour, linewidth=1.5)
    if upper_bounds is not None:
        for t, mses in upper_bounds:
            t = np.array([time + 1e-3 if time == 0 else time for time in t])
            ax.plot(np.log10(t), np.log10(mses), c=colour, alpha=0.5, linestyle='--')
    ax.set_xlabel('log 10 time (in seconds)')
    ax.set_ylabel('log 10 mean squared error')
    ax.grid()
    ax.legend()


def plot_asymptotic_efficiency_curves(save_dir, nus, vnce_runs, nce_runs, mle_runs, max_run_time):

    # For each nu value in (1, 5, 10), we find the 0.1, 0.5 & 0.9 quantiles of the mean-squared
    # error of the parameter estimates out of a population of 500 runs.
    vnce_quantile_to_run_dict = make_quantile_to_run_dict(vnce_runs, nus)
    nce_quantile_to_run_dict = make_quantile_to_run_dict(nce_runs, nus)
    mle_quantile_to_run_dict = make_quantile_to_run_dict(mle_runs, nus)

    # PLOT 1: nu against log_10 mean-squared error
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    plot1(ax1, vnce_quantile_to_run_dict, nus, colour='r', label='VNCE')
    plot1(ax1, nce_quantile_to_run_dict, nus, colour='b', label='NCE')
    plot1(ax1, mle_quantile_to_run_dict, nus, colour='k', label='MLE')
    fig1.savefig(os.path.join(save_dir, 'nu-against-mse.pdf'))

    # PLOT 2: nu against log_10 mean-squared error
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
    plot2(ax2, vnce_quantile_to_run_dict, nus, colour='r', label='VNCE')
    plot2(ax2, nce_quantile_to_run_dict, nus, colour='b', label='NCE')
    fig2.savefig(os.path.join(save_dir, 'nu-against-mse-scaling-param.pdf'))

    # save objects used for plotting and the plots themselves
    pickle.dump(vnce_quantile_to_run_dict, open(os.path.join(save_dir, "vnce_quantile_to_run_dict.p"), "wb"))
    pickle.dump(nce_quantile_to_run_dict, open(os.path.join(save_dir, "nce_quantile_to_run_dict.p"), "wb"))
    pickle.dump(mle_quantile_to_run_dict, open(os.path.join(save_dir, "mle_quantile_to_run_dict.p"), "wb"))

    pickle.dump(fig1, open(os.path.join(save_dir, "fig1.p"), "wb"))
    pickle.dump(fig2, open(os.path.join(save_dir, "fig2.p"), "wb"))


def plot_tradeoff_curves(save_dir, nus, vnce_runs, nce_runs, mle_runs, max_run_time):

    # Now we generate 'trade-off' curves such as those in fig1b of http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf
    tradeoff_times = 10 ** (np.linspace(-3, np.log10(max_run_time), num=500))
    vnce_median_min_mses, vnce_upper_bounds = make_tradeoff_curves(tradeoff_times, vnce_runs, nus=nus)
    nce_median_min_mses, nce_upper_bounds = make_tradeoff_curves(tradeoff_times, nce_runs, nus=nus)
    mle_median_min_mses, mle_upper_bounds = make_tradeoff_curves(tradeoff_times, mle_runs, nus=nus)

    # PLOT 3: trade-off curve for VNCE with upper bound curves
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    plot3(ax3, tradeoff_times, vnce_median_min_mses, label='VNCE', colour='r', upper_bounds=vnce_upper_bounds)
    fig3.savefig(os.path.join(save_dir, 'tradeoff-curve-vnce.pdf'))

    # PLOT 4: trade-off curve for all estimation methods
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 7))
    plot3(ax4, tradeoff_times, vnce_median_min_mses, label='VNCE', colour='r')
    plot3(ax4, tradeoff_times, nce_median_min_mses, label='NCE', colour='b')
    plot3(ax4, tradeoff_times, mle_median_min_mses, label='MLE', colour='k')
    fig4.savefig(os.path.join(save_dir, 'tradeoff-curves.pdf'))

    # save arrays for plotting the trade-off curves
    np.savez(os.path.join(save_dir, "vnce_tradeoff_curves"), tradeoff_times=tradeoff_times, median_min_mses=vnce_median_min_mses)
    np.savez(os.path.join(save_dir, "nce_tradeoff_curves"), tradeoff_times=tradeoff_times, median_min_mses=nce_median_min_mses)
    np.savez(os.path.join(save_dir, "mle_tradeoff_curves"), tradeoff_times=tradeoff_times, median_min_mses=mle_median_min_mses)

    # A selection of curves that touch the trade-off curve at some point
    pickle.dump(vnce_upper_bounds, open(os.path.join(save_dir, "vnce_upper_bounds.p"), "wb"))
    pickle.dump(nce_upper_bounds, open(os.path.join(save_dir, "nce_upper_bounds.p"), "wb"))
    pickle.dump(mle_upper_bounds, open(os.path.join(save_dir, "mle_upper_bounds.p"), "wb"))

    pickle.dump(fig3, open(os.path.join(save_dir, "fig3.p"), "wb"))
    pickle.dump(fig4, open(os.path.join(save_dir, "fig4.p"), "wb"))


def save_to_file(save_dir, config, vnce_runs, nce_runs, mle_runs):
    """Save the experimental configuration and simulation data to disk"""
    # write config to text file
    with open(os.path.join(save_dir, "config.txt"), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))

    # save config and plots
    pickle.dump(config, open(os.path.join(save_dir, "config.p"), "wb"))

    # save arrays containing the parameters
    vnce_unrolled_dict = unroll_runs(vnce_runs)
    nce_unrolled_dict = unroll_runs(nce_runs)
    mle_unrolled_dict = unroll_runs(mle_runs)
    np.savez(os.path.join(save_dir, "vnce"), **vnce_unrolled_dict)
    np.savez(os.path.join(save_dir, "nce"), **nce_unrolled_dict)
    np.savez(os.path.join(save_dir, "mle"), **mle_unrolled_dict)


# noinspection PyShadowingNames
def main(args, START_TIME):
    """Run population analysis experiment"""
    save_dir = os.path.join(args.save_dir, args.exp_name, args.name)
    os.makedirs(save_dir, exist_ok=True)

    DEFAULT_SEED = 909637236
    rng = rnd.RandomState(DEFAULT_SEED)

    results = run_experiment(args, rng)
    plot_asymptotic_efficiency_curves(save_dir, *results)
    plot_tradeoff_curves(save_dir, *results)

    END_TIME = strftime('%Y%m%d-%H%M', gmtime())
    config = vars(args)
    config.update({'nus': results[0], 'start_time': START_TIME, 'end_time': END_TIME, 'random_seed': DEFAULT_SEED})

    save_to_file(save_dir=save_dir, config=config, vnce_runs=results[1], nce_runs=results[2], mle_runs=results[3])

if __name__ == "__main__":
    main(args, START_TIME=START_TIME)
