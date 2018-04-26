""" Module containing useful functions for experimenting with
the latent NCE code. In particular, multiple plotting functions.
"""
import numpy as np
from bisect import bisect_left
from copy import deepcopy
from matplotlib import pyplot as plt

def mean_square_error(estimate, true_value, plot=True):
    true_values = np.ones_like(estimate)*true_value
    error = estimate - true_values
    square_error = error**2

    if plot:
        num_bins = int((len(square_error))**0.5)
        plt.hist(square_error, bins=num_bins)

    return np.mean(square_error)


def validate_shape(shape, correct_shape):
    """
    :param shape: tuple
        shape to validate
    :param correct_shape: tuple
        correct shape to validate against
    """
    assert shape == correct_shape, 'Expected ' \
        'shape {}, got {} instead'.format(correct_shape, shape)


def sigmoid(u):
    """ Standard sigmoid function
    :param u: array
    :return: array
    """
    return 1/(1 + np.exp(-u))


def get_true_weights(d, m):
    true_W = np.zeros((d+1, m+1))
    num_chosen = 10
    for i in range(m+1):
        chosen = rnd.randint(0, d+1, num_chosen)
        others = [i for i in range(d+1) if i not in chosen]
        true_W[:, i][chosen] = rnd.uniform(0.5, 1, num_chosen)
        true_W[:, i][others] = rnd.uniform(-0.1, 0.1, len(others))
    true_W[0, 0] = 0
    return true_W


def create_J_diff_plot(optimiser, Js_for_lnce_thetas, plot_posterior_ratio=False):
    """plot J (NCE objective function) minus J1 (lower bound to NCE objective)"""
    if plot_posterior_ratio:
        fig, axs = plt.subplots(2, 1, figsize=(15, 20))
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 20))
        axs = [axs]

    ax = axs[0]
    diff = np.sum(Js_for_lnce_thetas, axis=1) - np.sum(optimiser.J1s, axis=1)
    ax.plot(optimiser.times, diff, c='k', label='J - J1')
    diff1 = Js_for_lnce_thetas[:, 0] - optimiser.J1s[:, 0]
    ax.plot(optimiser.times, diff1, c='r', label='term1: J - J1')
    diff2 = Js_for_lnce_thetas[:, 1] - optimiser.J1s[:, 1]
    ax.plot(optimiser.times, diff2, c='b', label='term2: J - J1')

    if plot_posterior_ratio:
        ax = axs[1]
        ax.plot(optimiser.times, optimiser.posterior_ratio_vars, c='k', label='V(p(z|y)/q(z|y))')

    for ax in axs:
        for time_id in optimiser.E_step_ids:
            time = optimiser.times[time_id]
            ax.plot((time, time), ax.get_ylim(), c='0.5')
        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.legend()

    return fig, Js_for_lnce_thetas

def get_Js_for_vnce_thetas(X, nce_optimiser, optimiser, separate_terms):
    cur_theta = deepcopy(nce_optimiser.phi.theta)
    Js_for_vnce_thetas = []
    for i, theta_k in enumerate(optimiser.thetas):
        nce_optimiser.phi.theta = deepcopy(theta_k)
        Js_for_vnce_thetas.append(nce_optimiser.compute_J(X, separate_terms=separate_terms))
    Js_for_vnce_thetas = np.array(Js_for_vnce_thetas)
    nce_optimiser.phi.theta = cur_theta

    return Js_for_vnce_thetas

def get_optimal_J(X, nce_optimiser, true_theta, separate_terms):
    """calculate optimal J (i.e at true theta)"""
    nce_optimiser.phi.theta = deepcopy(true_theta.reshape(-1))
    optimal_J = nce_optimiser.compute_J(X, separate_terms=separate_terms)
    nce_optimiser.phi.theta = cur_theta

    return optimal_J

def plot_rbm_parameters(params, titles, d, m, with_bias=False, figsize=(15, 25)):
    """plot heatmaps of restricted boltzmann machine weights

    :param params: list arrays
        list of (flattened) params of different models
    :param titles: list of strings
        list of titles for each model
    :param d: int
        dimension of visibles
    :parm m: int
        dimension of latents
    """
    if with_bias:
        fig, axs = plt.subplots(len(params), 2, figsize=figsize)
        axs = axs.ravel()
        for i in range(0, len(axs), 2):
            j = int(i/2)
            params[j] = np.array(params[j].reshape(d+1, m+1).T)
            params[j][0, 0] = 0  # ignore scaling param

            axs[i].imshow(params[j], cmap='Blues')
            axs[i].set_title('{} with biases'.format(titles[j]), fontsize=16)
            axs[i].set_xlabel('visible weights', fontsize=16)
            axs[i].set_ylabel('hidden weights', fontsize=16)

            axs[i+1].imshow(params[j][1:, 1:], cmap='Blues')
            axs[i+1].set_title('{} without biases'.format(titles[j]), fontsize=16)
            axs[i+1].set_xlabel('visible weights', fontsize=16)
            axs[i+1].set_ylabel('hidden weights', fontsize=16)
    else:
        fig, axs = plt.subplots(len(params), 1, figsize=figsize)
        axs = axs.ravel()
        for i in range(0, len(axs)):
            params[i] = np.array(params[i].reshape(d+1, m+1).T)
            params[i][0, 0] = 0  # ignore scaling param

            axs[i].imshow(params[i], cmap='Blues')
            axs[i].set_title('{} with biases'.format(titles[i]), fontsize=16)
            axs[i].set_xlabel('visible weights', fontsize=16)
            axs[i].set_ylabel('hidden weights', fontsize=16)
    plt.tight_layout()

    return fig


def plot_log_likelihood_training_curves(training_curves, static_lines):
    """plot log-likelihood training curves

    :param training_curves: list of lists
        each inner list is of the form: [times, log-likelihoods, label]
        where times & log-likelihoods are arrays for plotting
    :param static_lines: list of lists
        each inner list is of the form [log-likelihood, label] where
        log-likelihood is a single value that will plotted as a
        horizontal line
    :param maxiter: int
        index of final time in seconds to plot on x-axis
    :param fig_ax: tuple
        figure, axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for triple in training_curves:
        ax.plot(triple[0], triple[1], label=triple[2])
    for pair in static_lines:
        ax.plot(plt.get(ax, 'xlim'), (pair[0], pair[0]), label=pair[1])
    ax.set_xlabel('time (seconds)', fontsize=16)
    ax.set_ylabel('log likelihood', fontsize=16)
    ax.legend()

    return fig


def average_log_likelihood(model, X):
    """average log-likelihood for unnormalised, latent variable model"""
    likelihoods = model.normalised_and_marginalised_over_z(X)[0]
    return np.mean(np.log(likelihoods))


def rescale_times(times1, times2):
    """make the resolution of the second array of times match the first array"""
    times2_new_ids = []
    # extract the indices for those elements of cd_times closest to those in lnce_times
    for i, theta in enumerate(times1):
        if theta > times2[-1]:
            break
        else:
            times2_new_ids.append(takeClosest(times2, theta))

    times2_new = times2[times2_new_ids]
    # if cd lasted longer, we need to get the remaining time ids
    cd_longer = times2[-1] > times1[-1]
    if cd_longer:
        gaps = [times2_new[i+1] - times2_new[i] for i in range(len(times2_new) - 1)]
        av_gap = np.mean(np.array(gaps))
        remaining_duration = times2[-1] - times2_new[-1]
        remaining_time_ids = [takeClosest(times2, times2_new[-1] + av_gap*i)
                              for i in range(int(remaining_duration / av_gap))]
        times2_new_ids.extend(remaining_time_ids)
        times2_new = np.concatenate((times2_new, times2[remaining_time_ids]))

    return times2_new, times2_new_ids


def takeClosest(myList, myNumber):
    """Get index of element closest to myNumber, but smaller than it

    If myNumber is smaller than all elements, return 0.

    :param myList: List
    :param myNumber: float
    :return closest element : float
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return pos
    else:
        return pos - 1
