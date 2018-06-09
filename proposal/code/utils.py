""" Module containing useful functions for experimenting with
the latent NCE code. In particular, multiple plotting functions.
"""
import numpy as np

from bisect import bisect_left
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt


def mean_square_error(estimate, true_value, plot=True):

    estimate_float_or_int = isinstance(estimate, float) | isinstance(estimate, int)
    true_val_float_or_int = isinstance(true_value, float) | isinstance(true_value, int)

    if estimate_float_or_int:
        if true_val_float_or_int:
            pass
        elif true_value.ndim == 1:
            estimate = np.array([estimate])
        elif true_value.ndim > 1:
            print('ground truth has dim {}, but estimate is an int or float'.format(true_value.ndim))
            raise TypeError

    elif isinstance(estimate, np.ndarray):
        if estimate.ndim == 1 and true_val_float_or_int:
            true_value = np.array([true_value])
        elif estimate.ndim > 1 and true_val_float_or_int:
            print('estimate has dim {}, but ground truth is an int or float'.format(estimate.ndim))
            raise TypeError
        elif estimate.ndim != true_value.ndim:
            print('estimate has dim {}, but ground truth has dim {}'.format(estimate.ndim, true_value.ndim))
            raise TypeError

    error = estimate - true_value
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


def evaluate_loss_at_param(loss_function, X, Y, theta=None, alpha=None, nz=None):
    """
    :param loss_function: see vnce_optimisers.py for examples
    :param theta: array
        model parameter setting
    :param X: array (n, d)
        data needed to compute objective function at true_theta.
    :param Y: array (n, d)
        noise needed to compute objective function at true_theta.
    """

    if nz:
        current_nz = deepcopy(loss_function.nz)
        loss_function.nz = nz  # increase accuracy of approximation

    current_theta = deepcopy(loss_function.model.theta)
    current_alpha = deepcopy(loss_function.q.alpha)

    if theta is not None:
        loss_function.model.theta = theta
    if alpha is not None:
        loss_function.q.alpha = alpha

    if alpha is not None:
        loss = loss_function(X, Y)
    else:
        loss = loss_function(X, Y, reuse_latent_samples=True)
    loss = np.sum(loss) if loss_function.separate_terms else loss

    # reset parameters to how they were before
    loss_function.model.theta, loss_function.q.alpha = current_theta, current_alpha
    if nz:
        loss_function.nz = current_nz

    return loss


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


def make_nce_minus_vnce_loss_plot(nce_loss_for_vnce_params, vnce_losses, times, e_step_ids):
    """plot J (NCE objective function) minus J1 (lower bound to NCE objective)"""

    fig, axs = plt.subplots(1, 1, figsize=(15, 20))
    axs = [axs]

    ax = axs[0]
    diff = np.sum(nce_loss_for_vnce_params, axis=1) - np.sum(vnce_losses, axis=1)
    ax.plot(times, diff, c='k', label='J - J1')
    diff1 = nce_loss_for_vnce_params[:, 0] - vnce_losses[:, 0]
    ax.plot(times, diff1, c='r', label='term1: J - J1')
    diff2 = nce_loss_for_vnce_params[:, 1] - vnce_losses[:, 1]
    ax.plot(times, diff2, c='b', label='term2: J - J1')

    for ax in axs:
        for time_id in e_step_ids:
            time = times[time_id]
            ax.plot((time, time), ax.get_ylim(), c='0.5')
        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.legend()

    return fig


def get_nce_loss_for_vnce_params(X, nce_optimiser, vnce_optimiser, separate_terms):
    """Evaluate the NCE objective at every parameter setting visited during VNCE optimisation"""
    cur_theta = deepcopy(nce_optimiser.model.theta)

    num_em_steps = len(vnce_optimiser.thetas)
    nce_losses = []

    # nce_optimiser.model.theta = deepcopy(vnce_optimiser.thetas[0][0])
    # append loss twice (to simulate an intial E and an M step)
    # nce_losses.append(nce_optimiser.compute_J(X, separate_terms=separate_terms))
    # nce_losses.append(nce_optimiser.compute_J(X, separate_terms=separate_terms))
    print("about to get nce losses for vnce params")
    for i in range(0, num_em_steps):
        # for every value of theta visited during the M-step, evaluate the nce objective
        for theta in vnce_optimiser.thetas[i]:
            nce_optimiser.model.theta = deepcopy(theta)
            nce_losses.append(nce_optimiser.compute_J(X, separate_terms=separate_terms))

        # during the E-step, the nce objective is constant
        for _ in vnce_optimiser.alphas[i]:
            nce_losses.append(nce_losses[-1])

    nce_losses = np.array(nce_losses)
    nce_optimiser.model.theta = cur_theta

    return nce_losses


def get_optimal_J(X, nce_optimiser, true_theta, separate_terms):
    """calculate optimal J (i.e at true theta)"""
    nce_optimiser.model.theta = deepcopy(true_theta.reshape(-1))
    optimal_J = nce_optimiser.compute_J(X, separate_terms=separate_terms)
    nce_optimiser.model.theta = cur_theta

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


def get_reduced_thetas_and_times(thetas, times, time_step_size):
    """reduce to #time_step_size results, evenly spaced on a log scale"""
    log_times = np.exp(np.linspace(-3, np.log(times[-1]), num=time_step_size))
    log_time_ids = np.unique(np.array([take_closest(times, t) for t in log_times]))
    reduced_times = deepcopy(times[log_time_ids])
    reduced_thetas = deepcopy(thetas[log_time_ids])

    return reduced_thetas, reduced_times


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
            times2_new_ids.append(take_closest(times2, theta))

    times2_new = times2[times2_new_ids]
    # if cd lasted longer, we need to get the remaining time ids
    cd_longer = times2[-1] > times1[-1]
    if cd_longer:
        gaps = [times2_new[i+1] - times2_new[i] for i in range(len(times2_new) - 1)]
        av_gap = np.mean(np.array(gaps))
        remaining_duration = times2[-1] - times2_new[-1]
        remaining_time_ids = [take_closest(times2, times2_new[-1] + av_gap*i)
                              for i in range(int(remaining_duration / av_gap))]
        times2_new_ids.extend(remaining_time_ids)
        times2_new = np.concatenate((times2_new, times2[remaining_time_ids]))

    return times2_new, times2_new_ids


def take_closest(myList, myNumber):
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


def remove_duplicate_legends(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
