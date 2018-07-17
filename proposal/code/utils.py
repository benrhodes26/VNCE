""" Module containing useful functions for experimenting with
the latent NCE code. In particular, multiple plotting functions.
"""
import numpy as np
import os
import pickle

from bisect import bisect_left
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt


def mean_square_error(estimate, true_value, plot=False):
    """Calculate MSE(estimate, true_value)"""
    # we want to handle three cases:
    # 1) both of {estimate, true_value} are floats/ints
    # 2) one of {estimate, true_value} is a float/int and the other is an array containing a float/int
    # 3) both {estimate, true_value} are arrays

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
        # plot the distribution over square_error
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


def evaluate_loss_at_param(loss_function, theta=None, alpha=None):
    """
    :param loss_function: see vnce_optimisers.py for examples
    :param theta: array
        model parameter setting
    :param X: array (n, d)
        data needed to compute objective function at true_theta.
    :param Y: array (n, d)
        noise needed to compute objective function at true_theta.
    """
    # alter parameters if necessary
    if theta is not None:
        current_theta = loss_function.get_theta()
        loss_function.set_theta(theta)
    if alpha is not None:
        current_alpha = loss_function.get_alpha()
        loss_function.set_alpha(alpha)

    # compute loss
    loss = loss_function()
    loss = np.sum(loss) if loss_function.separate_terms else loss

    # reset parameters to how they were before
    if theta is not None:
        loss_function.set_theta(current_theta)
    if alpha is not None:
        loss_function.set_alpha(current_alpha)

    return loss


def get_av_log_like(thetas, model, X):
    av_log_like = np.zeros(len(thetas))
    for i in np.arange(0, len(thetas)):
        model.theta = deepcopy(thetas[i])
        av_log_like[i] = average_log_likelihood(model, X)
    return av_log_like


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


def get_reduced_thetas_and_times(thetas, times, time_step_size, max_time=None):
    """reduce to #time_step_size results, evenly spaced on a log scale"""
    max_time = max_time if max_time else times[-1]
    log_times = np.exp(np.linspace(-3, np.log(max_time), num=time_step_size))
    log_time_ids = np.unique(np.array([take_closest(times, t) for t in log_times]))
    reduced_times = deepcopy(times[log_time_ids])
    reduced_thetas = deepcopy(thetas[log_time_ids])

    return reduced_thetas, reduced_times
