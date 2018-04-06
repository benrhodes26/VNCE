""" Module containing useful functions for experimenting with
the latent NCE code
"""
import numpy as np
from bisect import bisect_left
from copy import deepcopy
from matplotlib import pyplot as plt

def mean_square_error(estimates, true_value, plot=True):
    true_values = np.ones_like(estimates)*true_value
    error = estimates - true_values
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


def plot_rbm_log_likelihood_training_curves(lnce_ll, cd_ll, lnce_times, cd_times,
                                            init_ll=None, true_ll=None, end=None):
    """plot log-likelihood training curves for latent nce and cd applied to rbm"""
    if end:
        lnce_i = takeClosest(lnce_times, end)
        lnce_times = lnce_times[:lnce_i]
        lnce_ll = lnce_ll[:lnce_i]
        cd_i = takeClosest(cd_times, end)
        cd_times = cd_times[:cd_i]
        cd_ll = cd_ll[:cd_i]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(lnce_times, lnce_ll, label='Latent NCE')
    ax.plot(cd_times, cd_ll, label='CD')
    if init_ll is not None:
            ax.plot(plt.get(ax, 'xlim'), (init_ll, init_ll),
                    'r--', label='initial model')
    if true_ll is not None:
        ax.plot(plt.get(ax, 'xlim'), (true_ll, true_ll),
                'b--', label='True distribution')
    ax.set_xlabel('time (seconds)', fontsize=16)
    ax.set_ylabel('log likelihood', fontsize=16)
    ax.legend()

    return fig

def average_log_likelihood(model, X):
    """average log-likelihood for unnormalised, latent variable model"""
    likelihoods = model.normalised_and_marginalised_over_z(X)[0]
    return np.mean(np.log(likelihoods))


def rescale_cd_times(lnce_times, cd_times):
    """get cd timings on comparable scale (by default they are more freq)"""
    new_cd_time_ids = []
    # extract the indices for those elements of cd_times closest to those in lnce_times
    for i, theta in enumerate(lnce_times):
        if theta > cd_times[-1]:
            break
        else:
            new_cd_time_ids.append(takeClosest(cd_times, theta))

    new_cd_times = cd_times[new_cd_time_ids]
    # if cd lasted longer, we need to get the remaining time ids
    cd_longer = cd_times[-1] > lnce_times[-1]
    if cd_longer:
        gaps = [new_cd_times[i+1] - new_cd_times[i] for i in range(len(new_cd_times) - 1)]
        av_gap = np.mean(np.array(gaps))
        remaining_duration = cd_times[-1] - new_cd_times[-1]
        remaining_time_ids = [takeClosest(cd_times, new_cd_times[-1] + av_gap*i)
                              for i in range(int(remaining_duration / av_gap))]
        new_cd_time_ids.extend(remaining_time_ids)
        new_cd_times = np.concatenate((new_cd_times, cd_times[remaining_time_ids]))

    return new_cd_times, new_cd_time_ids


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
