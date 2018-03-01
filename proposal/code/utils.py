""" Module containing useful functions for experimenting with
the latent NCE code
"""
import numpy as np
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
