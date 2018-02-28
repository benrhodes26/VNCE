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
