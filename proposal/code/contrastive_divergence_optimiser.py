""" Optimiser that implements contrastive divergence

See the following for an introduction to contrastive divergence:
http://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf
"""

import numpy as np
import time
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import random as rnd
from scipy.optimize import minimize
from utils import validate_shape, average_log_likelihood, takeClosest

DEFAULT_SEED = 1083463236


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyTypeChecker
class CDOptimiser:
    """ Contrastive divergence Optimiser for estimating/learning the
    parameters of an unnormalised model as proposed in:
    http://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf

    For a simple, practical guide that informed this implementation, see:
    https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

    """
    def __init__(self, model, rng=None):
        """ Initialise unnormalised model and noise distribution

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, theta) where:
            - U is (n, d) array of data
        :param rng: random number generator
        """
        self.phi = model
        self.thetas = []  # for storing values of parameters during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def fit(self, X, theta0, num_gibbs_steps, learning_rate, batch_size, num_epochs=50):
        """ Fit the parameters of the model to the data X.

        We fit the data by using the contrastive divergence gradient updates:

        W_ij =  W_ij + learning_rate*( E(v_i*h_j) - E(v_i*h_j) )

        The first expectation is with respect to p_emp(x). p(z| x)
        where p_emp is the empirical distribution of the data.
        The second expectation is with respect to p_model(x, z), which
        we must use gibbs sampling to sample from.

        Following https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf,
        This implementation directly calculates the expected latents Z, given data
        (as opposed to first sampling from P(Z | X)).
        It also directly calculates the expected value of latents on the *final*
        step of gibbs sampling. for every step before that, we sample
        latents.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta0: array
            initial values for model's parameters
        :param num_gibbs_steps: int
            number of gibbs steps to perform during learning, when trying
            to sample from the model's distribution
        :param learning rate: float
            learning rate for gradient ascent
        :param batch_size: int
            size of a mini-batch
        :param num_epochs: int
            number of times we loop through training data
        :return thetas list of array
            values of model parameters (theta) after each gradient step
        """
        # todo: make it easy to access biases and non-bias weights separately? (model-specific hack...)
        # todo: write a get_learning_rate method using decay (this is more important)
        # initialise parameters
        self.phi.theta = deepcopy(theta0)
        self.thetas.append(deepcopy(theta0))
        self.times.append(time.time())
        n = X.shape[0]
        for j in range(num_epochs):
            # shuffle data
            perm = self.rng.permutation(n)
            X = X[perm]
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                Z, X_model, Z_model = self.phi.sample_for_contrastive_divergence(
                    X_batch, num_iter=num_gibbs_steps)

                data_grad = self.phi.grad_log_wrt_params(X_batch, Z)  # (len(theta), 1, n)
                data_grad = np.mean(data_grad, axis=(1, 2))  # (len(theta), )

                model_grad = self.phi.grad_log_wrt_params(X_model, Z_model)
                model_grad = np.mean(model_grad, axis=(1, 2))  # (len(theta), )

                grad = data_grad - model_grad  # (len(theta), )
                self.phi.theta += learning_rate * grad

            self.thetas.append(deepcopy(self.phi.theta))
            self.times.append(time.time())

        self.thetas = np.array(self.thetas)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

        return np.array(self.thetas), np.array(self.times)

    def av_log_like_for_each_iter(self, X, thetas=None):
        """Calculate average log-likelihood at each iteration

        NOTE: this method can only be applied to small models, where
        computing the partition function is not too costly
        """
        theta = deepcopy(self.phi.theta)
        if thetas is None:
            thetas = self.thetas

        av_log_likelihoods = np.zeros(len(thetas))
        for i in np.arange(0, len(thetas)):
            self.phi.theta = deepcopy(thetas[i])
            av_log_likelihoods[i] = average_log_likelihood(self.phi, X)

        self.phi.theta = theta  # reset theta to its original value
        return av_log_likelihoods

    def reduce_optimisation_results(self, time_step_size):
        """reduce to #time_step_size results, evenly spaced on a log scale"""
        log_times = np.exp(np.linspace(-3, np.log(self.times[-1]), num=time_step_size))
        log_time_ids = np.unique(np.array([takeClosest(self.times, t) for t in log_times]))
        reduced_times = deepcopy(self.times[log_time_ids])
        reduced_thetas = deepcopy(self.thetas[log_time_ids])

        return reduced_times, reduced_thetas

    def __repr__(self):
        return "CDOptimiser"
