""" Optimiser that implements Maximum likelihood estimation
"""

import numpy as np
import time
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import random as rnd
from scipy.optimize import minimize
from utils import validate_shape, takeClosest


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyTypeChecker
class MLEOptimiser:
    """ Optimiser for estimating/learning the parameters of a normalised model.

    This class wraps together the MLE objective function, its gradients with respect
    to the parameters of the model and a method for performing the optimisation: self.fit().
    """
    def __init__(self, model):
        """ Initialise unnormalised model and noise distribution

        :param model: FullyObservedModel
            normalised model whose parameters theta we want to optimise.
            see fully_observed_models.py for examples.
        """
        self.phi = model
        self.thetas = []  # for storing values of parameters during optimisation
        self.Ls = []  # for storing values of objective function during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation

    def compute_L(self, X):
        """Return value of average log-likelihood of model

        :param X: array (N, d)
            data sample we are training our model on
        :return: float
            Value of objective for current parameters
        """
        log_like = np.log(self.phi(X))
        return np.mean(log_like)

    def compute_L_grad(self, X):
        """Computes the grad of the average log likelihood

        :param X: array (N, d)
            data sample we are training our model on.
        :return grad: array of shape (len(phi.theta), )
        """
        grad = np.mean(self.phi.grad_log_wrt_params(X), axis=1)

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        assert grad.shape == self.phi.theta_shape, ' ' \
            'Expected grad to be shape {}, got {} instead'.format(self.phi.theta_shape,
                                                                  grad.shape)
        return grad

    def fit(self, X, theta0=np.array([0.5]), disp=True, ftol=1e-9, maxiter=100):
        """ Fit the parameters of the model to the data X

        optimise the objective function defined in self.compute_L().
        To do this, we use a gradient-based optimiser and define the gradient of J
        with respect to its parameters theta in self.compute_L_grad().

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta0: array of shape (1, )
            initial value of theta used for optimisation
        :param disp: bool
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return
            thetas_after_em_step: array
                values of parameters during optimisation
            Ls: array
                values of objective function during optimisation
            times: array
                ith element is time until ith iteration in seconds
        """
        # initialise parameters
        self.phi.theta = deepcopy(theta0)
        self.update_opt_results(X)

        # optimise w.r.t to theta
        self.maximize_L_wrt_theta(X, disp=disp, ftol=ftol, maxiter=maxiter)

        # todo: have an 'unfreeze' method that turns arrays back to lists for continued optimisation
        self.thetas = np.array(self.thetas)
        self.Ls = np.array(self.Ls)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

        return np.array(self.thetas), np.array(self.Ls), np.array(self.times)

    def maximize_L_wrt_theta(self, X, disp=True, ftol=1e-9, maxiter=100):
        """Return theta that maximises L

        :param X: array (n, 1)
            data
        :param disp: bool
            display optimisation results for each iteration
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        """

        def callback(_):
            self.update_opt_results(X)

        def L_k_neg(theta):
            self.phi.theta = theta
            return -self.compute_L(X)

        def L_k_grad_neg(theta):
            self.phi.theta = theta
            return -self.compute_L_grad(X)

        _ = minimize(L_k_neg, self.phi.theta, method='L-BFGS-B', jac=L_k_grad_neg,
                     callback=callback, options={'ftol': ftol, 'maxiter': maxiter, 'disp': disp})

    def plot_loss_curve(self):
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        axs.plot(self.times, self.Ls, c='k')

        return fig, axs

    def update_opt_results(self, X):
        """Save current parameter values, L values and time (in seconds)

        :param X: array (n,)
            data
        """
        self.times.append(time.time())
        self.thetas.append(deepcopy(self.phi.theta))
        self.Ls.append(self.compute_L(X))

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
            av_log_likelihoods[i] = self.compute_L(X)

        self.phi.theta = theta  # reset theta to its original value
        return av_log_likelihoods

    def reduce_optimisation_results(self, time_step_size):
        """reduce to #time_step_size results, evenly spaced on a log scale"""
        log_times = np.exp(np.linspace(-3, np.log(self.times[-1]), num=time_step_size))
        log_time_ids = [takeClosest(self.times, t) for t in log_times]
        reduced_times = deepcopy(self.times[log_time_ids])
        reduced_thetas = deepcopy(self.thetas[log_time_ids])

        return reduced_times, reduced_thetas

    def evaluate_L_at_param(self, theta, X):
        """
        :param theta: array
            model parameter setting
        :param X: array (n, d)
            data needed to compute objective function at true_theta.
        """
        current_theta = deepcopy(self.phi.theta)
        self.phi.theta = theta
        L = self.compute_L(X)
        self.phi.theta = current_theta
        return L

    def __repr__(self):
        return "MLEOptimiser"
