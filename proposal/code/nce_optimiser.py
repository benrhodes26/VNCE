""" Optimiser that implements noise-contrastive estimation

See the following for an introduction to NCE:
http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf
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
class NCEOptimiser:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model as proposed in http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf

    This class wraps together the NCE objective function, its gradients with respect
    to the parameters of the unnormalised model and a method for performing the
    optimisation: self.fit().
    """
    def __init__(self, model, noise, noise_samples, sample_size, nu=1, eps=1e-15):
        """ Initialise unnormalised model and noise distribution

        :param model: FullyObservedModel
            unnormalised model whose parameters theta we want to optimise.
            see fully_observed_models.py for examples.
        :param noise: Distribution
            noise distribution required for NCE.
            see distribution.py for examples
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.nu = nu
        self.sample_size = sample_size
        self.eps = eps
        self.Y = noise_samples
        self.thetas = []  # for storing values of parameters during optimisation
        self.Js = []  # for storing values of objective function during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation

    def h(self, U):
        """ Compute log(phi(U)) - log(pn(U).
        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :return: array of shape (?)
            ? is either n or n*nu
        """
        return np.log(self.phi(U)) - np.log(self.pn(U))

    def compute_J(self, X, separate_terms=False):
        """Return value of objective at current parameters

        :param X: array (N, d)
            data sample we are training our model on
        :return: float
            Value of objective for current parameters
        """
        Y, nu = self.Y, self.nu

        a0 = 1 + nu*np.exp(-self.h(X))
        a1 = 1 + (1 / nu)*np.exp(self.h(Y))

        first_term = - np.mean(np.log(a0))
        second_term = - nu*np.mean(np.log(a1))
        if separate_terms:
            return [first_term, second_term]

        return first_term + second_term

    def compute_J_grad(self, X):
        """Computes the NCE objective function, J.

        :param X: array (N, d)
            data sample we are training our model on.
        :return grad: array of shape (len(phi.theta), )
        """
        Y, nu = self.Y, self.nu

        gradX = self.phi.grad_log_wrt_params(X)  # (len(theta), n)
        # a0 = 1 / (1 + (1 / nu)*np.exp(self.h(X)))  # (n,)
        a0 = nu*self.pn(X) / (nu*self.pn(X) + self.phi(X))  # (n,)
        # expectation over X
        term_1 = np.mean(gradX*a0, axis=1)  # (len(theta), )

        gradY = self.phi.grad_log_wrt_params(Y)  # (len(theta), nu*n)
        a1 = self.phi(Y) / (nu*self.pn(Y) + self.phi(Y))  # (n)
        # Expectation over Y
        term_2 = - nu * np.mean(gradY*a1, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        assert grad.shape == self.phi.theta_shape, ' ' \
            'Expected grad to be shape {}, got {} instead'.format(self.phi.theta_shape,
                                                                  grad.shape)
        return grad

    def fit(self, X, theta0=np.array([0.5]), disp=True, plot=True, gtol=1e-4, ftol=1e-9, maxiter=100, separate_terms=False):
        """ Fit the parameters of the model to the data X

        optimise the objective function defined in self.compute_J().
        To do this, we use a gradient-based optimiser and define the gradient of J
        with respect to its parameters theta in self.compute_J_grad().

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
            Js: array
                values of objective function during optimisation
            times: array
                ith element is time until ith iteration in seconds
        """
        # initialise parameters
        self.phi.theta = deepcopy(theta0)
        self.thetas.append(deepcopy(theta0))
        self.Js.append(self.compute_J(X, separate_terms=separate_terms))
        self.times.append(time.time())

        # optimise w.r.t to theta
        self.maximize_J1_wrt_theta(X, disp, gtol=gtol, ftol=ftol, maxiter=maxiter, separate_terms=separate_terms)

        if plot:
            self.plot_loss_curve()

        self.thetas = np.array(self.thetas)
        self.Js = np.array(self.Js)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

        return np.array(self.thetas), np.array(self.Js), np.array(self.times)

    def maximize_J1_wrt_theta(self, X, disp, gtol=1e-4, ftol=1e-9, maxiter=100, separate_terms=False):
        """Return theta that maximises J1

        :param X: array (n, 1)
            data
        :param disp: bool
            display optimisation results for each iteration
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        """
        thetas, Js, times = [], [], []

        def callback(_):
            times.append(time.time())
            thetas.append(deepcopy(self.phi.theta))
            Js.append(self.compute_J(X, separate_terms=separate_terms))

        def J1_k_neg(theta):
            self.phi.theta = theta
            return -self.compute_J(X)

        def J1_k_grad_neg(theta):
            self.phi.theta = theta
            return -self.compute_J_grad(X)

        _ = minimize(J1_k_neg, self.phi.theta, method='L-BFGS-B', jac=J1_k_grad_neg,
                     callback=callback, options={'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter, 'disp': disp})

        self.thetas.extend(thetas)
        self.Js.extend(Js)
        self.times.extend(times)

    def plot_loss_curve(self):
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        axs.plot(self.times, self.Js, c='k')

        return fig, axs

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
            av_log_likelihoods[i] = np.mean(np.log(self.phi(X)))

        self.phi.theta = theta  # reset theta to its original value
        return av_log_likelihoods

    def reduce_optimisation_results(self, time_step_size):
        """reduce to #time_step_size results, evenly spaced on a log scale"""
        log_times = np.exp(np.linspace(-3, np.log(self.times[-1]), num=time_step_size))
        log_time_ids = [takeClosest(self.times, t) for t in log_times]
        reduced_times = deepcopy(self.times[log_time_ids])
        reduced_thetas = deepcopy(self.thetas[log_time_ids])

        return reduced_times, reduced_thetas

    def evaluate_J_at_param(self, theta, X, separate_terms=False):
        """
        :param theta: array
            model parameter setting
        :param X: array (n, d)
            data needed to compute objective function at true_theta.
        """
        current_theta = deepcopy(self.phi.theta)
        self.phi.theta = theta
        J = self.compute_J(X, separate_terms)
        self.phi.theta = current_theta
        return J

    def __repr__(self):
        return "NCEOptimiser"
