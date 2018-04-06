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
from utils import validate_shape

DEFAULT_SEED = 22012018


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyTypeChecker
class NCEOptimiser:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model as proposed in http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf

    This class wraps together the NCE objective function, its gradients with respect
    to the parameters of the unnormalised model and a method for performing the
    optimisation: self.fit().

    """
    def __init__(self, model, noise, sample_size, nu=1, eps=1e-15):
        """ Initialise unnormalised model and noise distribution

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, theta) where:
            - U is (n, d) array of data
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.nu = nu
        self.sample_size = sample_size
        self.eps = eps
        self.Y = self.pn.sample(int(sample_size * nu))  # generate noise

    def h(self, U):
        """ Compute log(phi(U)) - log(pn(U).
        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :return: array of shape (?)
            ? is either n or n*nu
        """
        return np.log(self.phi(U)) - np.log(self.pn(U))

    def compute_J1(self, X):
        """Return MC estimate of Lower bound of NCE objective

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

        return first_term + second_term

    def compute_J1_grad(self, X):
        """Computes J1_grad w.r.t theta using Monte-Carlo

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

    def fit(self, X, theta0=np.array([0.5]), disp=True, plot=True, gtol=1e-4):
        """ Fit the parameters of the model to the data X by optimising
        the objective function defined in self.compute_J1(). To do this,
        we use a gradient-based optimiser and define the gradient of J1
        with respect to its parameters theta in self.compute_J1_grad().

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
            thetas_after_em_step: (?, num_em_steps),
            J1s: list of lists
                each inner list contains the fevals computed for a
                single step (loop) in the EM-type optimisation
            J1_grads: list of lists
                each inner list contains the grad fevals computed for a
                single step (loop) in the EM-type optimisation
        """
        # initialise parameters
        self.phi.theta = deepcopy(theta0)

        # optimise w.r.t to theta
        J1s, J1_grads = self.maximize_J1_wrt_theta(X, disp, gtol)

        if plot:
            self.plot_loss_curve(J1s)

        return np.array(J1s), np.array(J1_grads)

    def maximize_J1_wrt_theta(self, X, disp, gtol=1e-4):
        """Return theta that maximises J1

        :param X: array (n, 1)
            data
        :param disp: bool
            display optimisation results for each iteration
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return J1s: list
            function evaluations during optimisation
                J1_grads: list
            gradient evaluations during optimisation
        """
        J1s, J1_grads = [], []

        def J1_k_neg(theta):
            self.phi.theta = theta
            val = -self.compute_J1(X)
            J1s.append(-val)
            return val

        def J1_k_grad_neg(theta):
            self.phi.theta = theta
            grad_val = -self.compute_J1_grad(X)
            J1_grads.append(-grad_val)
            return grad_val

        _ = minimize(J1_k_neg, self.phi.theta, method='L-BFGS-B', jac=J1_k_grad_neg,
                     options={'gtol': gtol, 'disp': disp})

        return J1s, J1_grads

    def plot_loss_curve(self, J1s):
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        t = np.arange(len(J1s))
        axs.plot(t, J1s, c='k')

        return fig, axs

    def __repr__(self):
        return "NCEOptimiser"
