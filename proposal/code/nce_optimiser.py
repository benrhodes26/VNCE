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
from scipy.optimize import minimize, check_grad
from utils import validate_shape, take_closest


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyTypeChecker
class NCEOptimiser:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model as proposed in http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf

    This class wraps together the NCE objective function, its gradients with respect
    to the parameters of the unnormalised model and a method for performing the
    optimisation: self.fit().
    """
    def __init__(self, model, noise, noise_samples, nu=1, eps=1e-15):
        """ Initialise unnormalised model and noise distribution

        :param model: FullyObservedModel
            normalised model whose parameters theta we want to optimise.
            see fully_observed_models.py for examples.
        :param noise: Distribution
            noise distribution required for NCE.
            see distribution.py for examples
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.model = model
        self.noise = noise
        self.Y = noise_samples
        self.nu = nu
        self.eps = eps
        self.thetas = []  # for storing values of parameters during optimisation
        self.Js = []  # for storing values of objective function during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation

    def h(self, U):
        return self.model(U, log=True) - self.noise(U, log=True)

    def compute_J(self, X, Y=None, separate_terms=False):
        """Return value of objective at current parameters

        :param X: array (N, d)
            data sample we are training our model on
        :return: float
            Value of objective for current parameters
        """
        if Y is None:
            Y = self.Y
        nu = self.nu

        h_x = self.h(X)
        #todo: actually avoid overflow! (i.e use np.where or equivalent)
        a = (h_x >= 0) * np.log(1 + nu * np.exp(-h_x))
        b = (h_x < 0) * (-h_x + np.log(nu + np.exp(h_x)))
        first_term = -np.mean(a + b)

        h_y = self.h(Y)
        #todo: actually avoid overflow! (i.e use np.where or equivalent)
        c = (h_y <= 0) * np.log(1 + (1/nu) * np.exp(h_y))
        d = (h_y > 0) * (h_y + np.log((1/nu) + np.exp(-h_y)))
        second_term = -np.mean(c + d)

        if separate_terms:
            return np.array([first_term, second_term])

        return first_term + second_term

    def compute_J_grad(self, X, Y=None):
        """Computes the NCE objective function, J.

        :param X: array (N, d)
            data sample we are training our model on.
        :return grad: array of shape (len(phi.theta), )
        """
        if Y is None:
            Y = self.Y
        nu = self.nu

        h_x = self.h(X)
        gradX = self.model.grad_log_wrt_params(X)  # (len(theta), n)
        #todo: actually avoid overflow! (i.e use np.where or equivalent)
        a0 = (h_x <= 0) * (1 / (1 + ((1 / nu) * np.exp(h_x))))
        a1 = (h_x > 0) * (np.exp(-h_x) / ((1 / nu) + np.exp(-h_x)))
        a = a0 + a1
        term_1 = np.mean(gradX*a, axis=1)  # (len(theta), )

        gradY = self.model.grad_log_wrt_params(Y)  # (len(theta), nu*n)
        h_y = self.h(Y)
        #todo: actually avoid overflow! (i.e use np.where or equivalent)
        b0 = (h_y > 0) * (1 / (1 + nu * np.exp(-h_y)))
        b1 = (h_y <= 0) * (np.exp(h_y) / (np.exp(h_y) + nu))
        b = b0 + b1
        term_2 = -nu * np.mean(gradY*b, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        assert grad.shape == self.model.theta_shape, ' ' \
            'Expected grad to be shape {}, got {} instead'.format(self.model.theta_shape,
                                                                  grad.shape)
        return grad

    def fit(self,
            X,
            theta0=np.array([0.5]),
            opt_method='L-BFGS-B',
            disp=True, ftol=1e-9,
            maxiter=100,
            learning_rate=0.3,
            batch_size=100,
            num_epochs=1000,
            separate_terms=False):
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
        self.model.theta = deepcopy(theta0)
        # self.update_opt_results(self.compute_J(X, separate_terms=separate_terms))

        # optimise w.r.t to theta
        if opt_method == 'SGD':
            self.maximize_J1_wrt_theta_SGD(X, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs, separate_terms=separate_terms)
        else:
            self.maximize_J1_wrt_theta(X, disp=disp, opt_method=opt_method, ftol=ftol, maxiter=maxiter, separate_terms=separate_terms)

        self.thetas = np.array(self.thetas)
        self.Js = np.array(self.Js)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

        return np.array(self.thetas), np.array(self.Js), np.array(self.times)

    def maximize_J1_wrt_theta(self, X, disp, opt_method='L-BFGS-B', ftol=1e-9, maxiter=100, separate_terms=False):
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
            self.update_opt_results(self.compute_J(X, separate_terms=separate_terms))
            # print("nce finite diff is: {}".format(check_grad(J1_k_neg, J1_k_grad_neg, self.model.theta)))

        def J1_k_neg(theta):
            self.model.theta = theta
            return -self.compute_J(X)

        def J1_k_grad_neg(theta):
            self.model.theta = theta
            return -self.compute_J_grad(X)

        _ = minimize(J1_k_neg, self.model.theta, method=opt_method, jac=J1_k_grad_neg,
                     callback=callback, options={'ftol': ftol, 'maxiter': maxiter, 'disp': disp})

        self.thetas.extend(thetas)
        self.Js.extend(Js)
        self.times.extend(times)

    def maximize_J1_wrt_theta_SGD(self, X, learning_rate, batch_size, num_epochs, separate_terms=False):
            """Maximise objective function using stochastic gradient descent

            :param X: array (n, d)
                data
            :param learning rate: float
                learning rate for gradient ascent
            :param batch_size: int
                size of a mini-batch
            :param num_em_steps int
                number of times we've been through the outer EM loop
            """
            n = X.shape[0]
            for i in range(num_epochs):
                for j in range(0, n, batch_size):
                    # get minibatch
                    X_batch, Y_batch = self.get_minibatch(X, batch_size, j)

                    # compute gradient
                    grad = self.compute_J_grad(X_batch, Y_batch)

                    # update params
                    self.model.theta += learning_rate * grad

                    # save a result at start of learning
                    if i == 0 and j == 0:
                        current_J = self.compute_J(X_batch, Y=Y_batch, separate_terms=separate_terms)
                        self.update_opt_results(current_J)

                # store results
                current_J = self.compute_J(X_batch, Y=Y_batch, separate_terms=separate_terms)
                self.update_opt_results(current_J)

                sum_current_J = np.sum(current_J) if separate_terms else current_J
                print('epoch {}: J = {}'.format(i, sum_current_J))

    def get_minibatch(self, X, batch_size, batch_start):
        """Return a minibatch of data (X), noise (Y) and latents for SGD"""

        batch_slice = slice(batch_start, batch_start + batch_size)
        noise_batch_start = int(batch_start * self.nu)
        noise_batch_slice = slice(noise_batch_start, noise_batch_start + int(batch_size * self.nu))

        X_batch = X[batch_slice]
        Y_batch = self.Y[noise_batch_slice]

        return X_batch, Y_batch

    def begin_epoch(self, X):
        """shuffle data and noise and save current results

        :return X: array
            shuffled_data
        """
        # shuffle data
        perm = self.rng.permutation(len(X))
        noise_perm = self.rng.permutation(len(self.Y))
        X = X[perm]
        self.Y = self.Y[noise_perm]

        return X

    def update_opt_results(self, J):
        self.times.append(time.time())
        self.thetas.append(deepcopy(self.model.theta))
        self.Js.append(deepcopy(J))

    def plot_loss_curve(self):
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        axs.plot(self.times, self.Js, c='k')

        return fig, axs

    def av_log_like_for_each_iter(self, X, thetas=None):
        """Calculate average log-likelihood at each iteration

        NOTE: this method can only be applied to small models, where
        computing the partition function is not too costly
        """
        theta = deepcopy(self.model.theta)
        if thetas is None:
            thetas = self.thetas

        av_log_likelihoods = np.zeros(len(thetas))
        for i in np.arange(0, len(thetas)):
            self.model.theta = deepcopy(thetas[i])
            av_log_likelihoods[i] = np.mean(np.log(self.model(X)))

        self.model.theta = theta  # reset theta to its original value
        return av_log_likelihoods

    def get_reduced_results(self, time_step_size):
        """reduce to #time_step_size results, evenly spaced on a log scale"""
        log_times = np.exp(np.linspace(-3, np.log(self.times[-1]), num=time_step_size))
        log_time_ids = np.unique(np.array([take_closest(self.times, t) for t in log_times]))
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
        current_theta = deepcopy(self.model.theta)
        self.model.theta = theta
        J = self.compute_J(X=X, separate_terms=separate_terms)
        self.model.theta = current_theta
        return J

    def __repr__(self):
        return "NCEOptimiser"
