""" Optimisers for an unnormalised statistical models with latent variables

These classes wrap together a set of methods for optimising an unnormalised
probabilistic model with latent variables using a novel variant of NCE.
See the following for an introduction to NCE:
(http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)
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


# noinspection PyPep8Naming,PyTypeChecker,PyMethodMayBeStatic
class VNCEOptimiser:
    """ Optimiser for learning the parameters of an unnormalised model with latent variables.

    This class can perform a variational EM-type optimisation procedure using
    fit_using_analytic_q(), which is applied to the objective function J1 (defined
    in compute_J1()).

    Currently, this optimiser requires an tractable expression for the posterior
    over latents p(z|x;theta) and estimates expectations with respect to this
    posterior using monte-carlo estimates. If you have access to analytic expressions
    of these expectations, consider using LatentNCEOptimiserWithAnalyticExpectations.
    """

    def __init__(self, model, noise, variational_dist, noise_samples, sample_size, nu=1, latent_samples_per_datapoint=1,
                 eps=1e-15, rng=None, pos_var_threshold=0.01, ratio_variance_method_1=True):
        """ Initialise unnormalised model and distributions.

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, Z, theta) where:
            - U is (n, d) array of data
            - Z is a (nz, n, m) collection of m-dimensional latent variables
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param variational_dist: Distribution
            variational distribution with parameters alpha. we use it
            to optimise a lower bound on the log likelihood of phi_model.
            Has arguments (Z, U) where:
            - Z is a (nz, n, m) dimensional latent variable
            - U is a (-1, d) dimensional array (either data or noise)
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.q = variational_dist
        self.nu = nu
        self.n = sample_size
        self.nz = latent_samples_per_datapoint
        self.eps = eps
        self.Y = noise_samples
        self.posterior_variance_threshold = pos_var_threshold
        self.ratio_variance_method_1 = ratio_variance_method_1
        self.ZX = None  # array of latent vars used during optimisation
        self.ZY = None  # array of latent vars used during optimisation
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

        # Attributes for storing results of optimisation
        self.thetas = []  # for storing values of parameters during optimisation
        self.J1s = []  # for storing values of objective function during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation
        self.E_step_ids = []  # list to track when we perform E step during optimisation
        self.posterior_ratio_vars = []

    def r(self, U, Z):
        """ compute ratio phi / (pn*q).

        phi refers to the unnormalised model, pn is the noise distribution,
        and q is the posterior distribution over latent variables.

        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :param Z: array of shape (nz, ?, m)
            ? is either n or n*nu
        :return: array of shape (nz, ?)
            ? is either n or n*nu
        """
        if len(Z.shape) == 2:
            Z = Z.reshape((1, ) + Z.shape)
        phi = self.phi(U, Z)
        q = self.q(Z, U)
        # val = phi / (q*self.pn(U) + self.eps)
        val = np.log(phi) - np.log((q * self.pn(U) + self.eps))
        validate_shape(val.shape, (Z.shape[0], Z.shape[1]))

        return val

    def compute_J1(self, X, ZX=None, ZY=None, separate_terms=False):
        """Return MC estimate of Lower bound of NCE objective

        :param X: array (N, d)
            data sample we are training our model on
        :param ZX: array (N, nz, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: array (nz, N*nu)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :return: float
            Value of objective for current parameters
        """
        Y, nu = self.Y, self.nu
        if ZX is None:
            ZX = self.q.sample(self.nz, X)
        if ZY is None:
            ZY = self.q.sample(self.nz, Y)

        r_x = self.r(X, ZX)
        # a = nu / (r_x + self.eps)  # (nz, n)
        a = nu * np.exp(-r_x)  # (nz, n)
        validate_shape(a.shape, (self.nz, self.n))
        first_term = -np.mean(np.log(1 + a))

        r_y = self.r(Y, ZY)
        # b = (1 / nu) * np.mean(r_y, axis=0)  # (n*nu, )
        b = (1 / nu) * np.mean(np.exp(r_y), axis=0)  # (n*nu, )
        validate_shape(b.shape, (self.n * self.nu, ))
        second_term = -nu * np.mean(np.log(1 + b))

        if separate_terms:
            return np.array([first_term, second_term])

        return first_term + second_term

    def compute_J1_grad(self, X, ZX=None, ZY=None, Y=None):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :param Y: array (N*nu, d)
            noise samples
        :param ZX: (N, nz, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: (nz, N*nu)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :return grad: array of shape (len(phi.theta), )
        """
        if Y is None:
            Y = self.Y
        if ZX is None:
            ZX = self.q.sample(self.nz, X)
        if ZY is None:
            ZY = self.q.sample(self.nz, Y)

        gradX = self.phi.grad_log_wrt_params(X, ZX)  # (len(theta), nz, n)
        a = (self._psi_1(X, ZX) - 1)/self._psi_1(X, ZX)  # (nz, n)
        # take double expectation over X and Z
        term_1 = np.mean(gradX * a, axis=(1, 2))  # (len(theta), )

        gradY = self.phi.grad_log_wrt_params(Y, ZY)  # (len(theta), nz, n)
        # r = self.r(Y, ZY)  # (nz, n)
        r = np.exp(self.r(Y, ZY))  # (nz, n)
        # Expectation over ZY
        E_ZY = np.mean(gradY * r, axis=1)  # (len(theta), n)
        one_over_psi2 = 1/self._psi_2(Y, ZY)  # (n, )
        # Expectation over Y
        term_2 = - np.mean(E_ZY * one_over_psi2, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        validate_shape(grad.shape, self.phi.theta_shape)

        return grad

    def _psi_1(self, U, Z):
        """Return array (nz, n)"""
        # return 1 + (self.nu / self.r(U, Z))
        return 1 + (self.nu * np.exp(-self.r(U, Z)))

    def _psi_2(self, U, Z):
        """Return array (n, )"""
        # return 1 + (1/self.nu) * np.mean(self.r(U, Z), axis=0)
        return 1 + (1/self.nu) * np.mean(np.exp(self.r(U, Z)), axis=0)

    def fit(self, X, theta_ids=None, theta0=np.array([0.5]), opt_method='L-BFGS-B', ftol=1e-5,
            maxiter=10, stop_threshold=1e-5, max_num_em_steps=20, learning_rate=0.01, batch_size=10,
            disp=True, plot=True, separate_terms=False, save_dir=None):
        """ Fit the parameters of the model to the data X with an
        EM-type algorithm that switches between optimising the model
        params to produce theta_k and then resetting the variational distribution
        q to be the analytic posterior q(z|x; alpha) =  p(z|x; theta_k). We assume that
        q is parametrised such that setting alpha = theta_k achieves this.

        In general, this posterior is intractable. If this is the case,
        then then you need to specify a parametric family for q, and optimise its parameters.
        You could use LatentNCEOptimiserWithAnalyticExpectations to do this, if you can compute
        certain expectations with respect to q. If you cannot easily compute these expectations,
        then you may be able to use the reparameterisation trick, but this is currently not
        implemented.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta_ids: array
            indices of elements of theta we want to optimise. By default
            this get populated with all indices.
        :param theta0: array of shape (1, )
            initial value of theta used for optimisation
        :param ftol: float
            tolerance parameter passed to L-BFGS-B method of scipy.minimize.
        :param maxiter: int
            maximum number of iterations that L-BFGS-B method of scipy.minimize
            can perform inside each M step of the EM algorithm.
        :param stop_threshold: float
            stop EM-type optimisation when the value of the lower
            bound J1 changes by less than stop_threshold
        :param max_num_em_steps: int
            maximum number of EM steps before termination
        :param batch_size: int
            if opt_method = SGD, then this is the size of mini-batches
        :param disp: bool
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :return
            thetas: array
                intermediate parameters during optimisation
            J1s: array
                function evals computed at each iteration of scipy.minimize
            times: array
                ith element contains time in seconds until ith iteration
        """

        # by default, optimise all elements of theta
        if theta_ids is None:
            theta_ids = np.arange(len(self.phi.theta))

        # initialise parameters
        self.phi.theta, self.q.alpha = deepcopy(theta0), deepcopy(theta0)

        # save time (in seconds), theta and J1(theta) after every iteration/epoch
        self.update_opt_results(X, theta_ids, separate_terms, E_step=True)

        num_em_steps = 0
        prev_J1, current_J1 = -99, -9  # arbitrary distinct numbers
        while np.abs(prev_J1 - current_J1) > stop_threshold \
                and num_em_steps < max_num_em_steps:

            # M-step
            if opt_method == 'SGD':
                self.maximize_J1_wrt_theta_SGD(X, learning_rate=learning_rate, batch_size=batch_size, num_em_steps=num_em_steps, separate_terms=separate_terms)
            else:
                self.maximize_J1_wrt_theta(theta_ids, X, opt_method=opt_method, ftol=ftol, maxiter=maxiter, disp=disp, separate_terms=separate_terms)

            # E-step
            self.q.alpha[theta_ids] = self.phi.theta[theta_ids]
            self.update_opt_results(X, theta_ids, separate_terms, E_step=True)

            prev_J1 = current_J1
            current_J1 = np.sum(deepcopy(self.J1s[-1])) if separate_terms else deepcopy(self.J1s[-1])
            num_em_steps += 1
            if opt_method != 'SGD':
                print('{} EM step: J1 = {}'.format(num_em_steps, current_J1))

            # todo: delete when no longer needed
            if num_em_steps % 5 == 0:
                fig, axs = plt.subplots(2, 1, figsize=(15, 20))
                axs = axs.ravel()
                axs[0].plot(self.times, self.posterior_ratio_vars, c='k', label='V(p(z|y)/q(z|y))')
                axs[1].hist(self.posterior_ratio_vars, bins=int(len(self.posterior_ratio_vars)/10), )
                fig.savefig('{}/J-J1_tracking.pdf'.format(save_dir))

        if plot:
            self.plot_loss_curve()
        self.make_opt_res_arrays()

        return self.thetas, self.J1s, self.times

    def maximize_J1_wrt_theta(self, theta_ids, X, opt_method='L-BFGS-B', ftol=1e-5, maxiter=10, disp=True, separate_terms=False):
        """Maximise objective function using one of the scipy.minimize methods

        :param theta_ids: array
            indices of elements of theta we want to optimise
        :param X: array (n, 1)
            data
        :param ZX: (nz, n, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: (nz, n*nu, m)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :param ftol: float
            tolerance parameter passed to L-BFGS-B method of scipy.minimize.
        :param maxiter: int
            maximum number of iterations that L-BFGS-B method of scipy.minimize
            can perform inside each M step of the EM algorithm.
        :param disp: bool
            display optimisation results for each iteration
        """
        self.ZX = self.q.sample(self.nz, X)
        self.ZY = self.q.sample(self.nz, self.Y)

        def callback(_):
            self.update_opt_results(X, theta_ids, separate_terms)
            if self.ratio_variance_method_1:
                av_ratio_variance = self.get_posterior_ratio_variance_1()
            else:
                av_ratio_variance = self.get_posterior_ratio_variance_2()

            if av_ratio_variance > self.posterior_variance_threshold:
                raise RuntimeError

        def J1_k_neg(theta_subset):
            self.phi.theta[theta_ids] = theta_subset
            return -self.compute_J1(X, ZX=self.ZX, ZY=self.ZY)

        def J1_k_grad_neg(theta_subset):
            self.phi.theta[theta_ids] = theta_subset
            return -self.compute_J1_grad(X, ZX=self.ZX, ZY=self.ZY)[theta_ids]

        try:
            _ = minimize(J1_k_neg, self.phi.theta[theta_ids], method=opt_method, jac=J1_k_grad_neg,
                         callback=callback, options={'ftol': ftol, 'maxiter': maxiter, 'disp': disp})
        except RuntimeError:
            pass

    def get_posterior_ratio_variance_1(self):
        """ E_y(V_z(p(z|u)/q(z|u)) where E=expectation, V=variance."""
        model_posterior = self.phi.p_latent_samples_given_visibles(self.ZY, self.Y)  # (nz, n)
        variational_posterior = self.q(self.ZY, self.Y)  # (nz, n)
        ratio_variance = (model_posterior / variational_posterior).var(axis=0)  # (n, )
        av_ratio_variance = ratio_variance.mean()
        self.posterior_ratio_vars.append(av_ratio_variance)

        return av_ratio_variance

    def get_posterior_ratio_variance_2(self):
        """ V_y(E_z(p(z|u)/q(z|u)) where E=expectation, V=variance."""
        model_posterior = self.phi.p_latent_samples_given_visibles(self.ZY, self.Y)  # (nz, n)
        variational_posterior = self.q(self.ZY, self.Y)  # (nz, n)
        ratio_mean = (model_posterior / variational_posterior).mean(axis=0)  # (n, )
        av_ratio_variance = ratio_mean.var()
        self.posterior_ratio_vars.append(av_ratio_variance)

        return av_ratio_variance

    def maximize_J1_wrt_theta_SGD(self, X, learning_rate, batch_size, num_em_steps, separate_terms=False):
            """Maximise objective function using stochastic gradient descent

            :param X: array (n, 1)
                data
            :param learning rate: float
                learning rate for gradient ascent
            :param batch_size: int
                size of a mini-batch
            :param num_em_steps int
                number of times we've been through the outer EM loop
            """

            # Each epoch, shuffle data and save current results
            if num_em_steps % int(len(X)/batch_size) == 0:
                X = self.begin_epoch(X, batch_size, num_em_steps, separate_terms=separate_terms)

            # get minibatch
            X_batch, Y_batch, ZX_batch, ZY_batch = self.get_minibatch(X, batch_size, num_em_steps)

            # compute gradient
            grad = self.compute_J1_grad(X_batch, ZX=ZX_batch, ZY=ZY_batch, Y=Y_batch)

            # update params
            self.phi.theta += learning_rate * grad

    def get_minibatch(self, X, batch_size, num_em_steps):
        """Return a minibatch of data (X), noise (Y) and latents for SGD"""
        num_batches = int(len(X) / batch_size)
        batch_start = (num_em_steps % num_batches) * batch_size
        batch_slice = slice(batch_start, batch_start + batch_size)
        noise_batch_start = int(batch_start * self.nu)
        noise_batch_slice = slice(noise_batch_start, noise_batch_start + int(batch_size * self.nu))
        X_batch = X[batch_slice]
        Y_batch = self.Y[noise_batch_slice]
        ZX_batch = self.ZX[:, batch_slice, :]
        ZY_batch = self.ZY[:, noise_batch_slice, :]

        return X_batch, Y_batch, ZX_batch, ZY_batch

    def begin_epoch(self, X, batch_size, num_em_steps, separate_terms=False):
        """shuffle data and noise and save current results

        :return X: array
            shuffled_data
        """
        # shuffle data
        perm = self.rng.permutation(len(X))
        noise_perm = self.rng.permutation(len(self.Y))
        X = X[perm]
        self.Y = self.Y[noise_perm]
        self.ZX = self.q.sample(self.nz, X)
        self.ZY = self.q.sample(self.nz, self.Y)

        # store results obtained at end of last epoch
        self.update_opt_results(X, np.arange(len(self.phi.theta)), separate_terms)
        current_J1 = np.sum(deepcopy(self.J1s[-1])) if separate_terms else deepcopy(self.J1s[-1])
        print('epoch {}: J1 = {}'.format(int(num_em_steps / int(len(X) / batch_size)), current_J1))

        return X

    def make_opt_res_arrays(self):
        self.thetas = np.array(self.thetas)
        self.J1s = np.array(self.J1s)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

    def update_opt_results(self, X, theta_inds, separate_terms, E_step=False):
        self.times.append(time.time())
        self.thetas.append(deepcopy(self.phi.theta[theta_inds]))
        if self.ZX is None:
            self.J1s.append(self.compute_J1(X, separate_terms=separate_terms))
        else:
            self.J1s.append(self.compute_J1(X, ZX=self.ZX, ZY=self.ZY, separate_terms=separate_terms))
        if E_step:
            self.E_step_ids.append(deepcopy(len(self.times))-1)
            self.posterior_ratio_vars.append(0)

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

    def plot_loss_curve(self, optimal_J1=None, separate_terms=False):
        """plot of objective function during optimisation

        :param maxiter: int
            index of final time in seconds to plot on x-axis
        :return:
            fig, ax
                plot of objective function during optimisation
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        t = self.times
        J1 = self.J1s

        # plot optimisation curve
        if separate_terms:
            ax.plot(t, J1[:, 0], c='k', label='term 1 of J1')
            ax.plot(t, J1[:, 1], c='k', label='term 2 of J1')
        else:
            ax.plot(t, J1, c='k', label='J1')

        # plot J1(true_theta) which should upper bound our training curve.
        if optimal_J1:
            ax.plot((t[0], t[-1]), (optimal_J1, optimal_J1), 'b--',
                    label='J1 evaluated at true theta')

        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.set_ylabel('J1', fontsize=16)
        ax.legend()

        return fig

    def evaluate_J1_at_param(self, theta, X, nz=None, separate_terms=False):
        """
        :param theta: array
            model parameter setting
        :param X: array (n, d)
            data needed to compute objective function at true_theta.
        """
        if not nz:
            nz = 100
        current_theta = deepcopy(self.phi.theta)
        current_nz = deepcopy(self.nz)
        self.phi.theta, self.q.alpha = theta, theta
        self.nz = nz  # increase accuracy of approximation
        J1 = self.compute_J1(X, separate_terms=separate_terms)
        # reset parameters to how they were before
        self.phi.theta, self.q.alpha = current_theta, current_theta
        self.nz = current_nz

        return J1

    def reduce_optimisation_results(self, time_step_size):
        """reduce to #time_step_size results, evenly spaced on a log scale"""
        log_times = np.exp(np.linspace(-3, np.log(self.times[-1]), num=time_step_size))
        log_time_ids = [takeClosest(self.times, t) for t in log_times]
        reduced_times = deepcopy(self.times[log_time_ids])
        reduced_thetas = deepcopy(self.thetas[log_time_ids])

        return reduced_times, reduced_thetas

    def __repr__(self):
        return "VNCEOptimiser"

# noinspection PyPep8Naming,PyTypeChecker,PyMethodMayBeStatic
class VNCEOptimiserWithoutImportanceSampling:
    """ Optimiser for learning the parameters of an unnormalised model with latent variables.

    This class can perform a variational EM-type optimisation procedure using
    fit_using_analytic_q(), which is applied to the objective function J1 (defined
    in compute_J1()).

    Currently, this optimiser requires an tractable expression for the posterior
    over latents p(z|x;theta) and estimates expectations with respect to this
    posterior using monte-carlo estimates. If you have access to analytic expressions
    of these expectations, consider using VNCEOptimiserWithAnalyticExpectations.
    """

    def __init__(self, model, noise, variational_dist, noise_samples, sample_size, nu=1, latent_samples_per_datapoint=1, eps=1e-15, rng=None):
        """ Initialise unnormalised model and distributions.

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, Z, theta) where:
            - U is (n, d) array of data
            - Z is a (nz, n, m) collection of m-dimensional latent variables
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param variational_dist: Distribution
            variational distribution with parameters alpha. we use it
            to optimise a lower bound on the log likelihood of phi_model.
            Has arguments (Z, U) where:
            - Z is a (nz, n, m) dimensional latent variable
            - U is a (-1, d) dimensional array (either data or noise)
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.q = variational_dist
        self.nu = nu
        self.n = sample_size
        self.nz = latent_samples_per_datapoint
        self.eps = eps
        self.Y = noise_samples
        self.thetas = []  # for storing values of parameters during optimisation
        self.J1s = []  # for storing values of objective function during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation
        self.ZX = None  # array of latent vars used during optimisation
        self.E_step_ids = []  # list to track when we perform E step during optimisation
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def r1(self, X, Z):
        """ compute ratio phi / (pn*q).

        phi refers to the unnormalised model, pn is the noise distribution,
        and q is the posterior distribution over latent variables.

        :param X: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :param Z: array of shape (nz, ?, m)
            ? is either n or n*nu
        :return: array of shape (nz, ?)
            ? is either n or n*nu
        """
        if len(Z.shape) == 2:
            Z = Z.reshape((1, ) + Z.shape)
        phi = self.phi(X, Z)
        q = self.q(Z, X)
        # val = phi / (q*self.pn(U) + self.eps)
        val = np.log(phi) - np.log((q * self.pn(X) + self.eps))
        validate_shape(val.shape, (Z.shape[0], Z.shape[1]))

        return val

    def r2(self, Y):
        """ compute ratio \sum_z (phi)/ (pn).

        phi refers to the unnormalised, latent var model, pn is the noise distribution.
        :param Y: array of shape (n*nu, d)
            noise samples
        :return: array of shape (n*nu, )
        """
        phi = self.phi.marginalised_over_z(Y)  # (n*nu, )
        val = np.log(phi) - np.log((self.pn(Y)))  # (n*nu, )

        return val

    def compute_J1(self, X, ZX=None, separate_terms=False):
        """Return MC estimate of Lower bound of NCE objective

        :param X: array (N, d)
            data sample we are training our model on
        :param ZX: array (N, nz, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: array (nz, N*nu)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :return: float
            Value of objective for current parameters
        """
        Y, nu = self.Y, self.nu
        if ZX is None:
            ZX = self.q.sample(self.nz, X)

        r_x = self.r1(X, ZX)
        a = nu * np.exp(-r_x)  # (nz, n)
        validate_shape(a.shape, (self.nz, self.n))
        first_term = -np.mean(np.log(1 + a))

        r_y = self.r2(Y)
        b = (1 / nu) * np.exp(r_y)  # (n*nu, )
        validate_shape(b.shape, (int(self.n * self.nu), ))
        second_term = -nu * np.mean(np.log(1 + b))

        if separate_terms:
            return np.array([first_term, second_term])

        return first_term + second_term

    def compute_J1_grad(self, X, ZX=None, Y=None):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :param Y: array (N*nu, d)
            noise samples
        :param ZX: (N, nz, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: (nz, N*nu)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :return grad: array of shape (len(phi.theta), )
        """
        if Y is None:
            Y = self.Y
        if ZX is None:
            ZX = self.q.sample(self.nz, X)

        gradX = self.phi.grad_log_wrt_params(X, ZX)  # (len(theta), nz, n)
        a = (self._psi_1(X, ZX) - 1)/self._psi_1(X, ZX)  # (nz, n)
        # take double expectation over X and Z
        term_1 = np.mean(gradX * a, axis=(1, 2))  # (len(theta), )

        gradY = self.phi.grad_log_visible_marginal_wrt_params(Y)  # (len(theta), nz, n)
        phiY = self.phi.marginalised_over_z(Y)
        a1 = phiY / (self.nu*self.pn(Y) + phiY)  # (n)
        # Expectation over Y
        term_2 = - self.nu * np.mean(gradY*a1, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        validate_shape(grad.shape, self.phi.theta_shape)

        return grad

    def _psi_1(self, U, Z):
        """Return array (nz, n)"""
        # return 1 + (self.nu / self.r(U, Z))
        return 1 + (self.nu * np.exp(-self.r1(U, Z)))

    def fit(self, X, theta_ids=None, theta0=np.array([0.5]), opt_method='L-BFGS-B', ftol=1e-5,
            maxiter=10, stop_threshold=1e-5, max_num_em_steps=20, learning_rate=0.01, batch_size=10,
            disp=True, plot=True, separate_terms=False, posterior_variance_threshold=0.01, save_dir=None):
        """ Fit the parameters of the model to the data X with an
        EM-type algorithm that switches between optimising the model
        params to produce theta_k and then resetting the variational distribution
        q to be the analytic posterior q(z|x; alpha) =  p(z|x; theta_k). We assume that
        q is parametrised such that setting alpha = theta_k achieves this.

        In general, this posterior is intractable. If this is the case,
        then then you need to specify a parametric family for q, and optimise its parameters.
        You could use VNCEOptimiserWithAnalyticExpectations to do this, if you can compute
        certain expectations with respect to q. If you cannot easily compute these expectations,
        then you may be able to use the reparameterisation trick, but this is currently not
        implemented.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta_ids: array
            indices of elements of theta we want to optimise. By default
            this get populated with all indices.
        :param theta0: array of shape (1, )
            initial value of theta used for optimisation
        :param ftol: float
            tolerance parameter passed to L-BFGS-B method of scipy.minimize.
        :param maxiter: int
            maximum number of iterations that L-BFGS-B method of scipy.minimize
            can perform inside each M step of the EM algorithm.
        :param stop_threshold: float
            stop EM-type optimisation when the value of the lower
            bound J1 changes by less than stop_threshold
        :param max_num_em_steps: int
            maximum number of EM steps before termination
        :param batch_size: int
            if opt_method = SGD, then this is the size of mini-batches
        :param disp: bool
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :return
            thetas: array
                intermediate parameters during optimisation
            J1s: array
                function evals computed at each iteration of scipy.minimize
            times: array
                ith element contains time in seconds until ith iteration
        """
        # by default, optimise all elements of theta
        if theta_ids is None:
            theta_ids = np.arange(len(self.phi.theta))
        # initialise parameters
        self.phi.theta, self.q.alpha = deepcopy(theta0), deepcopy(theta0)
        # save time (in seconds), theta and J1(theta) after every iteration/epoch
        self.update_opt_results(X, theta_ids, separate_terms, E_step=True)

        num_em_steps = 0
        prev_J1, current_J1 = -99, -9  # arbitrary distinct numbers
        while np.abs(prev_J1 - current_J1) > stop_threshold \
                and num_em_steps < max_num_em_steps:

            # M-step
            if opt_method == 'SGD':
                self.maximize_J1_wrt_theta_SGD(X, learning_rate=learning_rate, batch_size=batch_size, num_em_steps=num_em_steps, separate_terms=separate_terms)
            else:
                self.maximize_J1_wrt_theta(theta_ids, X, opt_method=opt_method, ftol=ftol, maxiter=maxiter, disp=disp,
                                           separate_terms=separate_terms, posterior_variance_threshold=posterior_variance_threshold)

            # E-step
            self.q.alpha[theta_ids] = self.phi.theta[theta_ids]
            self.update_opt_results(X, theta_ids, separate_terms, E_step=True)

            prev_J1 = current_J1
            current_J1 = np.sum(deepcopy(self.J1s[-1])) if separate_terms else deepcopy(self.J1s[-1])
            num_em_steps += 1
            if opt_method != 'SGD':
                print('{} EM step: J1 = {}'.format(num_em_steps, current_J1))

        if plot:
            self.plot_loss_curve()
        self.make_opt_res_arrays()

        return np.array(self.thetas), np.array(self.J1s), np.array(self.times)

    def maximize_J1_wrt_theta(self, theta_ids, X, opt_method='L-BFGS-B', ftol=1e-5, maxiter=10, disp=True,
                              separate_terms=False, posterior_variance_threshold=0.01):
        """Maximise objective function using one of the scipy.minimize methods

        :param theta_ids: array
            indices of elements of theta we want to optimise
        :param X: array (n, 1)
            data
        :param ZX: (nz, n, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: (nz, n*nu, m)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :param ftol: float
            tolerance parameter passed to L-BFGS-B method of scipy.minimize.
        :param maxiter: int
            maximum number of iterations that L-BFGS-B method of scipy.minimize
            can perform inside each M step of the EM algorithm.
        :param disp: bool
            display optimisation results for each iteration
        """
        self.ZX = self.q.sample(self.nz, X)

        def callback(_):
            self.update_opt_results(X, theta_ids, separate_terms)

        def J1_k_neg(theta_subset):
            self.phi.theta[theta_ids] = theta_subset
            return -self.compute_J1(X, ZX=self.ZX)

        def J1_k_grad_neg(theta_subset):
            self.phi.theta[theta_ids] = theta_subset
            return -self.compute_J1_grad(X, ZX=self.ZX)[theta_ids]

        _ = minimize(J1_k_neg, self.phi.theta[theta_ids], method=opt_method, jac=J1_k_grad_neg,
                     callback=callback, options={'ftol': ftol, 'maxiter': maxiter, 'disp': disp})

    def maximize_J1_wrt_theta_SGD(self, X, learning_rate, batch_size, num_em_steps, separate_terms=False):
            """Maximise objective function using stochastic gradient descent

            :param X: array (n, 1)
                data
            :param learning rate: float
                learning rate for gradient ascent
            :param batch_size: int
                size of a mini-batch
            :param num_em_steps int
                number of times we've been through the outer EM loop
            """

            # Each epoch, shuffle data and save current results
            if num_em_steps % int(len(X)/batch_size) == 0:
                X = self.begin_epoch(X, batch_size, num_em_steps, separate_terms=separate_terms)
            # get minibatch
            X_batch, Y_batch, ZX_batch, ZY_batch = self.get_minibatch(X, batch_size, num_em_steps)
            # compute gradient
            grad = self.compute_J1_grad(X_batch, ZX=ZX_batch, Y=Y_batch)
            # update params
            self.phi.theta += learning_rate * grad

    def get_minibatch(self, X, batch_size, num_em_steps):
        """Return a minibatch of data (X), noise (Y) and latents for SGD"""
        num_batches = int(len(X) / batch_size)
        batch_start = (num_em_steps % num_batches) * batch_size
        batch_slice = slice(batch_start, batch_start + batch_size)
        noise_batch_start = int(batch_start * self.nu)
        noise_batch_slice = slice(noise_batch_start, noise_batch_start + int(batch_size * self.nu))
        X_batch = X[batch_slice]
        Y_batch = self.Y[noise_batch_slice]
        ZX_batch = self.ZX[:, batch_slice, :]
        ZY_batch = self.ZY[:, noise_batch_slice, :]

        return X_batch, Y_batch, ZX_batch, ZY_batch

    def begin_epoch(self, X, batch_size, num_em_steps, separate_terms=False):
        """shuffle data and noise and save current results

        :return X: array
            shuffled_data
        """
        # shuffle data
        perm = self.rng.permutation(len(X))
        noise_perm = self.rng.permutation(len(self.Y))
        X = X[perm]
        self.Y = self.Y[noise_perm]
        self.ZX = self.q.sample(self.nz, X)
        self.ZY = self.q.sample(self.nz, self.Y)

        # store results obtained at end of last epoch
        self.update_opt_results(X, np.arange(len(self.phi.theta)), separate_terms)
        current_J1 = np.sum(deepcopy(self.J1s[-1])) if separate_terms else deepcopy(self.J1s[-1])
        print('epoch {}: J1 = {}'.format(int(num_em_steps / int(len(X) / batch_size)), current_J1))

        return X

    def make_opt_res_arrays(self):
        self.thetas = np.array(self.thetas)
        self.J1s = np.array(self.J1s)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

    def update_opt_results(self, X, theta_inds, separate_terms, E_step=False):
        self.times.append(time.time())
        self.thetas.append(deepcopy(self.phi.theta[theta_inds]))
        if self.ZX is None:
            self.J1s.append(self.compute_J1(X, separate_terms=separate_terms))
        else:
            self.J1s.append(self.compute_J1(X, ZX=self.ZX, separate_terms=separate_terms))
        if E_step:
            self.E_step_ids.append(deepcopy(len(self.times))-1)

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

    def plot_loss_curve(self, optimal_J1=None, separate_terms=False):
        """plot of objective function during optimisation

        :param maxiter: int
            index of final time in seconds to plot on x-axis
        :return:
            fig, ax
                plot of objective function during optimisation
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        t = self.times
        J1 = self.J1s

        # plot optimisation curve
        if separate_terms:
            ax.plot(t, J1[:, 0], c='k', label='term 1 of J1')
            ax.plot(t, J1[:, 1], c='k', label='term 2 of J1')
        else:
            ax.plot(t, J1, c='k', label='J1')

        # plot J1(true_theta) which should upper bound our training curve.
        if optimal_J1:
            ax.plot((t[0], t[-1]), (optimal_J1, optimal_J1), 'b--',
                    label='J1 evaluated at true theta')

        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.set_ylabel('J1', fontsize=16)
        ax.legend()

        return fig

    def evaluate_J1_at_param(self, theta, X, nz=None, separate_terms=False):
        """
        :param theta: array
            model parameter setting
        :param X: array (n, d)
            data needed to compute objective function at true_theta.
        """
        if not nz:
            nz = 100
        current_theta = deepcopy(self.phi.theta)
        current_nz = deepcopy(self.nz)
        self.phi.theta, self.q.alpha = theta, theta
        self.nz = nz  # increase accuracy of approximation
        J1 = self.compute_J1(X, separate_terms=separate_terms)
        # reset parameters to how they were before
        self.phi.theta, self.q.alpha = current_theta, current_theta
        self.nz = current_nz

        return J1

    def reduce_optimisation_results(self, time_step_size):
        """reduce to #time_step_size results, evenly spaced on a log scale"""
        log_times = np.exp(np.linspace(-3, np.log(self.times[-1]), num=time_step_size))
        log_time_ids = [takeClosest(self.times, t) for t in log_times]
        reduced_times = deepcopy(self.times[log_time_ids])
        reduced_thetas = deepcopy(self.thetas[log_time_ids])

        return reduced_times, reduced_thetas

    def __repr__(self):
        return "VNCEOptimiserWithoutImportanceSampling"

# noinspection PyPep8Naming,PyTypeChecker,PyMethodMayBeStatic
class VNCEOptimiserWithAnalyticExpectations:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model with latent variables. This Optimiser estimates all expectations
    with respect to the variational distribution using analytic expressions
    provided at initialisation.

    There are 5 expectations required to make this work:

    1) E(r(u, z))
    2) E(log(psi_1(u,z)))
    3) E(grad_theta(log(phi(u,z)) (psi_1(u, z) - 1) / psi_1(u, z))
    4) E(grad_theta(log(phi(u,z)) r(u, z) )
    5) grad_alpha(E(log(1 + nu/r(u, z))))

    Where each expectation is over z ~ q(z | u; alpha).
    r and psi_1 are given by:

    r(u, z) = phi(u, z; theta) / q(z| u; alpha)*pn(u)
    psi_1(u, z) = 1 + (nu/r(u, z))
    """

    def __init__(self, model, noise, noise_samples, variational_dist, sample_size,
                 E1, E2, E3, E4, E5, nu=1, latent_samples_per_datapoint=1,
                 eps=1e-15, rng=None):
        """ Initialise unnormalised model and distributions.

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, Z, theta) where:
            - U is (n, d) array of data
            - Z is a (nz, n, m) collection of m-dimensional latent variables
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param variational_dist: Distribution
            variational distribution with parameters alpha. we use it
            to optimise a lower bound on the log likelihood of phi_model.
            Has arguments (Z, U) where:
            - Z is a (nz, n, m) dimensional latent variable
            - U is a (-1, d) dimensional array (either data or noise)
        :param E1 function
            expectation E(r(u, z)) w.r.t var_dist. Takes args:
            - (U, phi, q, pn, eps)
        :param E2 function
            expectation E(log(psi_1(x,z))) w.r.t var_dist. Takes args:
            - (X, phi, q, pn, nu, eps)
        :param E3 function
            expectation E((psi_1(x, z) - 1) / psi_1(x, z)) w.r.t var_dist. Takes args:
            - (X, phi, q, pn, nu)
        :param E4 function
            expectation E( grad_theta(log(phi(y,z)) r(y, z) ) w.r.t var_dist. Takes args:
            - (Y, phi, q, pn, nu)
        :param E5 function
            gradient_alpha(E(log(1 + nu/r(x, z)))) w.r.t var_dist. Takes args:
            - (X, phi, q, pn, nu, eps)
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.q = variational_dist
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.E4 = E4
        self.E5 = E5
        self.nu = nu
        self.sample_size = sample_size
        self.nz = latent_samples_per_datapoint
        self.eps = eps
        self.Y = noise_samples
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

        # Attributes for storing results of optimisation
        self.thetas = []  # for storing values of parameters during optimisation
        self.alphas = []  # for storing values of variational parameters during optimisation
        self.J1s = []  # for storing values of objective function during optimisation
        self.times = []  # seconds spent to reach each iteration during optimisation
        self.M_step_ids = []  # list to track when we perform M step during optimisation
        self.E_step_ids = []  # list to track when we perform E step during optimisation
        self.posterior_ratio_vars = []

    def compute_J1(self, X, separate_terms=False):
        """Return analytic expression of Lower bound of NCE objective

        :param X: array (N, d)
            data sample we are training our model on
        :return: float
            Value of objective for current parameters
        """
        E2 = self.E2(X, self.phi, self.q, self.pn, self.nu, self.eps)
        first_term = -np.mean(E2)

        E1 = self.E1(self.Y, self.phi, self.q, self.pn, self.eps)
        psi_2 = 1 + (1/self.nu)*E1
        second_term = -self.nu * np.mean(np.log(psi_2))

        if separate_terms:
            return np.array([first_term, second_term])

        return first_term + second_term

    def compute_J1_grad_theta(self, X, Y=None):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :return grad: array of shape (len(phi.theta), )
        """
        if Y is None:
            Y = self.Y

        E3 = self.E3(X, self.phi, self.q, self.pn, self.nu, self.eps)  # (len(theta), n)
        term_1 = np.mean(E3, axis=1)

        E4 = self.E4(Y, self.phi, self.q, self.pn, self.nu, self.eps)  # (len(theta), n)
        a = 1/(self._psi_2(Y) + self.eps)  # (n, )
        term_2 = - np.mean(a * E4, axis=1)

        grad = term_1 + term_2  # (len(theta), )

        # need theta to be an array for optimisation
        if isinstance(grad, float):
            grad = np.array(grad)
        assert grad.shape == self.phi.theta_shape, 'Expected grad ' \
            'to be shape {}, got {} instead'.format(self.phi.theta_shape,
                                                    grad.shape)
        return grad

    def _psi_2(self, U):
        """Return array (n, )"""
        E1 = self.E1(U, self.phi, self.q, self.pn, self.eps)
        return 1 + (1/self.nu) * E1

    def compute_J1_grad_alpha(self, X):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (n, d)
            data sample we are training our model on.
        :return grad: array of shape (len(q.alpha), )
        """
        E5 = self.E5(X, self.phi, self.q, self.pn, self.nu, self.eps)
        grad = np.mean(E5, axis=1)

        assert grad.shape == self.q.alpha_shape, 'Expected grad ' \
            'to be shape {}, got {} instead'.format(self.phi.alpha_shape,
                                                    grad.shape)
        return grad

    def fit(self, X, theta0, alpha0, opt_method='L-BFGS-B', ftol=1e-5, maxiter=10, stop_threshold=1e-5,
            max_num_em_steps=20, learning_rate=0.01, batch_size=10, disp=True, separate_terms=False):
        """ Fit the parameters theta to the data X with a variational EM-type algorithm

        The algorithm alternates between optimising the model
        params to produce theta_k (holding the parameter alpha of q fixed)
        and then optimising the variational distribution q to produce alpha_k
        (holding theta_k fixed).

        To optimise w.r.t alpha, we have to differentiate an expectation
        w.r.t to q(.|alpha). In general, this is a difficult stochastic optimisation
        problem requiring clever techniques to solve (see the reparameterisation trick').

        This class assumes you have already provided the analytic expressions for
        the necessary expectations.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta0: array
            initial value of theta used for optimisation
        :param alpha0: array
            initial value of alpha used for optimisation
        :param disp:
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :param stop_threshold: float
            stop EM-type optimisation when the value of the lower
            bound J1 changes by less than stop_threshold
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return
            thetas_after_em_step: (?, num_em_steps),
            J1s: list of lists
                each inner list contains the fevals computed for a
                single E or M step in the EM-type optimisation
            J1_grads: list of lists
                each inner list contains the grad fevals computed for a
                single E or M step in the EM-type optimisation
        """

        # initialise parameters
        self.phi.theta, self.q.alpha = deepcopy(theta0), deepcopy(alpha0)

        # save time (in seconds), theta and J1(theta) after every iteration/epoch
        self.update_opt_results(X, separate_terms=separate_terms, E_step=False)
        self.E_step_ids.append(deepcopy(len(self.times))-1)
        self.alphas.append(deepcopy(self.q.alpha))

        num_em_steps = 0
        prev_J1, current_J1 = -999, -9999  # arbitrary negative numbers
        while np.abs(prev_J1 - current_J1) > stop_threshold\
                and num_em_steps < max_num_em_steps:

            # M-step: optimise w.r.t theta
            if opt_method == 'SGD':
                self.maximize_J1_wrt_theta_SGD(X, learning_rate=learning_rate, batch_size=batch_size, num_em_steps=num_em_steps, separate_terms=separate_terms)
            else:
                self.maximize_J1_wrt_theta(X, opt_method=opt_method, ftol=ftol, maxiter=maxiter, disp=disp, separate_terms=separate_terms)

            # E-step: optimise w.r.t alpha
            self.maximize_J1_wrt_alpha(X, opt_method=opt_method, ftol=ftol, maxiter=maxiter, disp=disp, separate_terms=separate_terms)

            prev_J1 = current_J1
            current_J1 = np.sum(deepcopy(self.J1s[-1])) if separate_terms else deepcopy(self.J1s[-1])
            num_em_steps += 1
            if opt_method != 'SGD':
                print('{} EM step: J1 = {}'.format(num_em_steps, current_J1))

        self.make_opt_res_arrays()

        return self.thetas, self.alphas, self.J1s, self.times

    def maximize_J1_wrt_theta(self, X, opt_method='L-BFGS-B', ftol=1e-9, maxiter=10, disp=True, separate_terms=False):
        """Return theta that maximises J1

        :param X: array (n, 1)
            data
        :param J1s: list
            list of function evaluations of J1 during optimisation
        :param J1_grads: list
            list of function evaluations of J1_grad during optimisation
        :param disp:
            display output from scipy.minimize
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return new_theta: array
            new theta that maximises the J1 defined by input theta
        """

        def callback(_):
            self.update_opt_results(X, separate_terms=separate_terms, E_step=False)

        def J1_k_neg(theta):
            self.phi.theta = theta
            return -self.compute_J1(X)

        def J1_k_grad_neg(theta):
            self.phi.theta = theta
            return -self.compute_J1_grad_theta(X)

        _ = minimize(J1_k_neg, self.phi.theta, method=opt_method, jac=J1_k_grad_neg, callback=callback,
                     options={'ftol': ftol, 'maxiter': maxiter, 'disp': disp})

    def maximize_J1_wrt_alpha(self, X, opt_method='L-BFGS-B', ftol=1e-9, maxiter=10, disp=True, separate_terms=False):
        """Return alpha that maximises J1

        :param X: array (n, 1)
            data
        :param J1s: list
            list of function evaluations of J1 during optimisation
        :param J1_grads: list
            list of function evaluations of J1_grad during optimisation
        :param disp:
            display output from scipy.minimize
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return new_alpha: array
            new alpha that maximises the J1 defined by input alpha
        """

        def callback(_):
            self.update_opt_results(X, separate_terms=separate_terms, E_step=True)

        def J1_k_neg(alpha):
            self.q.alpha = alpha
            return -self.compute_J1(X)

        def J1_k_grad_neg(alpha):
            self.q.alpha = alpha
            return -self.compute_J1_grad_alpha(X)

        _ = minimize(J1_k_neg, self.q.alpha, method=opt_method, jac=J1_k_grad_neg, callback=callback,
                     options={'ftol': ftol, 'maxiter': maxiter, 'disp': disp})

    def maximize_J1_wrt_theta_SGD(self, X, learning_rate, batch_size, num_em_steps, separate_terms=False):
        """Maximise objective function using stochastic gradient descent

        :param X: array (n, 1)
            data
        :param learning rate: float
            learning rate for gradient ascent
        :param batch_size: int
            size of a mini-batch
        :param num_em_steps int
            number of times we've been through the outer EM loop
        """

        # Each epoch, shuffle data and save current results
        if num_em_steps % int(len(X)/batch_size) == 0:
            X = self.begin_epoch(X, batch_size, num_em_steps, separate_terms=separate_terms)

        # get minibatch
        X_batch, Y_batch = self.get_minibatch(X, batch_size, num_em_steps)

        # compute gradient
        grad = self.compute_J1_grad(X_batch, Y=Y_batch)

        # update params
        self.phi.theta += learning_rate * grad

    def get_minibatch(self, X, batch_size, num_em_steps):
        """Return a minibatch of data (X), noise (Y) and latents for SGD"""
        num_batches = int(len(X) / batch_size)
        batch_start = (num_em_steps % num_batches) * batch_size
        batch_slice = slice(batch_start, batch_start + batch_size)
        noise_batch_start = int(batch_start * self.nu)
        noise_batch_slice = slice(noise_batch_start, noise_batch_start + int(batch_size * self.nu))
        X_batch = X[batch_slice]
        Y_batch = self.Y[noise_batch_slice]

        return X_batch, Y_batch

    def begin_epoch(self, X, batch_size, num_em_steps, separate_terms=False):
        """shuffle data and noise and save current results

        :return X: array
            shuffled_data
        """
        # shuffle data
        perm = self.rng.permutation(len(X))
        noise_perm = self.rng.permutation(len(self.Y))
        X = X[perm]
        self.Y = self.Y[noise_perm]

        # store results obtained at end of last epoch
        self.update_opt_results(X, separate_terms=separate_terms, E_step=False)
        current_J1 = np.sum(deepcopy(self.J1s[-1])) if separate_terms else deepcopy(self.J1s[-1])
        print('epoch {}: J1 = {}'.format(int(num_em_steps / int(len(X) / batch_size)), current_J1))

        return X

    def update_opt_results(self, X, separate_terms, E_step):
        """Save current parameter values, J1 values and time (in seconds)

        :param X: array (n,)
            data
        :param separate_terms: boolean
            whether or not to save data & noise terms of J1 separately
        :param E_step: boolean
            if True, update alpha params, else we are in the M-step, so
            update the theta params
        """
        self.times.append(time.time())
        self.J1s.append(self.compute_J1(X, separate_terms=separate_terms))
        if E_step:
            self.E_step_ids.append(deepcopy(len(self.times))-1)
            self.alphas.append(deepcopy(self.q.alpha))
        else:
            self.M_step_ids.append(deepcopy(len(self.times))-1)
            self.thetas.append(deepcopy(self.phi.theta))

    def make_opt_res_arrays(self):
        self.thetas = np.array(self.thetas)
        self.alphas = np.array(self.alphas)
        self.J1s = np.array(self.J1s)
        self.times = np.array(self.times)
        self.times -= self.times[0]  # count seconds from 0

    def plot_loss_curve(self, optimal_J1=None, separate_terms=False):
        """plot of objective function during optimisation

        :param maxiter: int
            index of final time in seconds to plot on x-axis
        :return:
            fig, ax
                plot of objective function during optimisation
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        t = self.times
        J1 = self.J1s

        # plot optimisation curve
        if separate_terms:
            ax.plot(t, J1[:, 0], c='k', label='term 1 of J1')
            ax.plot(t, J1[:, 1], c='k', label='term 2 of J1')
        else:
            ax.plot(t, J1, c='k', label='J1')

        # plot J1(true_theta) which should upper bound our training curve.
        if optimal_J1:
            ax.plot((t[0], t[-1]), (optimal_J1, optimal_J1), 'b--',
                    label='J1 evaluated at true theta')

        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.set_ylabel('J1', fontsize=16)
        ax.legend()

        return fig, ax

    def __repr__(self):
        return "VNCEOptimiserWithAnalyticExpectations"


