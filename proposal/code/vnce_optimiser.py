""" This module contains classes for optimising an unnormalised probabilistic model with latent variables
using a novel variant of NCE. See the following for an introduction to NCE: (http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

There are three types of classes:
- VemOptimiser: class for performing variational EM algorithm with any method for the E step, M step and any VNCE cost function
- SgdEmStep & ScipyMinimiseEmStep & exact_e_step: callables that can be passed to VemOptimiser for the E or M steps
- MonteCarloVnceLoss, MonteCarloVnceLossWithoutImportanceSampling & VnceLossWithAnalyticExpectations: variants of the VNCE loss
function (the underlying equation is the same, the only difference is which expectations are approximated with monte-carlo).
"""

import numpy as np
import time
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import random as rnd
from scipy.optimize import minimize
from utils import validate_shape, average_log_likelihood, take_closest, remove_duplicate_legends

DEFAULT_SEED = 1083463236

# noinspection PyPep8Naming,PyTypeChecker,PyMethodMayBeStatic
class VemOptimiser:
    """Variational Expectation-Maximisation Noise-contrastive Estimation optimiser.

    This class can perform a variational EM-type optimisation procedure using fit(), which is applied to the loss function
    provided at initialisation. This loss function needs to be a callable: see MonteCarloVnceLoss for an example.

    This class does not specify how to perform the inner loop of optimisation in the E/M steps. The E/M steps need to be
    provided at initialisation as objects with particular methods: see SgdEmStep or ScipyMinimiseEmStep for examples.
    """

    def __init__(self, m_step, e_step):
        """
        :param loss_function: object
            see e.g. MonteCarloVnceLoss
        :param e_step: object
            see e.g. SgdEmStep
        :param m_step: object
            see e.g SgdEmStep
        """
        self.m_step = m_step
        self.e_step = e_step

        # results during optimisation. Each is a list of lists. The inner lists pertain to a specific E or M step.
        self.thetas = []  # model parameters
        self.alphas = []  # variational parameters
        self.losses = []
        self.times = []

    def fit(self, loss_function, theta0, alpha0, stop_threshold=1e-6, max_num_em_steps=100):
        """Optimise the loss function initialised on this class to fit the data X

        :param X: array (n, d)
            data
        :param Y: array (n*nu, d)
            noise used in (V)NCE
        :param theta0: array
            starting value of the model's parameters
        :param alpha0: array
            starting value of the variational parameters
        :param stop_threshold:
            if loss decreases by less than this threshold after one EM loop, terminate
        :param max_num_em_steps:
            terminate optimisation after this many EM iterations
        """

        self.init_parameters(loss_function, alpha0, theta0)

        num_em_steps = 0
        prev_loss, current_loss = -99, -9  # arbitrary distinct numbers
        while np.abs(prev_loss - current_loss) > stop_threshold and num_em_steps < max_num_em_steps:

            m_results = self.m_step(loss_function)
            if m_results:
                self.update_results(*m_results, m_step=True)

            e_results = self.e_step(loss_function)
            if e_results and m_results:
                self.update_results(*e_results, e_step=True)

            prev_loss = current_loss
            current_loss = deepcopy(loss_function.current_loss)
            current_loss = np.sum(current_loss) if isinstance(current_loss, np.ndarray) else current_loss
            num_em_steps += 1

    def init_parameters(self, loss_function, alpha0, theta0):
        """Initialise parameters and update optimisation results accordingly"""
        loss_function.model.theta = deepcopy(theta0)
        loss_function.q.alpha = deepcopy(alpha0)

        #todo: how to change this to work with adaptive noise?
        # initial_loss = loss_function(X, Y)
        # self.update_results([theta0], [initial_loss], [time.time()], m_step=True)
        # self.update_results([alpha0], [initial_loss], [time.time()], e_step=True)

    def update_results(self, params, losses, times, m_step=False, e_step=False):
        """Update optimisation results during learning"""
        if m_step:
            self.thetas.append(params)
            self.losses.append(losses)
            self.times.append(times)
        if e_step:
            self.alphas.append(params)
            self.losses.append(losses)
            self.times.append(times)

    def get_flattened_result_arrays(self):
        thetas = np.array([theta for sublist in self.thetas for theta in sublist])
        alphas = np.array([alpha for sublist in self.alphas for alpha in sublist])
        losses = np.array([loss for sublist in self.losses for loss in sublist])
        times = np.array([time for sublist in self.times for time in sublist])

        times -= times[0]
        return thetas, alphas, losses, times

    def av_log_like_for_each_iter(self, X, loss_function, thetas=None):
        """Calculate average log-likelihood at each iteration

        NOTE: this method can only be applied to small models, where
        computing the partition function is not too costly
        """

        theta = deepcopy(self.loss_function.model.theta)
        if thetas is None:
            thetas = self.thetas

        av_log_likelihoods = np.zeros(len(thetas))
        for i in np.arange(0, len(thetas)):
            loss_function.model.theta = deepcopy(thetas[i])
            av_log_likelihoods[i] = average_log_likelihood(self.loss_function.model, X)

        loss_function.model.theta = theta  # reset theta to its original value
        return av_log_likelihoods

    def plot_loss_curve(self, optimal_loss=None, separate_terms=False):
        """plot of objective function during optimisation"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        _, _, losses, times = self.get_flattened_result_arrays()
        _, _, m_start_ids, e_start_ids = self.get_m_and_e_step_ids()

        # plot optimisation curve
        if separate_terms:
            ax.plot(times, losses[:, 0], c='k', label='term 1 of J1')
            ax.plot(times, losses[:, 1], c='k', label='term 2 of J1')
        else:
            ax.plot(times, losses, c='k', label='J1')

        # plot J1(true_theta) which should upper bound our training curve.
        if optimal_loss:
            ax.plot((times[0], times[-1]), (optimal_loss, optimal_loss), 'g--',
                    label='J1 evaluated at true theta')

        for id in m_start_ids:
            ax.plot((times[id], times[id]), (losses.min(), losses.max()), c='b', alpha=0.5, label='start of m-step')

        for id in e_start_ids:
            ax.plot((times[id], times[id]), (losses.min(), losses.max()), c='r', alpha=0.5, label='start of e-step')

        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.set_ylabel('J1', fontsize=16)

        # remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right')

        return fig

    def get_m_and_e_step_ids(self, e_step_results_exist=True):
        """Return the indices of the M steps and E steps in the flattened array of times/losses.
        Also return just the *first* index of each E and M step (this is useful for plotting).
        """
        m_ids, e_ids = [], []
        m_start_ids, e_start_ids = [], []
        id = 0
        m_step = True
        for m_or_e_step in self.times:
            for i in range(len(m_or_e_step)):
                if m_step:
                    m_ids.append(id)
                    if i == 0:
                        m_start_ids.append(id)
                else:
                    e_ids.append(id)
                    if i == 0:
                        e_start_ids.append(id)
                id += 1
            if e_step_results_exist:
                m_step = not m_step

        return m_ids, e_ids, m_start_ids, e_start_ids

    def __repr__(self):
        return "VemOptimiser"


class SgdEmStep:
    """Class that can be used in the E or M step of the VemOptimiser.

    If used during the M step, it optimises the VNCE objective w.r.t the model parameters theta.
    If used during the E step, it optimises the VNCE objective w.r.t the variational parameters alpha.
    In either case, it uses stochastic gradient descent with adjustable mini-batch size and learning rate.
    """
    def __init__(self, do_m_step, learning_rate, batch_size, num_batches_per_em_step, num_data_points, noise_to_data_ratio, rng):
        """
        :param do_m_step: boolean
            if True, optimise loss function w.r.t model params theta. Else w.r.t variational params alpha.
        :param learning_rate: float
        :param batch_size: int
        :param num_batches_per_em_step: int
        :param num_data_points: int
        :param noise_to_data_ratio: int
        :param rng: np.RandomState
        """
        self.do_m_step = do_m_step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_batches_per_em_step = num_batches_per_em_step
        self.batches_per_epoch = int(num_data_points / batch_size)
        self.current_batch_id = 0
        self.current_epoch = 0
        self.nu = noise_to_data_ratio
        self.rng = rng

    def __call__(self, loss_function):
        """Execute optimisation of loss function using stochastic grad descent
        :param X: array (n, d)
            data
        :param Y: array (n*nu, d)
            noise in (V)NCE
        :param loss_function:
            see e.g. MonteCarloVnceLoss
        """
        for _ in range(self.num_batches_per_em_step):
            # todo: work out how to tell loss function what batch to use
            if self.do_m_step:
                grad = loss_function.grad_wrt_theta()
                loss_function.model.theta += self.learning_rate * grad
            else:
                grad = loss_function.grad_wrt_alpha()
                # todo: check that += still forces us to resample (should do...)
                loss_function.q.alpha += self.learning_rate * grad

        # calculate loss, which is stored on the loss function and used in the stopping criterion of the em algorithm
        _ = loss_function()

        # If we're about to start a new epoch, shuffle the data and print status
        start_new_epoch = (self.current_batch_id + 1) % self.batches_per_epoch == 0
        if start_new_epoch or self.current_epoch == 0:
            current_loss = self.new_epoch(X, Y, X_batch, Y_batch, loss_function)
            return [deepcopy(loss_function.model.theta)], [current_loss], [time.time()]

    def __repr__(self):
        return "SgdEmStep"


class ScipyMinimiseEmStep:
    """Class that can be used in the E or M step of the VemOptimiser.

    If used during the M step, it optimises the VNCE objective w.r.t the model parameters theta.
    If used during the E step, it optimises the VNCE objective w.r.t the variational parameters alpha.
    In either case, it uses stochastic gradient descent with adjustable mini-batch size and learning rate.
    """
    def __init__(self, do_m_step, optimisation_method='L-BFGS-B', max_iter=100):
        """
        :param do_m_step: boolean
            if True, optimise loss function w.r.t model params theta. Else w.r.t variational params alpha.
        :param optimisation_method: string
            the 'method' argument to scipy.minimize
        :param max_iter: int
            the max_iter argument to scipy.minimize
        """
        self.do_m_step = do_m_step
        self.opt_method = optimisation_method
        self.max_iter = max_iter
        self.current_loss = None
        self.params = []
        self.losses = []
        self.times = []

    def __call__(self, loss_function):
        """Execute optimisation of loss function scipy.minimise
        :param X: array (n, d)
            data
        :param Y: array (n*nu, d)
            noise in (V)NCE
        :param loss_function:
            see e.g. MonteCarloVnceLoss
        """
        def callback(param):
            self.update_results(deepcopy(param))

        def loss_neg(param_k):
            if self.do_m_step:
                loss_function.model.theta = deepcopy(param_k)
                self.current_loss = loss_function()
            else:
                loss_function.q.alpha = deepcopy(param_k)
                self.current_loss = loss_function()
            return deepcopy(-np.sum(self.current_loss))

        def loss_grad_neg(param_k):
            if self.do_m_step:
                loss_function.model.theta = deepcopy(param_k)
                grad = -loss_function.grad_wrt_theta()
            else:
                loss_function.q.alpha = deepcopy(param_k)
                grad = -loss_function.grad_wrt_alpha()
            return grad

        if self.do_m_step:
            start_val = loss_function.model.theta
        else:
            start_val = loss_function.q.alpha

        _ = minimize(loss_neg, start_val, method=self.opt_method, jac=loss_grad_neg,
                     callback=callback, options={'maxiter': self.max_iter, 'disp': True})

        if self.do_m_step:
            self.update_results(deepcopy(loss_function.model.theta))
        else:
            self.update_results(deepcopy(loss_function.q.alpha))

        params, losses, times = deepcopy(self.params), deepcopy(self.losses), deepcopy(self.times)
        self.reset_results()

        return params, losses, times

    def update_results(self, theta):
        """Save current parameter, loss and time for plotting after optimisation"""
        self.params.append(deepcopy(theta))
        self.losses.append(deepcopy(self.current_loss))
        self.times.append(time.time())

    def reset_results(self):
        self.params = []
        self.losses = []
        self.times = []

    def __repr__(self):
        return "ScipyMinimiseEmStep"


class ExactEStep:

    def __init__(self, calculate_loss=True):
        """
         :param calculate_loss: boolean
            Calculating the loss in the E-step isn't necessary. It can be useful for tracking performance, but it can
            also be expensive (since we have to re-sample from the new q)."""
        self.calculate_loss = calculate_loss

    def __call__(self, loss_function):
        new_alpha = deepcopy(loss_function.model.theta)
        loss_function.q.alpha = new_alpha
        if self.calculate_loss:
            loss = loss_function()
            return [new_alpha], [loss], [time.time()]

    def __repr__(self):
        return "ExactEStep"


class AdaptiveEStep:

    def __init__(self, calculate_loss=True):
        """
         :param calculate_loss: boolean
            Calculating the loss in the E-step isn't necessary. It can be useful for tracking performance, but it can
            also be expensive (since we have to re-sample from the new q)."""
        self.calculate_loss = calculate_loss

    def __call__(self, loss_function):
        new_alpha = deepcopy(loss_function.model.theta)
        loss_function.noise.alpha = new_alpha
        if self.calculate_loss:
            loss = loss_function()
            return [new_alpha], [loss], [time.time()]

    def __repr__(self):
        return "AdaptiveEStep"


class MonteCarloVnceLoss:
    """VNCE loss function and gradients where all expectations are approximated using Monte Carlo

    Currently, grad_wrt_alpha(), where alpha is the variational parameter, is not implemented.
    This is fine when we have access to the exact posterior p(z|x; theta) and so do not need variational parameters.
    """

    def __init__(self,
                 model,
                 data,
                 noise,
                 noise_samples,
                 variational_noise,
                 noise_to_data_ratio,
                 num_latent_per_data,
                 use_minibatches=False,
                 batch_size=None,
                 separate_terms=False,
                 eps=1e-7):
        """
        :param model: see latent_variable_model.py for examples
        :param noise: see distribution.py for examples
        :param variational_noise: see distribution.py for examples
        :param noise_to_data_ratio: int
        :param separate_terms: boolean
            the VNCE loss function is made of two terms: an expectation w.r.t the data and an expectation w.r.t the noise.
            If separate_terms=True, then the loss function outputs a each term separately in an array. This is useful for plotting / debugging.
        """
        self.model = model
        self.data = data
        self.noise = noise
        self.noise_samples = noise_samples
        self.variational_noise = variational_noise
        self.nu = noise_to_data_ratio
        self.nz = num_latent_per_data
        self.separate_terms = separate_terms
        self.use_minibatches = use_minibatches
        self.eps = eps

        if use_minibatches:
            self.X = None
            self.Y = None
        else:
            self.X = data
            self.Y = noise_samples
        self.batch_size = batch_size
        self.batches_per_epoch = int(len(data) / batch_size)
        self.current_batch_id = 0
        self.current_epoch = 0

        self.ZX = None
        self.ZY = None
        self.resample_from_variational_noise = True
        self.current_loss = None

    def set_theta(self, new_theta):
        self.model.theta = deepcopy(new_theta)

    def set_alpha(self, new_alpha):
        self.variational_noise.alpha = new_alpha
        self.resample_from_variational_noise = True

    def __call__(self, next_minibatch=False):
        """Return Monte Carlo estimate of Lower bound of NCE objective

       :param X: array (N, d)
           data sample we are training our model on
       :param Y: array (nz, N*nu)
           noise samples
       :param reuse_latent_samples: boolean
           If true, reuse the samples ZX and ZY saved on this class as attributes. This is useful
           during variational EM, since we want to reuse the same samples for every update in the M step.
       :return: float
           Value of objective for current parameters
       """
        if next_minibatch:
            self.next_minibatch()
        self.resample_noise_if_necessary()

        nu = self.nu
        h_x = self.h(self.X, self.ZX)
        a = (h_x > 0) * np.log(1 + nu * np.exp(-h_x))
        b = (h_x < 0) * (-h_x + np.log(nu + np.exp(h_x)))
        first_term = -np.mean(a + b)

        h_y = self.h(self.Y, self.ZY)
        expectation = np.mean(np.exp(h_y), axis=0)
        c = (1 / nu) * expectation  # (n*nu, )
        second_term = -nu * np.mean(np.log(1 + c))

        validate_shape(a.shape, (self.nz, len(self.X)))
        validate_shape(c.shape, (len(self.Y), ))

        if self.separate_terms:
            val = np.array([first_term, second_term])
        else:
            val = first_term + second_term
        self.current_loss = val

        return val

    def h(self, U, Z):
        """Compute the ratio: model / (noise*q).

        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :param Z: array of shape (nz, ?, m)
            ? is either n or n*nu
        :return: array of shape (nz, ?)
            ? is either n or n*nu
        """

        if len(Z.shape) == 2:
            Z = Z.reshape((1, ) + Z.shape)

        phi = self.model(U, Z)
        q = self.variational_noise(Z, U)
        val = np.log(phi) - np.log((q * self.noise(U) + self.eps))
        validate_shape(val.shape, (Z.shape[0], Z.shape[1]))

        return val

    def grad_wrt_theta(self, next_minibatch=False):
        """Computes grad of loss w.r.t theta using Monte-Carlo

        :param X: array (n, d)
            data sample we are training our model on.
        :param Y: array (n*nu, d)
            noise samples
        :return grad: array of shape (len(model.theta), )
        """
        if next_minibatch:
            self.next_minibatch()
        self.resample_noise_if_necessary()

        joint_noise = self.nu * self.noise(self.X) * self.variational_noise(self.ZX, self.X)
        a = joint_noise / (joint_noise + self.model(self.X, self.ZX))  # (nz, n)

        gradX = self.model.grad_log_wrt_params(self.X, self.ZX)  # (len(theta), nz, n)
        term_1 = np.mean(gradX * a, axis=(1, 2))  # (len(theta), )

        gradY = self.model.grad_log_wrt_params(self.Y, self.ZY)  # (len(theta), nz, n)
        r = np.exp(self.h(self.Y, self.ZY))  # (nz, n)

        # Expectation over ZY
        E_ZY = np.mean(gradY * r, axis=1)  # (len(theta), n)
        one_over_psi = 1/self._psi(self.Y, self.ZY)  # (n, )

        # Expectation over Y
        term_2 = - np.mean(E_ZY * one_over_psi, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        validate_shape(grad.shape, self.model.theta_shape)

        return grad

    def _psi(self, U, Z):
        """Return array (n, )"""
        return 1 + ((1/self.nu) * np.mean(np.exp(self.h(U, Z)), axis=0))

    def grad_wrt_alpha(self):
        raise NotImplementedError

    def next_minibatch(self):
        batch_start = self.current_batch_id * self.batch_size
        batch_slice = slice(batch_start, batch_start + self.batch_size)

        noise_batch_start = int(batch_start * self.nu)
        noise_batch_slice = slice(noise_batch_start, noise_batch_start + int(self.batch_size * self.nu))

        self.X = self.data[batch_slice]
        self.Y = self.noise_samples[noise_batch_slice]

        self.current_batch_id += 1
        if self.current_batch_id % self.batches_per_epoch == 0:
            self.current_batch_id = 0

    def new_epoch(self):
        """Shuffle data X and noise Y and print current loss"""
        self.rng.shuffle(self.data)
        self.rng.shuffle(self.noise_samples)

        current_loss = loss_function(X_batch, Y_batch)
        print('epoch {}: J1 = {}'.format(self.current_epoch, current_loss))
        self.current_epoch += 1

        return current_loss

    def resample_noise_if_necessary(self):
        if (self.ZX is None) or (self.ZY is None) or self.resample_from_variational_noise:
            self.ZX = self.variational_noise.sample(self.nz, self.X)
            self.ZY = self.variational_noise.sample(self.nz, self.Y)
            self.resample_from_variational_noise = False

    def __repr__(self):
        return "MonteCarloVnceLoss"


class MonteCarloVnceLossWithoutImportanceSampling:
    """VNCE loss function identical to MonteCarloVnceLoss except we do not use importance sampling in second term

    This loss function requires us to be able to analytically integrate out the latent variables in our model in order
    to avoid using importance sampling. This is NOT a realistic assumption for many models, but when it is feasible, it
    allows us to see how much `damage' importance sampling is doing.

    Currently, grad_wrt_alpha(), where alpha is the variational parameter, is not implemented.
    This is fine when we have access to the exact posterior p(z|x; theta) and so do not need variational parameters.
    """
    def __init__(self, model, noise, noise_samples, variational_q, noise_to_data_ratio, num_latent_per_data, separate_terms=False, eps=1e-7):
        self.model = model
        self.noise = noise
        self.Y = noise_samples
        self.q = variational_q
        self.nu = noise_to_data_ratio
        self.nz = num_latent_per_data
        self.separate_terms = separate_terms
        self.eps = eps
        self.current_loss = None
        self.ZX = None
        self.ZY = None

    def __call__(self, X, Y=None, reuse_latent_samples=False):
        """Return Monte Carlo estimate of Lower bound of NCE objective

       :param X: array (N, d)
           data sample we are training our model on
       :param Y: array (nz, N*nu)
           noise samples
       :param reuse_latent_samples: boolean
           If true, reuse the samples ZX and ZY saved on this class as attributes. This is useful
           during variational EM, since we want to reuse the same samples for every update in the M step.
       :return: float
           Value of objective for current parameters
       """
        if Y is None:
            Y = self.Y
        if (self.ZX is None) or (not reuse_latent_samples):
            self.ZX = self.q.sample(self.nz, X)

        nu = self.nu
        h_x = self.h1(X, self.ZX)
        a = (h_x > 0) * np.log(1 + nu * np.exp(-h_x))
        b = (h_x < 0) * (-h_x + np.log(nu + np.exp(h_x)))
        first_term = -np.mean(a + b)

        h_y = self.h2(Y)
        c = (h_y < 0) * np.log(1 + (1/nu) * np.exp(h_y))
        d = (h_y > 0) * (h_y + np.log((1/nu) + np.exp(-h_y)))
        second_term = -np.mean(c + d)

        validate_shape(a.shape, (self.nz, len(X)))
        validate_shape(c.shape, (len(Y), ))

        if self.separate_terms:
            val = np.array([first_term, second_term])
        else:
            val = first_term + second_term

        self.current_loss = val

        return val

    def h1(self, X, Z):
        """ compute ratio model / (noise*q).

        :param X: array of shape (n, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :param Z: array of shape (nz, n, m)
            ? is either n or n*nu
        :return: array of shape (nz, n)
            ? is either n or n*nu
        """
        if len(Z.shape) == 2:
            Z = Z.reshape((1, ) + Z.shape)

        phi = self.model(X, Z)
        q = self.q(Z, X)
        val = np.log(phi) - np.log((q * self.noise(X) + self.eps))
        validate_shape(val.shape, (Z.shape[0], Z.shape[1]))

        return val

    def h2(self, Y):
        """ compute ratio \sum_z(model)/ (noise).

        :param Y: array of shape (n*nu, d)
            noise samples
        :return: array of shape (n*nu, )
        """
        model = self.model.marginalised_over_z(Y)  # (n*nu, )
        val = np.log(model) - np.log((self.noise(Y)))  # (n*nu, )

        return val

    def grad_wrt_theta(self, X, Y=None, reuse_latent_samples=False):
        """Computes grad of loss w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :param Y: array (N*nu, d)
            noise samples
        :return grad: array of shape (len(phi.theta), )
        """
        if Y is None:
            Y = self.Y
        if (self.ZX is None) or (not reuse_latent_samples):
            self.ZX = self.q.sample(self.nz, X)

        joint_noise = self.nu * self.noise(X) * self.q(self.ZX, X)
        a = joint_noise / (joint_noise + self.model(X, self.ZX))  # (nz, n)

        gradX = self.model.grad_log_wrt_params(X, self.ZX)  # (len(theta), nz, n)
        term_1 = np.mean(gradX * a, axis=(1, 2))  # (len(theta), )

        gradY = self.model.grad_log_visible_marginal_wrt_params(Y)  # (len(theta), nz, n)
        phiY = self.model.marginalised_over_z(Y)
        a1 = phiY / (self.nu*self.noise(Y) + phiY)  # (n)
        term_2 = - self.nu * np.mean(gradY*a1, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        validate_shape(grad.shape, self.model.theta_shape)

        return grad

    def grad_wrt_alpha(self, X):
        raise NotImplementedError

    def __repr__(self):
        return "MonteCarloVnceLossWithoutImportanceSampling"


class VnceLossWithAnalyticExpectations:
    """VNCE loss function and gradients where all expectations w.r.t q are assumed to be analytic

     There are 5 expectations required to make this work:
     1) E1 = E(r(u, z))
     2) E2 = E(log(phi(u,z)/ (phi(u, z) + nu pn(u)))
     3) E3 = E(grad_theta(log(phi(u,z)) (psi_1(u, z) - 1) / psi_1(u, z))
     4) E4 = E(grad_theta(log(phi(u,z)) r(u, z) )
     5) E5 = grad_alpha(E(log(1 + nu/r(u, z))))

     Where each expectation is over z ~ q(z | u; alpha).
     r and psi_1 are given by:
     r(u, z) = phi(u, z; theta) / q(z| u; alpha)*pn(u)
     psi_1(u, z) = 1 + (nu/r(u, z))
     """

    def __init__(self, model, noise, noise_samples, variational_q, E1, E2, E3, E4, E5, noise_to_data_ratio, separate_terms=False, eps=1e-7):
        self.model = model
        self.noise = noise
        self.Y = noise_samples
        self.q = variational_q
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.E4 = E4
        self.E5 = E5
        self.nu = noise_to_data_ratio
        self.separate_terms = separate_terms
        self.eps = eps
        self.current_loss = None
        self.ZX = None
        self.ZY = None

    def __call__(self, X, Y=None, reuse_latent_samples=False):
        """Return Monte Carlo estimate of Lower bound of NCE objective

       :param X: array (N, d)
           data sample we are training our model on
       :param Y: array (nz, N*nu)
           noise samples
       :param reuse_latent_samples: boolean
           If true, reuse the samples ZX and ZY saved on this class as attributes. This is useful
           during variational EM, since we want to reuse the same samples for every update in the M step.
       :return: float
           Value of objective for current parameters
       """
        if Y is None:
            Y = self.Y
        E2 = self.E2(X, self.model, self.q, self.noise, self.nu, self.eps)
        first_term = np.mean(E2)

        E1 = self.E1(Y, self.model, self.q, self.noise, self.eps)
        psi_2 = 1 + (1/self.nu)*E1
        second_term = -self.nu * np.mean(np.log(psi_2))

        if self.separate_terms:
            val = np.array([first_term, second_term])
        else:
            val = first_term + second_term

        self.current_loss = val

        return val

    def grad_wrt_theta(self, X, Y=None, reuse_latent_samples=False):
        """Computes grad of loss w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :param Y: array (N*nu, d)
            noise samples
        :return grad: array of shape (len(phi.theta), )
        """
        if Y is None:
            Y = self.Y
        E3 = self.E3(X, self.model, self.q, self.noise, self.nu, self.eps)  # (len(theta), n)
        term_1 = np.mean(E3, axis=1)

        E4 = self.E4(Y, self.model, self.q, self.noise, self.nu, self.eps)  # (len(theta), n)
        a = 1/(self._psi_2(Y) + self.eps)  # (n, )
        term_2 = - np.mean(a * E4, axis=1)

        grad = term_1 + term_2  # (len(theta), )

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        validate_shape(grad.shape, self.model.theta_shape)

        return grad

    def _psi_2(self, U):
        """Return array (n, )"""
        E1 = self.E1(U, self.model, self.q, self.noise, self.eps)
        return 1 + (1/self.nu) * E1

    def grad_wrt_alpha(self, X, reuse_latent_samples=False):
        """Compute J1_grad w.r.t theta using Monte-Carlo

        :param X: array (n, d)
            data sample we are training our model on.
        :return grad: array of shape (len(q.alpha), )
        """
        E5 = self.E5(X, self.model, self.q, self.noise, self.nu, self.eps)
        grad = np.mean(E5, axis=1)
        validate_shape(grad.shape, self.q.alpha_shape)

        return grad

    def __repr__(self):
        return "VnceLossWithAnalyticExpectations"
