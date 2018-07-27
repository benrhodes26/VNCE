"""Module provides classes for unnormalised models without latent variables
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from itertools import product
from numpy import random as rnd
from matplotlib import pyplot as plt
from plot import *
from scipy.stats import norm, multivariate_normal
from sklearn.neighbors import KernelDensity as kd
from utils import sigmoid, validate_shape
DEFAULT_SEED = 1083463236


# noinspection PyPep8Naming
class Model(metaclass=ABCMeta):

    def __init__(self, theta, rng=None):
        """ Initialise parameters of model: theta.

        Theta should be a one-dimensional array or int/float"""
        if isinstance(theta, float) or isinstance(theta, int):
            theta = np.array([theta])
        assert theta.ndim == 1, 'Theta should have dimension 1, ' \
                                'not {}'.format(theta.ndim)
        self._theta = deepcopy(theta)
        self.theta_shape = theta.shape
        if not rng:
            self.rng = rnd.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if isinstance(value, float) or isinstance(value, int):
            value = np.array([value])
        assert value.ndim == 1, 'Theta should have dimension 1, ' \
                                'not {}'.format(value.ndim)
        assert value.shape == self.theta_shape, 'Tried to ' \
            'set theta to array with shape {}, should be {}'.format(value.shape,
                                                                    self.theta_shape)
        self._theta = value

    @abstractmethod
    def __call__(self, U):
        """Evaluate model on data

        :param U: array (n, d)
            n input data points to evaluate model on
        """
        raise NotImplementedError

    @abstractmethod
    def grad_log_wrt_params(self, U):
        """Returns nabla_theta(log(phi(x, theta)))

        :param U: array (n, d)
            data sample we are training our model on
        """
        raise NotImplementedError


# noinspection PyPep8Naming,PyArgumentList,PyTypeChecker
class SumOfTwoUnnormalisedGaussians(Model):
    """Sum of two unnormalisedGaussians (no latent variable)

    The form of the model is given by :
    phi(u; theta) = e^-c*(N(u; 0, sigma) + N(u; 0, sigma1))
    where sigma = np.exp(theta[1]), c = theta[0]
    """

    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n)
        """
        U = U.reshape(-1)
        a = np.exp(-self.theta[0])  # scaling parameter
        b = np.exp(self.theta[1])  # stdev parameter
        term_1 = np.exp(-U**2 / (2 * b**2))
        term_2 = np.exp(-U**2 / (2 * self.sigma1**2))

        return a*(term_1 + term_2)

    def grad_log_wrt_params(self, U):
        """ Nabla_theta(log(phi(u; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :return grad: array (len(theta)=2, n)
        """
        n = U.shape[0]
        grad = np.zeros((len(self.theta), n))

        grad[0] = -1  # grad w.r.t scaling param

        sigma = np.exp(self.theta[1])
        a = np.exp(-U**2 / (2 * sigma**2))
        b = np.exp(-U**2 / (2 * self.sigma1 ** 2))
        c = a / (a + b)  # (n, 1)

        d = (U**2 * np.exp(-2*self.theta[1]))  # (n, 1)
        grad[1] = (c*d).reshape(-1)  # (n, )

        correct_shape = self.theta_shape + (n, )
        assert grad.shape == correct_shape, ' ' \
            'gradient should have shape {}, got {} instead'.format(correct_shape,
                                                                   grad.shape)
        return grad

    def sample(self, n):
        """ Sample n values from the model, with uniform(0,1) dist over latent vars

        :param n: number of data points to sample
        :return: data array of shape (n, 1)
        """
        sigma = np.exp(self.theta[1])
        a = self.sigma1 / (sigma + self.sigma1)
        w = rnd.uniform(0, 1, n) < a
        x = (w == 0)*(self.rng.randn(n)*sigma) + (w == 1)*(self.rng.randn(n)*self.sigma1)
        return x.reshape(-1, 1)

    def normalised(self, U):
        """Return values of p(U), where p is the normalised distribution over x

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n, )
            probabilities of datapoints under p(x) i.e with z marginalised out
            and the distribution has been normalised
        """
        sigma = np.exp(self.theta[1])
        a = sigma / (sigma + self.sigma1)
        return a*norm.pdf(U.reshape(-1), 0, sigma) + (1 - a)*norm.pdf(U.reshape(-1), 0, self.sigma1)

        # noinspection PyUnusedLocal

    def plot_sample_density_against_true_density(self, X, figsize=(10, 7), bandwidth=0.2):
        """Compare kernel density estimate of sample X to true density

        :param X: array (N, 1)
            X is a sample, possibly generated from self.sample
        :param figsize: tuple
            size of figure
        :param bandwidth:
            bandwidth parameter passed to sklearn.KernelDensity
        """
        _ = plt.figure(figsize=figsize)
        u = np.arange(-10, 10, 0.01)

        x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        plt.plot(u, x_density_samples, label='kde')

        px = self.normalised(u)
        plt.plot(u, px, label='true', c='r', linestyle='--')

        plt.legend()
        plt.grid()


# noinspection PyPep8Naming,PyArgumentList,PyTypeChecker
class SumOfTwoUnnormalisedGaussians2(Model):
    """Sum of two unnormalisedGaussians (no latent variable)

    The form of the model is given by :
    phi(u; theta) = e^-c*(N(u; 0, sigma) + N(u; 0, sigma1))
    where sigma = np.exp(theta[1]), c = theta[0]
    """

    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n)
        """
        U = U.reshape(-1)
        return self.marginal_z_0(U) + self.marginal_z_1(U)

    def grad_log_wrt_params(self, U):
        """ Nabla_theta(log(phi(u; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :return grad: array (len(theta)=2, n)
        """
        n = U.shape[0]
        grad = np.zeros((len(self.theta), n))

        first_term = self.marginal_z_0(U)  # (n, )
        second_term = self.marginal_z_1(U)  # (n, )

        grad[0] = - first_term / (first_term + second_term)
        grad[1] = - second_term / (first_term + second_term)
        a = (U**2 * np.exp(-2 * self.theta[2])).reshape(-1)
        grad[2] = (first_term * a) / (first_term + second_term)

        correct_shape = self.theta_shape + (n, )
        assert grad.shape == correct_shape, ' ' \
            'gradient should have shape {}, got {} instead'.format(correct_shape,
                                                                   grad.shape)
        return grad

    def marginal_z_0(self, U):
        """ Return value of model for z=0
        :param U: array (n, 1)
            N is number of data points
        :return array (n,)
        """
        U = U.reshape(-1)
        c_0 = np.exp(-self.theta[0])  # 1st scaling parameter
        sigma = np.exp(self.theta[2])  # stdev parameter
        b = np.exp(-U**2 / (2*(sigma**2)))
        return c_0 * b

    def marginal_z_1(self, U):
        """ Return value of model for z=1
        :param U: array (n, 1)
            N is number of data points
        :return array (n,)
        """
        U = U.reshape(-1)
        c_1 = np.exp(-self.theta[1])  # 2nd scaling parameter
        b = np.exp(-U**2 / (2*(self.sigma1**2)))
        return c_1 * b

    def sample(self, n):
        """ Sample n values from the model, with uniform(0,1) dist over latent vars

        :param n: number of data points to sample
        :return: data array of shape (n, 1)
        """
        raise NotImplementedError

    def normalised(self, U):
        """Return values of p(U), where p is the normalised distribution over x

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n, )
            probabilities of datapoints under p(x) i.e with z marginalised out
            and the distribution has been normalised
        """
        raise NotImplementedError

        # noinspection PyUnusedLocal

    def plot_sample_density_against_true_density(self, X, figsize=(10, 7), bandwidth=0.2):
        """Compare kernel density estimate of sample X to true density

        :param X: array (N, 1)
            X is a sample, possibly generated from self.sample
        :param figsize: tuple
            size of figure
        :param bandwidth:
            bandwidth parameter passed to sklearn.KernelDensity
        """

        raise NotImplementedError


# noinspection PyPep8Naming,PyArgumentList,PyTypeChecker
class SumOfTwoNormalisedGaussians(Model):
    """Sum of two unnormalisedGaussians (no latent variable)

    The form of the model is given by :
    phi(u; theta) = ((sigma/(sigma+sigma1))N(u; 0, sigma) + (sigma1/(sigma+sigma1))N(u; 0, sigma1)),
    where sigma = np.exp(theta)
    """

    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n)
        """
        return self.term1(U) + self.term2(U)

    def term1(self, U):
        U = U.reshape(-1)
        sigma = np.exp(self.theta)  # stdev parameter
        a = sigma / (sigma + self.sigma1)
        return a*norm.pdf(U.reshape(-1), 0, sigma)

    def term2(self, U):
        U = U.reshape(-1)
        sigma = np.exp(self.theta)  # stdev parameter
        a = sigma / (sigma + self.sigma1)
        return (1 - a)*norm.pdf(U.reshape(-1), 0, self.sigma1)

    def grad_log_wrt_params(self, U):
        """ Nabla_theta(log(phi(u; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :return grad: array (len(theta)=1, n)
        """
        n = U.shape[0]

        sigma = np.exp(self.theta)
        a = -1/(sigma + self.sigma1)
        b = ((U**2/sigma**3) + a).reshape(-1)  # (n, )

        numerator = self.term1(U)*b + self.term2(U)*a  # (n, )
        denominator = self.term1(U) + self.term2(U)  # (n, )

        grad = (numerator / denominator).reshape(1, n)  # (1, n)

        correct_shape = self.theta_shape + (n, )
        assert grad.shape == correct_shape, ' ' \
            'gradient should have shape {}, got {} instead'.format(correct_shape,
                                                                   grad.shape)
        return grad

    def sample(self, n):
        """ Sample n values from the model, with uniform(0,1) dist over latent vars

        :param n: number of data points to sample
        :return: data array of shape (n, 1)
        """
        sigma = np.exp(self.theta)
        a = self.sigma1 / (sigma + self.sigma1)
        w = rnd.uniform(0, 1, n) < a
        x = (w == 0)*(self.rng.randn(n)*sigma) + (w == 1)*(self.rng.randn(n)*self.sigma1)
        return x.reshape(-1, 1)

        # noinspection PyUnusedLocal

    def plot_sample_density_against_true_density(self, X, figsize=(10, 7), bandwidth=0.2):
        """Compare kernel density estimate of sample X to true density

        :param X: array (N, 1)
            X is a sample, possibly generated from self.sample
        :param figsize: tuple
            size of figure
        :param bandwidth:
            bandwidth parameter passed to sklearn.KernelDensity
        """
        _ = plt.figure(figsize=figsize)
        u = np.arange(-10, 10, 0.01)

        x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        plt.plot(u, x_density_samples, label='kde')

        px = self.normalised(u)
        plt.plot(u, px, label='true', c='r', linestyle='--')

        plt.legend()
        plt.grid()


class MixtureOfTwoGaussians(Model):
    """Sum of two evenly weighted Gaussians
    """

    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n)
        """
        return self.term1(U) + self.term2(U)

    def term1(self, U):
        U = U.reshape(-1)
        sigma = np.exp(self.theta)
        return 0.5*norm.pdf(U.reshape(-1), 0, sigma)

    def term2(self, U):
        U = U.reshape(-1)
        return 0.5*norm.pdf(U.reshape(-1), 0, self.sigma1)

    def grad_log_wrt_params(self, U):
        """
        :param U: array (n, 1)
             either data or noise for NCE
        :return grad: array (len(theta)=1, n)
        """
        NotImplementedError

    def sample(self, n):
        """ Sample n values from the model, with uniform(0,1) dist over latent vars

        :param n: number of data points to sample
        :return: data array of shape (n, 1)
        """
        sigma = np.exp(self.theta)
        w = rnd.uniform(0, 1, n) < 0.5
        x = (w == 0)*(self.rng.randn(n)*sigma) + (w == 1)*(self.rng.randn(n)*self.sigma1)
        return x.reshape(-1, 1)

        # noinspection PyUnusedLocal

    def plot_sample_density_against_true_density(self, X, figsize=(10, 7), bandwidth=0.2):
        """Compare kernel density estimate of sample X to true density

        :param X: array (N, 1)
            X is a sample, possibly generated from self.sample
        :param figsize: tuple
            size of figure
        :param bandwidth:
            bandwidth parameter passed to sklearn.KernelDensity
        """
        # _ = plt.figure(figsize=figsize)
        # u = np.arange(-10, 10, 0.01)
        #
        # x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        # x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        # plt.plot(u, x_density_samples, label='kde')
        #
        # px = self.normalised(u)
        # plt.plot(u, px, label='true', c='r', linestyle='--')
        #
        # plt.legend()
        # plt.grid()
        raise NotImplementedError


class VisibleRestrictedBoltzmannMachine(Model):

    def __init__(self, W, rng=None):
        """RBM with hidden units summed out

        The sum over the hidden states of an RBM contains
        2^m terms. However, this sum can be broken into
        a product of sums over each hidden unit, which can
        be computed in O(m) time, which is what we do here.

        :param W: array (d+1, m+1)
            Weight matrix. d=num_visibles, m=num_hiddens
        """
        self.norm_const = None
        self.W_shape = W.shape  # (d+1, m+1)
        super().__init__(W.reshape(-1), rng=rng)

    def __call__(self, U, normalise=False, reset_norm_const=True):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, d)
             either data or noise for NCE
        :return array (n, )
            probability of each datapoint under the model
        :param normalise: bool
            if true, first normalise the distribution.
        :param reset_norm_const: bool
            if True, recalculate the normalisation constant using current
            parameters. If false, use already saved norm const. Note: this
            argument only matters if normalise=True.
        :return: array (n, )
            probability of each datapoint under the model
        """
        W = self.theta.reshape(self.W_shape)
        d, m = np.array(W.shape) - 1
        U = self.add_bias_terms(U)
        uW = np.dot(U, W)  # (n, m+1)
        exp_uW = np.exp(uW)
        exp_uW += np.concatenate((np.zeros((len(U), 1)), np.ones((len(U), m))), axis=1)
        val = np.product(exp_uW, axis=1)  # (n, )
        if normalise:
            if (not self.norm_const) or reset_norm_const:
                self.reset_norm_const()
            val /= self.norm_const

        return val

    def grad_log_wrt_params(self, U):
        """ Nabla_theta(log(phi(u; theta))) where phi is the unnormalised model

        :param U: array (n, d)
             either data or noise for NCE
        :return grad: array (len(theta), n)
        """
        W = self.theta.reshape(self.W_shape)
        d_add_1, m_add_1 = np.array(W.shape)
        n = len(U)
        U = self.add_bias_terms(U)

        W_1 = W[:, 1:]  # (d+1, m)
        W_2 = sigmoid(np.dot(U, W_1))  # (n, m)
        W_3 = np.concatenate((np.ones((n, 1)), W_2), axis=1)  # (n, m+1)

        U = np.repeat(U, m_add_1, axis=-1)  # (n, [d+1]*[m+1])
        W_4 = np.tile(W_3, d_add_1)  # (n, [d+1]*[m+1])

        grad = U * W_4
        grad = grad.T  # ([d+1]*[m+1], n)
        validate_shape(grad.shape, self.theta_shape + (n, ))

        return grad

    def add_bias_terms(self, V):
        """prepend 1s along final dimension
        :param V: array (..., k)
        :return: array (..., k+1)
        """
        # add bias terms
        V_bias = np.ones(V.shape[:-1] + (1, ))
        V = np.concatenate((V_bias, V), axis=-1)
        return V

    def reset_norm_const(self):
        """Reset normalisation constant using current theta"""
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        d, m = W.shape[0] - 1, W.shape[1] - 1
        assert d*(2**m) <= 10**7, "Calculating the normalisation" \
            "constant has O(d 2**m) cost. Assertion raised since d*2^m " \
            "equal to {}, which exceeds the current limit of 10^7,".format(d*(2**m))

        W_times_all_latents = self.get_z_marginalization_matrix(W)  # (d+1, 2**m)
        W_times_all_latents = np.exp(W_times_all_latents)
        W_times_all_latents += np.concatenate((np.zeros((1, 2**m)),
                                               np.ones((d, 2**m))), axis=0)
        self.norm_const = np.sum(np.product(W_times_all_latents, axis=0))

    def get_z_marginalization_matrix(self, W):
        """Stack each of the 2**m possible binary latent vectors into a
        (m+1, 2**m) matrix Z, and then return the matrix multiply WZ

        :param W: array (d+1, m+1)
            weight matrix of RBM
        :return: array (d+1, 2**m)
            WZ, where Z contains all possible m-length binary vectors
        """
        m = W.shape[1] - 1
        Z = self.get_all_binary_vectors(m)  # (2**m, m)
        Z = self.add_bias_terms(Z)  # (2**m, m+1)
        Wz = np.dot(W, Z.T)  # (d+1, 2**m)

        return Wz

    def get_all_binary_vectors(self, k):
        """Return matrix of all k-length binary vectors

        :param k: int
        :return: array (2**k, k)
        """
        assert k <= 20, "Won't construct all binary vectors with dimension {}. " \
            "maximum dimension is 20, since this operation has O(2**k) cost".format(k)

        binary_pairs = [[0, 1] for _ in range(k)]
        all_binary_vectors = np.array(list(product(*binary_pairs)))  # (2**k, k)

        return all_binary_vectors


# class UnnormalisedTruncNorm(Model):
#
#     def __init__(self, scaling_param, mean, chol, rng=None):
#         """
#         :param scaling_param: array (1, )
#         :param mean: array (d, )
#         :param chol: array (d, d)
#             Lower triangular cholesky decomposition
#         :param rng:
#         :return:
#         """
#         self.mean_len = len(mean)
#
#         # cholesky of precision with log of diagonal elements (to enforce positivity)
#         chol = deepcopy(chol)
#         idiag = np.diag_indices_from(chol)
#         chol[idiag] = np.log(chol[idiag])
#         lower_chol = chol[np.tril_indices(self.mean_len)]
#         self.chol_len = len(lower_chol)
#
#         theta = np.concatenate((scaling_param, mean.reshape(-1), lower_chol.reshape(-1)))
#         super().__init__(theta, rng=rng)
#
#     def __call__(self, U, log=False):
#         """Evaluate unnormalised model
#
#         :param U: array (n, k)
#             observed data
#         :param log: boolean
#             if True, return value of logphi, where phi is the unnormalised model
#         """
#         scaling_param = deepcopy(self.theta[0])
#         mean, chol, chol_diag = self.get_mean_and_chol()
#         truncation_mask = np.all(U >= 0, axis=-1)  # (n, )
#
#         U_centred = U - mean
#         P = np.dot(chol, chol.T)  # (k, k) - precision matrix
#         VP = np.dot(U_centred, P)  # (n, k)
#         power = -0.5 * np.sum(VP * U_centred, axis=-1)  # (n, )
#         log_norm_const = (-self.mean_len / 2) * np.log(2 * np.pi) + np.sum(np.log(chol_diag))
#
#         val = -scaling_param + log_norm_const + power
#         if log:
#             val = truncation_mask * val + (1 - truncation_mask) * -15  # should be -infty, but -15 avoids numerical issues
#         else:
#             val = truncation_mask * np.exp(val)
#
#         return val  # (n, )
#
#     def grad_log_wrt_params(self, U):
#         """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model
#
#         :param U: array (n, d)
#              either data or noise for NCE
#         :param Z: (nz, n, m) or (n, m)
#             m-dimensional latent variable samples. nz per datapoint in U.
#         :return grad: array (len(theta), nz, n)
#         """
#         truncation_mask = np.all(U >= 0, axis=-1)  # (n, )
#
#         mean, chol, chol_diag = self.get_mean_and_chol()
#         ones_with_chol_diag = np.ones_like(chol)  # (d , d)
#         ones_with_chol_diag[np.diag_indices_from(ones_with_chol_diag)] = chol_diag
#
#         grad_wrt_to_scaling_param = np.ones((1, len(U))) * -1  # (1, n)
#
#         U_centred = U - mean
#         P = np.dot(chol, chol.T)  # (d, d) - precision matrix
#         grad_wrt_to_mean = np.dot(U_centred, P.T).T  # (d, n)
#
#         V1 = np.repeat(U_centred, self.mean_len, axis=-1)  # (n, d*d)
#         V2 = np.tile(U_centred, self.mean_len)  # (n, d*d)
#         v3_shape = (len(U), ) + (self.mean_len, self.mean_len)
#         V3 = (V1 * V2).reshape(v3_shape)  # (n, d, d)
#         V4 = np.dot(V3, chol)  # (n, d, d)
#         V4 *= ones_with_chol_diag  # (n, d, d)
#         V5 = (np.identity(self.mean_len) - V4)  # (n, d, d)
#
#         ilower = np.tril_indices(self.mean_len)
#         grad_wrt_chol = V5[:, ilower[0], ilower[1]].reshape((len(U), -1)).T  # (d(d+1)/2, n)
#
#         grad = np.concatenate((grad_wrt_to_scaling_param, grad_wrt_to_mean, grad_wrt_chol), axis=0)  # (len(theta), n)
#         grad *= truncation_mask  # truncate
#
#         return grad
#
#     def sample(self, n):
#         """ Sample from a truncated multivariate normal with rejection sampling with uniform proposal (note: this won't scale)
#         :param n: sample size
#         """
#         mean, chol, chol_diag = self.get_mean_and_chol()
#         stds = (1 / chol_diag)
#         low_proposal = np.maximum(mean - (5 * stds), 0)
#         high_proposal = mean + (5 * stds)
#         k = len(mean)
#
#         sample = np.zeros((n, k))
#         total_n_accepted = 0
#         expected_num_proposals_per_accept = (25 / (2 * np.pi))**(k/2)
#         proposal_size = int(4 * n * expected_num_proposals_per_accept)
#         if proposal_size > 10**8:
#             print('WARNING: GENERATING THE MAXIMUM NUMBER (10**8) OF SAMPLES FROM THE PROPOSAL DISTRIBUTION INSIDE A WHILE LOOP,'
#                   ' AS PART OF THE ACCEPT-REJECT ALGORITHM. THIS COULD TAKE A VERY LONG TIME. THE DIMENSIONALITY OF YOUR DATA MAY'
#                   ' BE TOO HIGH')
#             proposal_size = 10**8
#
#         print('sampling from the model...')
#         while total_n_accepted < n:
#             proposal = self.rng.uniform(low_proposal, high_proposal, (proposal_size, k))  # (proposal_size, k)
#             V = proposal - mean
#
#             P = np.dot(chol, chol.T)  # (k, k) - precision matrix
#             VP = np.dot(V, P)  # (proposal_size, k)
#             acceptance_prob = np.exp(-0.5 * np.sum(VP * V, axis=-1))  # (proposal_size, )
#             accept = self.rng.uniform(0, 1, proposal_size) < acceptance_prob
#             accepted = proposal[accept]
#
#             n_accepted = len(accepted)
#             if total_n_accepted + n_accepted >= n:
#                 remain = n - total_n_accepted
#                 sample[total_n_accepted:] = accepted[:remain]
#             else:
#                 sample[total_n_accepted: total_n_accepted+n_accepted] = accepted
#             total_n_accepted += n_accepted
#             print('total num samples from model accepted: {}'.format(min(total_n_accepted, n)))
#         print('finished sampling!')
#
#         return sample
#
#     def get_mean_and_chol(self):
#         """get mean and lower triangular cholesky of the precision matrix"""
#         mean, chol = deepcopy(self.theta[1:1+self.mean_len]), deepcopy(self.theta[1+self.mean_len:])
#
#         # lower-triangular cholesky decomposition of the precision
#         chol_mat = np.zeros((self.mean_len, self.mean_len))
#         ilower = np.tril_indices(self.mean_len)
#         chol_mat[ilower] = chol
#
#         # exp the diagonal, to enforce positivity
#         idiag = np.diag_indices_from(chol_mat)
#         chol_mat[idiag] = np.exp(chol_mat[idiag])
#
#         return mean, chol_mat, chol_mat[idiag]


class UnnormalisedTruncNorm(Model):

    def __init__(self, scaling_param, mean, precision, rng=None):
        """
        :param scaling_param: array (1, )
        :param mean: array (d, )
        :param chol: array (d, d)
            Lower triangular cholesky decomposition
        :param rng:
        :return:
        """

        self.mean_len = len(mean)

        # cholesky of precision with log of diagonal elements (to enforce positivity)
        prec = deepcopy(precision)
        idiag = np.diag_indices_from(prec)
        prec[idiag] = np.log(prec[idiag])
        lower_prec = prec[np.tril_indices(self.mean_len)]
        self.chol_len = len(lower_prec)

        theta = np.concatenate((scaling_param, mean.reshape(-1), lower_prec.reshape(-1)))
        super().__init__(theta, rng=rng)

    def __call__(self, U, log=False):
        """Evaluate unnormalised model

        :param U: array (n, k)
            observed data
        :param log: boolean
            if True, return value of logphi, where phi is the unnormalised model
        """
        scaling_param = deepcopy(self.theta[0])
        mean, precision, _, _ = self.get_mean_and_lprecision()
        truncation_mask = np.all(U >= 0, axis=-1)  # (nz, n)

        V_centred = U - mean
        VP = np.dot(V_centred, precision)  # (nz, n, k)
        power = -0.5 * np.sum(VP * V_centred, axis=-1)  # (nz, n)
        # log_norm_const = (-self.mean_len / 2) * np.log(2 * np.pi) + np.sum(np.log(chol_diag))

        # val = -scaling_param + log_norm_const + power
        val = -scaling_param + power
        if log:
            val = truncation_mask * val + (1 - truncation_mask) * -15  # should be -infty, but -15 avoids numerical issues
        else:
            val = truncation_mask * np.exp(val)

        return val  # (nz, n)

    def grad_log_wrt_params(self, U):
        """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, d)
             either data or noise for NCE
        :param Z: (nz, n, m) or (n, m)
            m-dimensional latent variable samples. nz per datapoint in U.
        :return grad: array (len(theta), nz, n)
        """
        n = U.shape[0]
        truncation_mask = np.all(U >= 0, axis=-1)  # (n, )

        mean, precision, lprecision, precision_diag = self.get_mean_and_lprecision()
        ones_with_prec_diag = np.ones_like(precision)  # (d , d)
        ones_with_prec_diag[np.diag_indices_from(ones_with_prec_diag)] = precision_diag

        grad_wrt_to_scaling_param = np.ones((1, n)) * -1  # (1, n)

        V_centred = U - mean  # (n, d)
        grad_wrt_to_mean = np.dot(V_centred, precision.T)  # (n, d)
        grad_wrt_to_mean = grad_wrt_to_mean.T  # (d, n)

        V1 = np.repeat(V_centred, self.mean_len, axis=-1)  # (n, d*d)
        V2 = np.tile(V_centred, self.mean_len)  # (n, d*d)
        v3_shape = (n, self.mean_len, self.mean_len)
        V3 = - (V1 * V2).reshape(v3_shape)  # (n, d, d)

        V4 = 0.5 * V_centred**2  # (n, d)
        idiag = np.diag_indices(self.mean_len)
        V3[:, idiag[0], idiag[1]] += V4  # (n, d, d)
        V3 *= ones_with_prec_diag  # because we optimise log of diagonals

        ilower = np.tril_indices(self.mean_len)
        grad_wrt_lprecision = V3[:, ilower[0], ilower[1]].reshape(n, -1)  # (n, d(d+1)/2)
        grad_wrt_lprecision = grad_wrt_lprecision.T  # (d(d+1)/2, n)

        grad = np.concatenate((grad_wrt_to_scaling_param, grad_wrt_to_mean, grad_wrt_lprecision), axis=0)  # (len(theta), n)
        grad *= truncation_mask  # truncate

        return grad

    def sample(self, n):
        """ Sample from a truncated multivariate normal with rejection sampling with uniform proposal (note: this won't scale)
        :param n: sample size
        """
        # mean, chol, chol_diag = self.get_mean_and_chol()
        # stds = (1 / chol_diag)
        mean, precision, lprecision, precision_diag = self.get_mean_and_lprecision()
        cov = np.linalg.inv(precision)
        stds = np.diag(cov)**0.5
        low_proposal = np.maximum(mean - (5 * stds), 0)
        high_proposal = mean + (5 * stds)
        k = len(mean)

        sample = np.zeros((n, k))
        total_n_accepted = 0
        expected_num_proposals_per_accept = (100 / (2 * np.pi))**(k/2)
        proposal_size = int(4 * n * expected_num_proposals_per_accept)
        if proposal_size > 10**8:
            print('WARNING: GENERATING THE MAXIMUM NUMBER (10**8) OF SAMPLES FROM THE PROPOSAL DISTRIBUTION INSIDE A WHILE LOOP,'
                  ' AS PART OF THE ACCEPT-REJECT ALGORITHM. THIS COULD TAKE A VERY LONG TIME. THE DIMENSIONALITY OF YOUR DATA MAY'
                  ' BE TOO HIGH')
            proposal_size = 10**8

        print('sampling from the model...')
        while total_n_accepted < n:
            proposal = self.rng.uniform(low_proposal, high_proposal, (proposal_size, k))  # (proposal_size, k)
            V = proposal - mean

            # P = np.dot(chol, chol.T)  # (k, k) - precision matrix
            VP = np.dot(V, precision)  # (proposal_size, k)
            acceptance_prob = np.exp(-0.5 * np.sum(VP * V, axis=-1))  # (proposal_size, )
            accept = self.rng.uniform(0, 1, proposal_size) < acceptance_prob
            accepted = proposal[accept]

            n_accepted = len(accepted)
            if total_n_accepted + n_accepted >= n:
                remain = n - total_n_accepted
                sample[total_n_accepted:] = accepted[:remain]
            else:
                sample[total_n_accepted: total_n_accepted+n_accepted] = accepted
            total_n_accepted += n_accepted
            print('total num samples from model accepted: {}'.format(min(total_n_accepted, n)))
        print('finished sampling!')

        return sample

    def get_mean_and_lprecision(self):
        """get mean and lower triangular elements of the precision matrix (note: we get log of diagonal elements)"""
        mean, lprec = deepcopy(self.theta[1:1+self.mean_len]), deepcopy(self.theta[1+self.mean_len:])

        # lower-triangular cholesky decomposition of the precision
        lprecision = np.zeros((self.mean_len, self.mean_len))
        ilower = np.tril_indices(self.mean_len)
        lprecision[ilower] = lprec

        # exp the diagonal, to enforce positivity
        idiag = np.diag_indices_from(lprecision)
        lprecision[idiag] = np.exp(lprecision[idiag])

        precision = lprecision + lprecision.T - np.diag(np.diag(lprecision))

        return mean, precision, lprecision, precision[idiag]
