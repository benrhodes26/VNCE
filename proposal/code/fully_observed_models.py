"""Module provides classes for unnormalised models without latent variables
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from itertools import product
from numpy import random as rnd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity as kd
from utils import sigmoid, validate_shape
DEFAULT_SEED = 1083463236


# noinspection PyPep8Naming
class Model(metaclass=ABCMeta):

    def __init__(self, theta):
        """ Initialise parameters of model: theta.

        Theta should be a one-dimensional array or int/float"""
        if isinstance(theta, float) or isinstance(theta, int):
            theta = np.array([theta])
        assert theta.ndim == 1, 'Theta should have dimension 1, ' \
                                'not {}'.format(theta.ndim)
        self._theta = theta
        self.theta_shape = theta.shape

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

    def __init__(self, theta, sigma1=1):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta)

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
        x = (w == 0)*(rnd.randn(n)*sigma) + (w == 1)*(rnd.randn(n)*self.sigma1)
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
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

        self.norm_const = None
        self.W_shape = W.shape  # (d+1, m+1)
        super().__init__(W.reshape(-1))

    def __call__(self, U,  normalise=False, reset_norm_const=True):
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

