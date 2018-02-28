"""Module provides classes for unnormalised models without latent variables
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from numpy import random as rnd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity as kd


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
        """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :return grad: array (len(theta)=2, nz, n)
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
