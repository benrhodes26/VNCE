"""Module provides classes for probability distributions.
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from numpy import random as rnd
from scipy.stats import norm


# noinspection PyPep8Naming
class Distribution(metaclass=ABCMeta):

    def __init__(self, alpha):
        """
        :param alpha: float or array
            parameter(s) of distribution
        """
        if isinstance(alpha, float) or isinstance(alpha, int):
            alpha = np.array([alpha])
        self._alpha = alpha
        self.alpha_shape = alpha.shape

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if isinstance(value, float) or isinstance(value, int):
            value = np.array([value])
        assert value.shape == self.alpha_shape, 'Tried to ' \
            'set alpha to array with shape {}, should be {}'.format(value.shape,
                                                                    self.alpha_shape)
        self._alpha = value

    @abstractmethod
    def __call__(self, U):
        """ evaluate distribution on datapoints U[i,:]

        :param Z: array (nz, n, m)
        :param U: array (n, d)
            n datapoints with d dimensions
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, nz):
        """Return n samples from the distribution

        :param n: sample size
        """
        raise NotImplementedError


# noinspection PyPep8Naming,PyMissingConstructor,PyMethodOverriding,PyMethodMayBeStatic
class PolynomialSigmoidBernoulli(Distribution):

    def __init__(self, alpha):
        """Init bernoulli distribution Ber(p)
        where p is parametrised by a sigmoid where:
        U = poly(u, degree) = [1, u, u^2, ..., u^degree]
        p(alpha) = 1 - sigmoid(np.dot(alpha.T, U))

        :param alpha: array (degree + 1, )
        :param degree: int
        """
        super().__init__(alpha)
        self.degree = self.alpha.shape[0] - 1

    def __call__(self, Z, U):
        """
        :param Z: array (nz, n, 1)
        :param U: array (n, 1)
            n data points with dimension 1
        :return: array (nz, n)
            probability data comes from gaussian
            with stdev = sigma1 (as opposed to the gaussian with stdev = alpha)
        """
        nz, n = Z.shape[0], Z.shape[1]
        U, Z = U.reshape(n), Z.reshape(nz, n)
        p1 = self.calculate_p(U)
        return (Z == 0)*(1 - p1) + (Z == 1)*p1

    def make_poly_design_matrix(self, u):
        """convert u -> [1, u, u^2, .., u^degree]

        :param u: array (n, 1)
        :return: array (n, degree + 1)
        """
        u = u.reshape(-1)
        U_poly = np.zeros((len(u), self.degree + 1))
        for i in range(self.degree + 1):
            U_poly[:, i] = u**i
        return U_poly

    def sigmoid(self, u):
        """ Standard sigmoid function"""
        return 1/(1 + np.exp(-u))

    def calculate_p(self, U):
        """Return probability binary variable is 1"""
        U = self.make_poly_design_matrix(U)  # (n, degree + 1)
        p1 = 1 - self.sigmoid(np.dot(U, self.alpha))  # (n, )
        return p1

    def grad_p_wrt_alpha(self, u):
        """ Return gradient of p w.r.t alpha
        :param U: array (n, 1)
            n data points with dimension 1
        :return: array (degree+1, n)
            grad of p w.r.t parameter alpha
        """
        p1 = self.calculate_p(u)  # (n, )
        U = self.make_poly_design_matrix(u).T  # (degree+1, n)

        return -p1*(1 - p1)*U

    def sample(self, nz, U):
        """For each datapoint U[i,:], return n

        :param nz: int
            number of samples per datapoint in U
        :param U: U: array (n, 1)
            n data points with dimension 1
        :return array of shape (nz, n, 1)
        """
        Z = np.zeros((nz, len(U),  1))
        for i in range(len(U)):
            Z[:, i, 0] = rnd.uniform(0, 1, nz) < self.calculate_p(U[i])

        return Z


# noinspection PyPep8Naming,PyMissingConstructor,PyMethodOverriding
class MixtureOfTwoGaussianBernoulliPosterior(Distribution):

    def __init__(self, alpha, sigma1):
        """Init bernoulli distribution given by
        the posterior over the latent binary variable
        in a mixture of 2 gaussians. See __call__
        method for exact formula.

        :param alpha: array (1, )
            note: must be an array, NOT a float.
        """
        self.sigma1 = sigma1
        super().__init__(alpha)

    def __call__(self, Z, u):
        """
        :param Z: array (nz, n, 1)
        :param u: array (n, 1)
            n data points with dimension 1
        :return: array (n, nz)
            probability data comes from gaussian
            with stdev = sigma1 (as opposed to the gaussian
            with stdev = alpha)
        """
        nz, n = Z.shape[0], Z.shape[1]
        u, Z = u.reshape(n), Z.reshape(nz, n)
        p1 = self.calculate_p(u)
        return (Z == 0)*(1 - p1) + (Z == 1)*p1

    def calculate_p(self, U):
        """Return probability binary variable is 1"""
        sigma = np.exp(self.alpha)
        a = norm.pdf(U, 0, sigma)
        b = norm.pdf(U, 0, self.sigma1)
        return b / (a + b)

    def sample(self, nz, U):
        """For each datapoint U[i,:], return n

        :param nz: int
            number of samples per datapoint in U
        :param U: U: array (n, 1)
            n data points with dimension 1
        :return array of shape (nz, n, 1)
        """
        Z = np.zeros((nz, len(U),  1))
        for i in range(len(U)):
            Z[:, i, 0] = rnd.uniform(0, 1, nz) < self.calculate_p(U[i])

        return Z


# noinspection PyPep8Naming,PyMissingConstructor
class GaussianNoise(Distribution):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, U):
        """evaluate probability of data U

        :param U: U: array (n, 1)
            n data points with dimension 1
        :return array of shape (n, )
        """
        return norm.pdf(U.reshape(-1), self.mean, self.std)

    def sample(self, num_noise_samples):
        """

        :param num_noise_samples: int
            number of noise samples used in NCE
        :return: array (num_noise_samples, 1)
        """
        return (self.mean + rnd.randn(num_noise_samples)*self.std).reshape(-1, 1)
