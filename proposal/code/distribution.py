"""Module provides classes for probability distributions.
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numpy import random as rnd
from pomegranate import BayesianNetwork
from plot import *
from scipy.stats import norm, multivariate_normal
from utils import sigmoid, validate_shape

DEFAULT_SEED = 1083463236


# noinspection PyPep8Naming
class Distribution(metaclass=ABCMeta):

    def __init__(self, alpha, rng=None):
        """
        :param alpha: float or array
            parameter(s) of distribution
        """
        if isinstance(alpha, float) or isinstance(alpha, int):
            alpha = np.array([alpha])
        self._alpha = alpha
        self.alpha_shape = alpha.shape
        if not rng:
            self.rng = rnd.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

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

        :param U: array (n, d)
            n datapoints with d dimensions
        :return array (n, )
            probability of each input datapoint
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, nz):
        """Return n samples from the distribution

        :param nz: int
            sample size
        :return array (n, )
            a sample from the distribution
        """
        raise NotImplementedError


# noinspection PyPep8Naming,PyMissingConstructor,PyMethodOverriding,PyMethodMayBeStatic
class PolynomialSigmoidBernoulli(Distribution):

    def __init__(self, alpha, rng=None):
        """Init bernoulli distribution Ber(p)
        where p is parametrised by a sigmoid where:
        U = poly(u, degree) = [1, u, u^2, ..., u^degree]
        p(alpha) = 1 - sigmoid(np.dot(alpha.T, U))

        :param alpha: array (degree + 1, )
        """
        super().__init__(alpha, rng=rng)
        self.degree = self.alpha.shape[0] - 1

    def __call__(self, Z, U):
        """
        :param Z: array (nz, n, 1)
        :param U: array (n, 1)
            n data points with dimension 1
        :return: array (nz, n)
            probability data comes from gaussian with stdev = sigma1
            (as opposed to the gaussian with stdev = alpha)
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

    def calculate_p(self, U):
        """Return probability binary variable is 1
        :param U: array (n, 1)
        :return: array (n,)
        """
        U = self.make_poly_design_matrix(U)  # (n, degree + 1)
        p1 = 1 - sigmoid(np.dot(U, self.alpha))  # (n, )
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
        Z = np.zeros((nz, len(U), 1))
        for i in range(len(U)):
            Z[:, i, 0] = self.rng.uniform(0, 1, nz) < self.calculate_p(U[i])

        return Z


# noinspection PyPep8Naming,PyMissingConstructor,PyMethodOverriding
class MixtureOfTwoGaussianBernoulliPosterior(Distribution):

    def __init__(self, alpha, sigma1, rng=None):
        """Init bernoulli distribution given by
        the posterior over the latent binary variable
        in a mixture of 2 gaussians. See __call__
        method for exact formula.

        :param alpha: array (1, )
            note: must be an array, NOT a float.
        """
        self.sigma1 = sigma1
        super().__init__(alpha, rng=rng)

    def __call__(self, Z, u):
        """
        :param Z: array (nz, n, 1)
        :param u: array (n, 1)
            n data points with dimension 1
        :return: array (n, nz)
            probability data comes from gaussian with stdev = sigma1
            (as opposed to the gaussian with stdev = alpha)
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
        Z = np.zeros((nz, len(U), 1))
        for i in range(len(U)):
            Z[:, i, 0] = self.rng.uniform(0, 1, nz) < self.calculate_p(U[i])

        return Z


# noinspection PyPep8Naming,PyMissingConstructor,PyMethodOverriding
class RBMLatentPosterior(Distribution):
    # todo: write docstring for class

    def __init__(self, W, rng=None):
        """Initialise P(z|x) for restricted boltzmann machine

        :param W: array (d+1, m+1)
            Weight matrix for restricted boltzmann machine.
            See relevant class in latent_variable_model.py
        """
        self.W_shape = W.shape  # (d+1, m+1)
        alpha = W.reshape(-1)
        super().__init__(alpha, rng=rng)

    def __call__(self, Z, U):
        """
        :param Z: array (nz, n, m)
        :param U: array (n, d)
            n data points with dimension d
        :return: array (nz, n)
            probability of latent variables given data
        """
        p1 = self.calculate_p(U)  # (n, m)

        posteriors = (Z == 0)*(1 - p1) + (Z == 1)*p1  # (nz, n, m)
        posterior = np.product(posteriors, axis=-1)  # (nz, n)

        return posterior

    def add_bias_terms(self, U):
        """prepend U and Z with 1s
        :param U: array (n, d)
            either data or noise for NCE
        :return: array (n, d+1)
        """
        # add bias terms
        U_bias = np.ones(len(U)).reshape(-1, 1)
        U = np.concatenate((U_bias, U), axis=-1)
        return U

    def calculate_p(self, U):
        """probability that binary latents equal 1, for each datapoint
        :param U: array (n, d)
            n (visible) data points that we condition on
        :return: array (n, m)
            Conditioning on each of our n data points, calculate the
            probability that each of the m latent hidden variables
            is equal to 1
        """
        # only add bias term to U, not Z
        U = self.add_bias_terms(U)  # (n, d+1)
        W = self.alpha.reshape(self.W_shape)

        a = np.dot(U, W[:, 1:])  # (n, m)
        p1 = sigmoid(a)

        return p1

    def sample(self, nz, U):
        """For each datapoint U[i,:], return nz latent variables

        :param nz: tuple
            number of latent samples per datapoint
        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (nz, n, m)
        """
        p1 = self.calculate_p(U)  # (n, m)
        Z_shape = (nz, ) + p1.shape

        Z = self.rng.uniform(0, 1, Z_shape) < p1  # (nz, n, m)

        return Z.astype(int)


# noinspection PyMissingConstructor,PyMethodOverriding
class StarsAndMoonsPosterior(Distribution):

    def __init__(self, nn, rng=None):
        """Initialise approx posterior q(z|x) for StarsAndMoonsModel (see latent_variable_model.py)

        :params nn: neural network object
            neural network needs to have an fprop method that yields the output of the network given the input
        """
        self.nn = nn
        if rng:
            self.rng = rng
        else:
            self.rng = rnd.RandomState(DEFAULT_SEED)

    def __call__(self, Z, U, log=False, outputs=None):
        mean, cov = self.get_mean_and_cov(U, outputs)
        cholesky = cov**0.5
        term1 = -np.log(2 * np.pi * np.product(cholesky, axis=-1))  # (n, )
        term2 = -0.5 * np.sum((Z - mean)**2 / cov, axis=-1)  # (nz, n)

        val = term1 + term2  # (nz, n)
        if not log:
            val = np.exp(val)

        return val

    def get_mean_and_cov(self, U, outputs=None):
        """
        :param U: array (n, d)
        :return mean, cov: arrays (n, 2)
        """
        if outputs is None:
            outputs = self.nn.fprop(U)[-1]
        mean, cov = outputs[:, :2], np.exp(outputs[:, 2:])**2
        return mean, cov

    def sample(self, nz, U, outputs=None):
        E = self.sample_E(nz, len(U))  # (nz, n, 2)
        Z = self.get_Z_samples_from_E(E, U, outputs=outputs)  # (nz, n, 2)
        return Z

    def sample_E(self, nz, n):
        samples = self.rng.multivariate_normal(mean=np.zeros(2), cov=np.identity(2), size=(nz, n))
        return samples.reshape(nz, n, 2)

    def get_Z_samples_from_E(self, E, U, outputs=None):
        """Converts samples E from some simple base distribution to samples from posterior

        This function is needed to apply the reparametrization trick. E is from a standard normal Gaussian
        and Z is from a Gaussian whose mean and covariance depends on a neural network with parameters alpha.

        :param E: array (nz, n, 2)
            random variable to be transformed into Z (via reparam trick)
        :param U: array (n, d)
        """
        mean, cov = self.get_mean_and_cov(U, outputs)  # (n, 2)
        cholesky = cov**0.5
        Z = E * cholesky + mean  # (nz, n, 2)
        validate_shape(Z.shape, E.shape)

        return Z

    def grad_of_Z_wrt_nn_outputs(self, outputs, E):
        """gradient of Z w.r.t to the outputs of the neural network

        :param outputs: array (n, 4)
            outputs of neural net
        :param E: array (nz, n, 2)
            random variable to be transformed into Z (via reparam trick)
        :return: array (4, nz, n, 2)
            grad of z1 and z2 w.r.t alpha
        """
        grad_shape = (outputs.shape[1], ) + E.shape  # (4, nz, n, 2)
        grad = np.zeros(grad_shape)
        grad[0, :, :, 0] = 1
        grad[1, :, :, 1] = 1
        grad[2, :, :, 0] = E[:, :, 0] * np.exp(outputs[:, 2])
        grad[3, :, :, 1] = E[:, :, 1] * np.exp(outputs[:, 3])

        return grad  # (4, nz, n, 2)

    def entropy(self, U, outputs=None):
        _, cov = self.get_mean_and_cov(U, outputs)  # (n, 2)
        return 1 + np.log(2*np.pi) + (0.5*np.log(np.product(cov, axis=1)))  # (n, )

    def grad_log_wrt_nn_outputs(self, outputs, Z):
        grad_shape = outputs.shape  # (n, 2)
        grad = np.zeros(grad_shape)

        grad[:, :2] = 0  # mean of gaussian does not contribute to the entropy
        grad[:, 2:] = 1

        return grad


class MissingDataProductOfTruncNormsPosterior(Distribution):

    def __init__(self, nn, data_dim, rng=None):
        self.nn = nn
        self.dim = data_dim
        if rng:
            self.rng = rng
        else:
            self.rng = rnd.RandomState(DEFAULT_SEED)

    def __call__(self, Z, U, log=False, nn_outputs=None):
        miss_mask = self.get_miss_mask(Z)  # (n, k)
        truncation_mask = np.all(Z >= 0, axis=-1)  # (nz, n)
        if np.sum(miss_mask) == 0:
            # no missing data, so log probabilities are all 0
            val = np.zeros(Z.shape[:2])
        else:
            mean, chol = self.get_mean_and_chol(U, miss_mask, nn_outputs)  # (n, k)  - cholesky of diagonal precision matrix
            precision = chol**2

            power = -(1/2) * precision * (Z - mean)**2  # (nz, n, k)
            log_chol = np.log(chol, out=np.zeros_like(chol), where=chol != 0)
            log_norm_const_1 = (-1/2) * np.log(2 * np.pi) + log_chol  # (n, k) - norm const for usual Gaussian
            log_norm_const_2 = - np.log(1 - norm.cdf(-mean * chol))  # (n, k) - norm const due to truncation
            log_probs = power + log_norm_const_1 + log_norm_const_2  # (nz, n, k)

            log_probs *= miss_mask  # mask out observed data
            val = np.sum(log_probs, axis=-1)  # (nz, n)
        if log:
            val = truncation_mask * val + (1 - truncation_mask) * -15  # should be -infty, but -15 avoids numerical issues
        else:
            val = truncation_mask * np.exp(val)

        return val  # (nz, n)

    def grad_log_wrt_nn_outputs(self, nn_outputs, grad_z_wrt_nn_outputs, Z):
        miss_mask = self.get_miss_mask(Z)  # (n, k)
        truncation_mask = np.all(Z >= 0, axis=-1)  # (nz, n)

        mean, chol = self.get_mean_and_chol(U=None, miss_mask=miss_mask, nn_outputs=nn_outputs)  # (n, k)  - cholesky of diagonal precision matrix
        precision = chol**2

        grad_Z_wrt_nn_output1 = grad_z_wrt_nn_outputs[:, :, :self.dim]  # (nz, n, k) - grads wrt mean
        grad_Z_wrt_nn_output2 = grad_z_wrt_nn_outputs[:, :, self.dim:]  # (nz, n, k) - grads wrt to np.log(cholesky)

        grad_log_wrt_z = - precision * (Z - mean)  # (nz, n, k)
        # self.checkMask(grad_log_wrt_z, miss_mask)

        grad_log_through_z_1 = np.transpose(grad_log_wrt_z * grad_Z_wrt_nn_output1, [2, 0, 1])  # (k, nz, n)
        grad_log_through_z_2 = np.transpose(grad_log_wrt_z * grad_Z_wrt_nn_output2, [2, 0, 1])  # (k, nz, n)
        grad_log_through_z = np.concatenate((grad_log_through_z_1, grad_log_through_z_2), axis=0)  # (len(nn_outputs), nz, n)

        a = norm.pdf(-mean * chol)  # (n, k)
        b = 1 / (1 - norm.cdf(-chol * mean))
        c = a * b * miss_mask

        grad_log_wrt_nn_output1 = (precision * (Z - mean)) - c  # (nz, n, k)
        grad_log_wrt_nn_output2 = 1 - (precision * (Z - mean)**2) - (c * mean)  # (nz, n, k)
        grad_log_wrt_nn_output1 *= miss_mask
        grad_log_wrt_nn_output2 *= miss_mask

        grad_log_wrt_nn_output1 = np.transpose(grad_log_wrt_nn_output1, [2, 0, 1])  # (k, nz, n)
        grad_log_wrt_nn_output2 = np.transpose(grad_log_wrt_nn_output2, [2, 0, 1])  # (k, nz, n)
        grad_log_wrt_nn_output1 *= truncation_mask
        grad_log_wrt_nn_output2 *= truncation_mask
        grad_log_wrt_nn_outputs = np.concatenate((grad_log_wrt_nn_output1, grad_log_wrt_nn_output2), axis=0)  # (len(nn_outputs), nz, n)

        grad = grad_log_through_z + grad_log_wrt_nn_outputs  # (len(nn_outputs), nz, n)

        return grad

    def grad_of_Z_wrt_nn_outputs(self, nn_outputs, E):
        """gradient of Z w.r.t to the outputs of the neural network

        :param outputs: array (n, len(nn_outputs))
            outputs of neural net
        :param E: array (nz, n, self.dim)
            random variable to be transformed into Z (via reparam trick)
        :return: array (nz, n, len(nn_outputs))
            grad of each z_i w.r.t neural net output
        """
        miss_mask = self.get_miss_mask(E)  # (nz, n, k)
        grad_shape = E.shape[:2] + (nn_outputs.shape[1], )  # (nz, n, len(nn_outputs))
        mean, chol = self.get_mean_and_chol(U=None, miss_mask=miss_mask, nn_outputs=nn_outputs)  # (n, k)  - cholesky of diagonal precision matrix

        E_minus_1 = (E - 1) * miss_mask  # (nz, n, k)
        alpha = -mean * chol  # (n, k)
        a = (norm.cdf(alpha) * - E_minus_1) + E  # (nz, n, k)
        a *= miss_mask
        b = self.normal_cdf_inverse(a)  # (nz, n, k)
        c = norm.pdf(b) * miss_mask  # (nz, n, k)
        d = np.divide(norm.pdf(alpha) * miss_mask, c, out=np.zeros_like(c), where=c != 0)
        e = np.divide(b, chol, out=np.zeros_like(b), where=chol != 0)

        grad_wrt_mean = d * E_minus_1 + 1  # (nz, n, k)
        grad_wrt_mean *= miss_mask

        grad_wrt_log_chol = -e + (d * E_minus_1 * mean)  # (nz, n, k)

        # self.checkMask(grad_wrt_mean, miss_mask)
        # self.checkMask(grad_wrt_log_chol, miss_mask)

        grad = np.zeros(grad_shape)
        grad[:, :, :self.dim] = grad_wrt_mean  # (nz, n, k)
        grad[:, :, self.dim:] = grad_wrt_log_chol  # (nz, n, k)

        return grad  # (nz, n, len(nn_outputs))

    def sample(self, nz, U, miss_mask=None, nn_outputs=None):
        if miss_mask is None:
            Z = np.zeros((nz, ) + U.shape)
        else:
            E = self.sample_E(nz, miss_mask)  # (nz, n, 2)
            Z = self.get_Z_samples_from_E(nz, E, U, miss_mask, nn_outputs=nn_outputs)  # (nz, n, 2)
        return Z

    def sample_E(self, nz, miss_mask):
        return self.rng.uniform(0, 1, size=(nz, ) + miss_mask.shape) * miss_mask

    def get_Z_samples_from_E(self, nz, E, U, miss_mask, nn_outputs=None):
        """Converts samples E from some simple base distribution to samples from posterior

        This function is needed to apply the reparametrization trick. E is from a uniform distribution
        and Z is from a product of truncated normals whose location & scale depend on a neural network with parameters alpha.

        :param E: array (nz, n, d)
            random variable to be transformed into Z (via reparam trick)
        :param U: array (n, d)
        """
        mean, chol = self.get_mean_and_chol(U=U, miss_mask=miss_mask, nn_outputs=nn_outputs)  # (n, d)  - cholesky of diagonal precision matrix
        one_minus_E = (1 - E) * miss_mask  # (nz, n, d)
        a = (norm.cdf(-mean * chol) * one_minus_E) + E  # (nz, n, d)
        Z = self.normal_cdf_inverse(a)  # (nz, n, d)
        Z *= miss_mask  # (nz, n, d)
        Z = np.divide(Z, chol, out=np.zeros_like(Z), where=chol != 0)
        Z += mean

        # self.checkMask(Z, miss_mask)
        return Z  # (nz, n, d)

    def get_mean_and_chol(self, U, miss_mask, nn_outputs=None):
        """
        :param U: array (n, k)
        :return mean, cov: arrays (n, k)
        """
        if nn_outputs is None:
            nn_outputs = self.nn.fprop(deepcopy(U))[-1]
        mean = deepcopy(nn_outputs[:, :self.dim])
        chol = deepcopy(np.exp(nn_outputs[:, self.dim:]))  # diagonal cholesky of precision
        return mean * miss_mask, chol * miss_mask

    def get_miss_mask(self, Z):
        """ Get 2d array of missing data mask
        :param Z: array (nz, n, k)
        :return: array (n, k)
        """
        miss_mask = np.zeros(Z.shape[1:])
        miss_mask[np.nonzero(Z[0])] = 1
        return miss_mask

    def normal_cdf_inverse(self, x):
        """"Ignores any elements of x equal to 0"""
        with np.errstate(divide='ignore', invalid='ignore'):
            y = norm.ppf(x)
            y[~ np.isfinite(y)] = 0  # -inf inf NaN
        return y

    def checkMask(self, array, mask):
        if array.ndim == 3:
            # mask is 2d, but Z arrays are 3d due to multiple latents per visible
            new_mask = np.ones_like(array)
            new_mask *= mask
        else:
            new_mask = mask

        if not np.all((array != 0) == (new_mask != 0)):
            print('GOTCHA')
        assert np.all((array != 0) == (new_mask != 0)), 'non-zero elements of array are inconsistent with the missing data mask'


class MissingDataProductOfTruncNormNoise(Distribution):

    def __init__(self, mean, chol, rng=None):
        self.mean_len = len(mean)
        if rng:
            self.rng = rng
        else:
            self.rng = rnd.RandomState(DEFAULT_SEED)

        # log to enforce positivity
        chol = np.log(chol)
        alpha = np.concatenate((mean.reshape(-1), chol.reshape(-1)))
        super().__init__(alpha, rng=rng)

    def __call__(self, U, log=False, nn_outputs=None):
        observed_mask = self.get_observed_mask(U)  # (n, k) - 0's represent missing data
        truncation_mask = np.all(U >= 0, axis=-1)  # (nz, n)

        mean, chol = self.get_mean_and_chol(observed_mask)  # (n, k)
        precision = chol**2

        power = -(1/2) * precision * (U - mean)**2  # (nz, n, k)
        log_chol = np.log(chol, out=np.zeros_like(chol), where=chol != 0)
        log_norm_const_1 = (-1/2) * np.log(2 * np.pi) + log_chol  # (n, k) - norm const for usual Gaussian
        log_norm_const_2 = - np.log(1 - norm.cdf(-mean * chol))  # (n, k) - norm const due to truncation
        log_probs = power + log_norm_const_1 + log_norm_const_2  # (nz, n, k)

        log_probs *= observed_mask
        val = np.sum(log_probs, axis=-1)  # (nz, n)
        if log:
            val = truncation_mask * val + (1 - truncation_mask) * -15  # should be -infty, but -15 avoids numerical issues
        else:
            val = truncation_mask * np.exp(val)

        return val  # (nz, n)

    def sample(self, n):
        E = self.sample_E(n)  # (n, k)
        Z = self.get_samples_from_E(E)  # (n, k)
        return Z

    def sample_E(self, n):
        return self.rng.uniform(0, 1, size=(n, self.mean_len))

    def get_samples_from_E(self, E):
        """Converts samples E from some simple base distribution to samples from posterior

        This function is needed to apply the reparametrization trick. E is from a uniform distribution
        and U is from a product of truncated normals whose location & scale depend on a neural network with parameters alpha.

        :param E: array (nz, n, k)
            random variable to be transformed into U (via reparam trick)
        :param U: array (n, k)
        """
        observed_mask = self.get_observed_mask(E)  # all data sampled from noise is observed
        mean, chol = self.get_mean_and_chol(observed_mask)  # (n, k)  - cholesky of diagonal precision matrix
        a = (norm.cdf(-mean * chol) * (1 - E)) + E
        U = norm.ppf(a)  # (n, k)
        U *= (1 / chol)
        U += mean
        return U  # (n, k)

    def get_mean_and_chol(self, observed_mask):
        """
        :param U: array (n, k)
        :return mean, cov: arrays (n, k)
        """
        mean = deepcopy(self.alpha[:self.mean_len])
        chol = deepcopy(np.exp(self.alpha[self.mean_len:]))  # diagonal cholesky of precision
        return mean * observed_mask, chol * observed_mask

    def get_observed_mask(self, U):
        """ Get 2d array of missing data mask
        :param U: array (n, k)
        :return: array (n, k)
        """
        observed_mask = np.zeros_like(U)
        observed_mask[np.nonzero(U)] = 1
        return observed_mask

    def checkMask(self, array, mask):
        assert np.all((array != 0) == (mask != 0)), 'Non-zero elements of array do ' \
                                                    'not match those of the missing data mask'


# noinspection PyPep8Naming,PyMissingConstructor
class GaussianNoise(Distribution):
    def __init__(self, mean=0, cov=1, rng=None):
        if isinstance(mean, float) or isinstance(mean, int):
            mean = np.array([mean])
        if isinstance(cov, float) or isinstance(cov, int):
            cov = np.array([[cov]])
        self.mean = mean
        self.cov = cov
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def __call__(self, U):
        """evaluate probability of data U

        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        return multivariate_normal.pdf(U, self.mean, self.cov)

    def sample(self, num_noise_samples):
        """

        :param num_noise_samples: int
            number of noise samples used in NCE
        :return: array (num_noise_samples, d)
        """
        return self.rng.multivariate_normal(self.mean, self.cov, num_noise_samples)


# noinspection PyPep8Naming,PyMissingConstructor
class MultivariateBernoulliNoise(Distribution):

    def __init__(self, marginals, rng=None):
        """
        :param marginals: array (d,)
            probabilities that each of d binary random variables equal 1
        """
        self.p = marginals
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def __call__(self, U):
        """evaluate probability of binary data U

        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        return np.product(U*self.p + (1-U)*(1-self.p), axis=1)

    def sample(self, num_noise_samples):
        """
        :param num_noise_samples: int
            number of noise samples used in NCE
        :return: array (num_noise_samples, d)
        """
        noise = self.rng.uniform(0, 1, (num_noise_samples, len(self.p))) < self.p
        return noise.astype(int)


# noinspection PyPep8Naming,PyMissingConstructor
class EmpiricalDist(Distribution):

    def __init__(self, X, rng=None):
        """
        :param X: array
            data that defines empirical distribution
        """
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng
        self.X = X
        self.sample_size = len(X)
        self.freqs = self.construct_freqs()

    def __call__(self, U):
        """evaluate probability of binary data U

        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        probs = np.zeros(U.shape[0])
        for i, u in enumerate(U):
            probs[i] = self.freqs.get(str(u), 0)

        return probs

    def sample(self, num_noise_samples):
        """
        :param num_noise_samples: int
            number of noise samples used in NCE
        :return: array (num_noise_samples, d)
        """
        sample_inds = self.rng.randint(0, self.sample_size, num_noise_samples)
        return self.X[sample_inds]

    def construct_freqs(self):
        """Construct dict of binary vector to frequency in data X"""
        # Count number of instances of each unique datapoint
        unique_X = np.unique(self.X, axis=0)
        emp_dist = {str(patch): 0 for patch in unique_X}
        for i, x in enumerate(unique_X):
            emp_dist[str(x)] = np.sum(np.all(self.X == x, axis=1))

        # normalise the counts
        emp_dist = {x: count/self.sample_size for x, count in emp_dist.items()}
        total_prob = np.sum(np.array(list(emp_dist.values())))
        assert np.allclose(total_prob, 1), 'Expected probability empirical distribution to ' \
                                           'sum to 1. Instead it sums to {}'.format(total_prob)

        return emp_dist

    def __repr__(self):
        return "empirical distribution"


# noinspection PyMissingConstructor
class ChowLiuTree(Distribution):

    def __init__(self, X, rng=None):
        """initialise tree structure and estimate the model parameters

        NOTE: this implementation only works for binary vector data. This
        is because I had to write the sample() method from scratch, since
        pomegranate has yet to implement this feature.

        :param X: array (n, d)
            data used to construct tree
        """
        bn = BayesianNetwork()
        self.model = bn.from_samples(X, algorithm='chow-liu')
        self.child_dict, self.parent_dict = self.get_children_and_parent_dicts()
        self.top_order = self.get_topological_ordering()
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def __call__(self, U):
        """
        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        return self.model.probability(U)

    def sample(self, n):
        """sample n datapoints from the distribution

        :param n: int
            number of datapoints to sample
        :return: array (n, )
            data sampled from the distribution
        """
        num_nodes = self.model.state_count()
        sample = np.zeros((n, num_nodes))
        node_to_parent_id = {c.name: int(p.name) for c, p in self.parent_dict.items()}

        for i in range(n):
            for node in self.top_order:
                node_id = int(node.name)
                if node not in self.parent_dict.keys():
                    if 1 in node.distribution.parameters[0]:
                        sample[i, node_id] = self.rng.uniform(0, 1) < node.distribution.parameters[0][1]
                    else:
                        sample[i, node_id] = self.rng.uniform(0, 1) > node.distribution.parameters[0][0]
                else:
                    cpt = np.array(node.distribution.parameters[0])
                    parent_id = node_to_parent_id[node.name]
                    parent_sample_val = sample[i, parent_id]
                    sub_table = cpt[cpt[:, 0] == parent_sample_val]
                    if sub_table[sub_table[:, 1] == 1].size != 0:
                        sample[i, node_id] = self.rng.uniform(0, 1) < sub_table[sub_table[:, 1] == 1][0, -1]
                    else:
                        sample[i, node_id] = self.rng.uniform(0, 1) > sub_table[sub_table[:, 1] == 0][0, -1]

        return sample

    def get_topological_ordering(self):
        """Get topological ordering of variables in graph"""
        topological_order = []
        top_nodes = [v for v in self.model.states if v not in self.parent_dict.keys()]
        current_generation = top_nodes
        keep_going = True
        while keep_going:
            topological_order.extend(current_generation)
            next_generation = []
            for node in current_generation:
                if node in self.child_dict.keys():
                    next_generation.extend(self.child_dict[node])

            current_generation = next_generation
            if not current_generation:
                keep_going = False

        return topological_order

    def get_children_and_parent_dicts(self):
        node_to_children_dict = {}
        node_to_parents_dict = {}
        for e in self.model.edges:
            node_to_children_dict[e[0]] = node_to_children_dict.get(e[0], []) + [e[1]]
            node_to_parents_dict[e[1]] = e[0]

        return node_to_children_dict, node_to_parents_dict


# noinspection PyPep8Naming,PyMissingConstructor
class MissingDataUniformNoise(Distribution):
    def __init__(self, low, high, rng=None):
        self.low = low
        self.high = high
        self.dim = len(low)
        self.vol = np.product(high - low)
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def __call__(self, U, log=False):
        """evaluate probability of data U

        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        V = self.get_missing_mask(U) * self.low  # fill in the missing vals with self.low (an arbitrary choice inside boundaries)
        if log:
            return np.log((1 / self.vol) * self.is_inside_boundaries(U+V))
        else:
            return (1 / self.vol) * self.is_inside_boundaries(U+V)

    def sample(self, num_noise_samples):
        """

        :param num_noise_samples: int
            number of noise samples used in NCE
        :return: array (num_noise_samples, d)
        """
        return self.rng.uniform(self.low, self.high, size=(num_noise_samples, self.dim))

    def is_inside_boundaries(self, X):
        return np.all(X >= self.low, axis=1) * np.all(X <= self.high, axis=1)

    def get_missing_mask(self, U):
        """ Get 2d array where 1s denote missing data
        :param U: array (n, k)
        :return: array (n, k)
        """
        miss_mask = np.zeros_like(U)
        miss_mask[np.where(U == 0)] = 1
        return miss_mask
