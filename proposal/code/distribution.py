"""Module provides classes for probability distributions.
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from itertools import product
from numpy import random as rnd
from plot import *
from scipy.special import erfcx
from scipy.stats import norm, multivariate_normal
from utils import *

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
    def sample(self, nz, U, miss_mask=None):
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

    def __call__(self, Z, U, log=False):
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
        val = (Z == 0)*(1 - p1) + (Z == 1)*p1
        if log:
            val = np.log(val)
        return val

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

    def sample(self, nz, U, miss_mask=None):
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

    def __call__(self, Z, u, log=False):
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
        val = (Z == 0)*(1 - p1) + (Z == 1)*p1
        if log:
            val = np.log(val)
        return val

    def calculate_p(self, U):
        """Return probability binary variable is 1"""
        sigma = np.exp(self.alpha)
        a = norm.pdf(U, 0, sigma)
        b = norm.pdf(U, 0, self.sigma1)
        return b / (a + b)

    def sample(self, nz, U, miss_mask=None):
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

    def sample(self, nz, U, miss_mask=None):
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

    def get_mean_and_cov(self, U=None, outputs=None):
        """
        :param U: array (n, d)
        :return mean, cov: arrays (n, 2)
        """
        if outputs is None:
            if U is None:
                raise ValueError
            else:
                outputs = self.nn.fprop(U)[-1]
        mean, cov = outputs[:, :2], np.exp(outputs[:, 2:])**2
        return mean, cov

    def sample(self, nz, U, nn_outputs=None):
        E = self.sample_E(nz, len(U))  # (nz, n, 2)
        Z = self.get_Z_samples_from_E(E, U, outputs=nn_outputs)  # (nz, n, 2)
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

    def grad_entropy_wrt_nn_outputs(self, outputs):
        grad_shape = outputs.shape  # (n, 4)
        grad = np.zeros(grad_shape)

        grad[:, :2] = 0  # mean of gaussian does not contribute to the entropy
        grad[:, 2:] = 1

        return grad

    def grad_log_wrt_nn_outputs(self, outputs, grad_z_wrt_nn_outputs, Z):
        """
        :param outputs: array (n, 4)
        :param grad_z_wrt_nn_outputs: array (4, nz, n, 2)
        :param Z:  array (nz, n, 2)
        :return: array (4, nz, n)
        """
        mean, cov = self.get_mean_and_cov(outputs=outputs)  # (n, 2)
        grad_log_wrt_out_1 = (Z - mean) / cov  # (nz, n, 2) - out_1 refers to the mean
        grad_log_wrt_z = - grad_log_wrt_out_1
        grad_log_wrt_out_2 = (Z - mean)**2 / cov - 1  # (nz, n, 2) - out_2 refers to log std

        # the gradient is the sum of two components: one component is obtained `indirectly' by differentiating through z using the
        # chain rule. The other component of the gradient is obtained `directly`.
        grad_log_indirect = np.sum(grad_z_wrt_nn_outputs * grad_log_wrt_z, axis=-1)  # (4, nz, n)
        grad_log_wrt_out_1 = np.transpose(grad_log_wrt_out_1, axes=(2, 0, 1))  # (2, nz, n)
        grad_log_wrt_out_2 = np.transpose(grad_log_wrt_out_2, axes=(2, 0, 1))  # (2, nz, n)
        grad_log_direct = np.concatenate((grad_log_wrt_out_1, grad_log_wrt_out_2), axis=0)  # (4, nz, n)

        grad_log_wrt_nn_outputs = grad_log_indirect + grad_log_direct  # (4, nz, n)

        return grad_log_wrt_nn_outputs


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
            mean, std_inv = self.get_pretruncated_params(U * (1 - miss_mask), miss_mask, nn_outputs)  # (n, k)  - cholesky of diagonal precision matrix
            precision = std_inv**2

            power = -(1/2) * precision * (Z - mean)**2  # (nz, n, k)
            log_chol = np.log(std_inv, out=np.zeros_like(std_inv), where=std_inv != 0)
            log_norm_const_1 = (-1/2) * np.log(2 * np.pi) + log_chol  # (n, k) - norm const for usual Gaussian
            log_norm_const_2 = - np.log(1 - norm.cdf(-mean * std_inv))  # (n, k) - norm const due to truncation
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

        mean, chol = self.get_pretruncated_params(U=None, miss_mask=miss_mask, nn_outputs=nn_outputs)  # (n, k)  - cholesky of diagonal precision matrix
        precision = chol**2

        grad_Z_wrt_nn_output1 = grad_z_wrt_nn_outputs[:, :, :self.dim]  # (nz, n, d) - grads wrt mean
        grad_Z_wrt_nn_output2 = grad_z_wrt_nn_outputs[:, :, self.dim:]  # (nz, n, d) - grads wrt to np.log(cholesky)

        grad_log_wrt_z = - precision * (Z - mean)  # (nz, n, d)
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
        mean, chol = self.get_pretruncated_params(U=None, miss_mask=miss_mask, nn_outputs=nn_outputs)  # (n, k)  - cholesky of diagonal precision matrix

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
        mean, std_inv = self.get_pretruncated_params(U=U, miss_mask=miss_mask, nn_outputs=nn_outputs)  # (n, d)  - cholesky of diagonal precision matrix
        one_minus_E = (1 - E) * miss_mask  # (nz, n, d)
        a = (norm.cdf(-mean * std_inv) * one_minus_E) + E  # (nz, n, d)
        Z = self.normal_cdf_inverse(a)  # (nz, n, d)
        Z *= miss_mask  # (nz, n, d)
        Z = np.divide(Z, std_inv, out=np.zeros_like(Z), where=std_inv != 0)
        Z += mean

        # self.checkMask(Z, miss_mask)
        return Z  # (nz, n, d)

    def get_pretruncated_params(self, U, miss_mask, nn_outputs=None):
        """returns mean and 1/std of PRE-TRUNCATED normal
        :param U: array (n, d)
        :return mean, 1/std: arrays (n, d)
        """
        if nn_outputs is None:
            nn_outputs = self.nn.fprop(deepcopy(U))[-1]
        mean = deepcopy(nn_outputs[:, :self.dim])
        std_inv = deepcopy(np.exp(nn_outputs[:, self.dim:]))  # diagonal cholesky of precision
        return mean * miss_mask, std_inv * miss_mask

    def get_truncated_params(self, U, miss_mask, nn_outputs=None):
        """returns mean and variance of TRUNCATED normal
        :param U: array (n, d)
        :return mean, std: arrays (n, d)
        """
        mean, std_inv = self.get_pretruncated_params(U, miss_mask, nn_outputs)
        alpha = -mean * std_inv
        erf_term = np.divide(1, erfcx(alpha / 2**0.5), out=np.zeros_like(alpha), where=erfcx(alpha / 2**0.5) != 0)
        std = np.divide(1, std_inv, out=np.zeros_like(std_inv), where=std_inv != 0)
        var = std**2
        const = (2 / np.pi)**0.5

        trunc_mean = mean + const * erf_term * std
        trunc_var = var * (1 + const * alpha * erf_term - (const * erf_term)**2)
        return trunc_mean, trunc_var

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
        assert np.all((array != 0) == (new_mask != 0)), 'non-zero elements of array are inconsistent with the missing data mask'


class MissingDataLogNormalPosterior(Distribution):

    def __init__(self, mean, precision, rng=None):
        self.mean_len = len(mean)

        # choleksy decomposition of precision with log of diagonal elements (to enforce positivity)
        prec = deepcopy(precision)
        chol = np.linalg.cholesky(prec)
        idiag = np.diag_indices_from(chol)
        chol[idiag] = np.log(chol[idiag])
        chol_flat = chol[np.tril_indices(self.mean_len)]

        alpha = np.concatenate((mean.reshape(-1), chol_flat.reshape(-1)))
        super().__init__(alpha, rng=rng)

    def __call__(self, Z, U, log=False, nn_outputs=None):
        miss_mask = self.get_miss_mask(Z)  # (n, k)
        truncation_mask = np.all(Z >= 0, axis=-1)  # (nz, n)
        if np.sum(miss_mask) == 0:
            # no missing data, so log probabilities are all 0
            vals = np.zeros(Z.shape[:2])
        else:
            missing = get_missing_variables(Z, miss_mask)
            means, precs, _, _ = self.get_conditional_params(U, miss_mask)  # lists of arrays

            vals = np.zeros(Z.shape[:2], dtype='float64')  # (nz, n)
            for i in range(len(means)):
                z = missing[i]  # (nz, k)
                if z.size == 0:
                    continue
                mean = means[i]  # (k, )
                prec = precs[i]  # (k, k)
                trunc = truncation_mask[:, i]  # (nz, )

                y = np.log(z) - mean  # (nz, k)
                yP = np.dot(y, prec)  # (nz, k)
                power = -0.5 * np.sum(yP * y, axis=-1)  # (nz, )
                sign, logdet = np.linalg.slogdet(prec)

                log_norm_const = (-len(mean)/2) * np.log(2 * np.pi) + 0.5 * (sign * logdet)
                jacobian_term = - np.sum(np.log(z), axis=-1)  # (nz, )
                log_probs = power + log_norm_const + jacobian_term  # (nz, )
                if log:
                    val = trunc * log_probs + (1 - trunc) * -15  # should be -infty, but -15 avoids numerical issues
                else:
                    val = trunc * np.exp(log_probs)
                vals[:, i] = val

        return np.array(vals)  # (nz, n)

    def grad_log_wrt_alpha(self, U, E, Z_bar, miss_mask):
        return self._grad_wrt_alpha(U, E, Z_bar, miss_mask, model=False)  # (len(alpha), nz, n)

    def grad_log_model_wrt_alpha(self, U, E, Z_bar, miss_mask):
        return self._grad_wrt_alpha(U, E, Z_bar, miss_mask, model=True)  # (len(alpha), nz, n)

    def _grad_wrt_alpha(self, U, E, Z_bar, miss_mask, model=True):
        """Depending on whether model=True, compute one of two gradient terms required to update alpha.

        When model=True, compute nabla_alpha(log_model)
        Otherwise, compute nabla_alpha(log_var_dist)

        :param U: array (n, d)
            observed data (zeros represent missing vals)
        :param E: list of arrays
            n-length list of arrays of shape (nz, k), where k varies
        :Z_bar: list of arrays
            n-length list of arrays of shape (nz, k), where k varies.
            contains gradients of missing vals w.r.t log_model
        : model: boolean
            if True, returns the grad w.r.t log_model. Else, returns the grad w.r.t log_var_dist
        :returns array (len(alpha), nz, n)
            gradients of log_model w.r.t variational params alpha
        """
        V = deepcopy(U * (1 - miss_mask))  # (n, d)
        joint_mean, joint_prec, joint_chol = self.get_joint_pretruncated_params()
        cond_means, cond_precs, H_matrices, inds_and_means = self.get_conditional_params(U, miss_mask)
        miss_inds, obs_inds, obs, obs_means, miss_mean = inds_and_means

        len_alpha = len(self.alpha)
        nz = E[0].shape[0]
        grad_shape = (len_alpha, nz, V.shape[0])
        grads = np.zeros(grad_shape, dtype='float64')
        for i, vars in enumerate(zip(E, Z_bar, cond_means, cond_precs, H_matrices, obs, obs_means, miss_inds, obs_inds)):
            e, z_bar, mean, prec, H, o, o_mean, i_miss, i_obs = vars  # e & z_bar have shape (nz, k).
            if e.size == 0:
                continue
            nz, k = e.shape

            prec_chol = np.linalg.cholesky(prec)  # (k, k)
            prec_chol_T = prec_chol.T  # (k, k)
            cov_chol_T = np.linalg.inv(prec_chol)  # (k, k)
            cov_chol = cov_chol_T.T  # (k, k)
            cov = np.dot(cov_chol, cov_chol_T)  # (k, k)

            if model:
                W_bar = np.exp(np.dot(cov_chol, e.T).T + mean) * z_bar  # (nz, k)
            else:
                W_bar = - np.ones_like(z_bar, dtype='float64')
            L_bar = np.einsum('ij,ik->ijk', W_bar, e)  # outer product. (nz, k, k)
            L_bar_T = np.transpose(L_bar, (0, 2, 1))
            A = -dot_2d_with_3d(cov_chol, L_bar_T)  # (nz, k, k)
            prec_chol_bar = np.dot(A, cov_chol)

            A = dot_2d_with_3d(prec_chol_T, prec_chol_bar)  # (nz, k, k)
            A = get_lower_tri_halving_diag(A)  # (nz, k, k)
            B = np.dot(A, cov_chol_T)  # (nz, k, k)
            R = dot_2d_with_3d(cov_chol, B)  # (nz, k, k)
            prec_bar_1 = R

            D_bar = - np.einsum('ij,k->ijk', W_bar, o - o_mean)  # outer product. (nz, k, d-k)
            cov_bar_2 = np.dot(D_bar, H.T)
            E = np.dot(cov_bar_2, cov)  # (nz, k, k)
            prec_bar_2 = - dot_2d_with_3d(cov, E)  # (nz, k, k)

            prec_bar = prec_bar_1 + prec_bar_2  # (nz, k, k)
            if not model:
                prec_bar += 0.5 * cov

            S1_bar = reshape_condprec_to_prec_shape(prec_bar, i_miss, nz, self.mean_len)  # (nz, d, d)

            H_bar = dot_2d_with_3d(cov.T, D_bar)  # (nz, k, d-k)
            S2_bar = reshape_H_to_prec_shape(H_bar, i_miss, i_obs, nz, self.mean_len)  # (nz, d, d)

            joint_prec_bar = S1_bar + S2_bar  # (nz, d, d)
            joint_prec_bar_T = np.transpose(joint_prec_bar, (0, 2, 1))
            joint_prec_bar = joint_prec_bar + joint_prec_bar_T
            joint_chol_bar = np.dot(joint_prec_bar, joint_chol)  # (nz, d, d)
            i_diag = np.diag_indices_from(joint_chol)
            joint_chol_bar[:, i_diag[0], i_diag[1]] *= joint_chol[i_diag]  # since our parametrisation of the diag is in log-domain

            tril =  np.tril_indices(self.mean_len)
            alpha1_bar = joint_chol_bar[:, tril[0], tril[1]]  # (nz, -1)
            alpha1_bar = alpha1_bar.T  # (-1, nz)

            mean_bar = np.zeros((nz, self.mean_len), dtype='float64')
            mean_bar[:, i_miss] = W_bar  # (nz, d)
            mean_bar = mean_bar.T  # (d, nz)

            G = np.dot(H.T, cov)  # (d-k, k)
            o_mean_bar = np.dot(G, W_bar.T)  # (d-k, nz)
            mean_bar[i_obs, :] = o_mean_bar  # (d, nz)

            alpha_bar = np.concatenate((mean_bar, alpha1_bar), axis=0)  # (len(alpha), nz)
            grads[:, :, i] = alpha_bar

        return grads  # (len(alpha), nz, n)

    def grad_of_Z_wrt_nn_outputs(self, nn_outputs, E):
        """gradient of Z w.r.t to the outputs of the neural network

        :param outputs: array (n, len(nn_outputs))
            outputs of neural net
        :param E: array (nz, n, self.dim)
            random variable to be transformed into Z (via reparam trick)
        :return: array (nz, n, len(nn_outputs))
            grad of each z_i w.r.t neural net output
        """
        raise NotImplementedError

    def sample(self, nz, U, miss_mask=None, nn_outputs=None):
        if miss_mask is None:
            Z = np.zeros((nz, ) + U.shape)
        else:
            E = self.sample_E(nz, miss_mask)  # (nz, n, 2)
            Z = self.get_Z_samples_from_E(nz, E, U, miss_mask, nn_outputs=nn_outputs)  # (nz, n, 2)
        return Z  # n-long list of (nz, k) arrays (note: k varies).

    def sample_E(self, nz, miss_mask):
        miss_row_i, miss_col_i = np.nonzero(miss_mask)
        missing_indices = group_cols_by_row(miss_row_i, miss_col_i, miss_mask.shape[0])  # list of missing dims for each datapoint
        return [self.rng.randn(nz, len(m)) for m in missing_indices]

    def get_Z_samples_from_E(self, nz, E, U, miss_mask, nn_outputs=None):
        """Converts samples E from some simple base distribution to samples from posterior

        This function is needed to apply the reparametrization trick. E is from a uniform distribution
        and Z is from a product of truncated normals whose location & scale depend on a neural network with parameters alpha.
        """
        means, precs, _, _ = self.get_conditional_params(U, miss_mask)  # lists of arrays
        Z = np.tile(miss_mask, (nz, 1, 1)) * 1.
        Z = np.transpose(Z, (1, 0, 2))  # (n, nz, d)
        for i, triple in enumerate(zip(E, means, precs)):
            e, mean, prec = triple
            if e.size == 0:
                continue

            chol = np.linalg.cholesky(prec)
            chol_T = chol.T
            z_minus_mean = np.linalg.solve(chol_T, e.T)  # (k, nz)
            z = np.exp(z_minus_mean.T + mean)  # (nz, k)
            miss_inds = np.nonzero(Z[i, 0, :])  # k-length tuples
            Z[i, :, :][:, miss_inds[0]] *= z

        Z = np.transpose(Z, (1, 0, 2))  # (nz, n, d)
        return Z  # (nz, n, d)

    def get_conditional_params(self, U, miss_mask):
        """Get the means and variances of the conditional normal distributions over latent dimensions
        :return:
            cond_means: list of arrays, each of shape (num_missing,)
            cond_vars: list of arrays, each of shape (num_missing, num_missing)
            H_matrices: list of arrays, of shape (num_missing, num_observed
        """
        V = deepcopy(U * (1 - miss_mask))  # (n, d)
        mean, precision, chol = self.get_joint_pretruncated_params()

        cond_means = []
        cond_vars = []

        # for each datapoint, get indices of the missing dimensions and observed dimensions
        # get vector of observed data and correpsonding means for each data point
        inds_and_means = self.split_data_and_means(miss_mask, V, mean)
        miss_inds, obs_inds, observed, obs_means, miss_means = inds_and_means

        # get precision of conditional for each data point
        cond_precs = get_conditional_precisions(precision, miss_inds)  # list of cond precisions

        # get the H matrices, which are the blocks of the precision associated to the cross-terms between observed and
        # unobserved dimensions (page 2 of https://www.apps.stat.vt.edu/leman/VTCourses/Precision.pdf)
        H_matrices = get_conditional_H(precision, miss_inds, obs_inds)
        cond_means = []
        for obs, o_mean, m_mean, prec, H in zip(observed, obs_means, miss_means, cond_precs, H_matrices):
            if m_mean.size == 0:
                cond_means.append(np.array([]))
            else:
                prec_inv =  np.linalg.inv(prec)  #todo: is there a better method? lingalg solve?
                A =  - np.dot(prec_inv, H)  # (num_missing, num_observed)
                cond_mean = m_mean + np.dot(A, obs - o_mean)  # (num_missing, )
                cond_means.append(cond_mean)

        return cond_means, cond_precs, H_matrices, inds_and_means

    def split_data_and_means(self, miss_mask, V, mean):
        # for each datapoint, get indices of the missing dimensions and observed dimensions
        miss_inds, obs_inds = get_missing_and_observed_indices(miss_mask, miss_mask.shape[0])

        # get vector of observed data and correpsonding means for each data point
        observed = [v[obs_inds] for v, obs_inds in zip(V, obs_inds)]
        obs_means = [mean[obs_inds] for obs_inds in obs_inds]
        miss_means = [mean[miss_inds] for miss_inds in miss_inds]

        return miss_inds, obs_inds, observed, obs_means, miss_means

    def get_joint_pretruncated_params(self, mean=None, lprec=None):
        """Get mean and precision of the log dist (which is a joint multivariate normal)"""
        if mean is None:
            # mean, lprec = deepcopy(self.alpha[:self.mean_len]), deepcopy(self.alpha[self.mean_len:])
            mean, chol_flat = deepcopy(self.alpha[:self.mean_len]), deepcopy(self.alpha[self.mean_len:])

        # parametrised in terms of cholesky decomposition of precision (with diagonal in log-domain)
        d = len(mean)
        chol = np.zeros((d, d))
        chol[np.tril_indices(d)] = chol_flat

        # exp the diagonal, to enforce positivity
        chol[np.diag_indices(d)] = np.exp(chol[np.diag_indices(d)])
        precision = np.dot(chol, chol.T)

        return mean, precision, chol

    def get_miss_mask(self, Z):
        """ Get 2d array of missing data mask
        :param Z: array (nz, n, k)
        :return: array (n, k)
        """
        miss_mask = np.zeros(Z.shape[1:])
        miss_mask[np.nonzero(Z[0])] = 1
        return miss_mask

    def checkMask(self, array, mask):
        if array.ndim == 3:
            # mask is 2d, but Z arrays are 3d due to multiple latents per visible
            new_mask = np.ones_like(array)
            new_mask *= mask
        else:
            new_mask = mask
        assert np.all(
            (array != 0) == (new_mask != 0)), 'non-zero elements of array are inconsistent with the missing data mask'


class UnivariateTruncNormTruePosteriors(Distribution):

    def __init__(self, mean, precision, rng=None):
        self.dim = len(mean)

        # lower triangular part of precision with log of diagonal elements (to enforce positivity)
        prec = deepcopy(precision)
        idiag = np.diag_indices_from(prec)
        prec[idiag] = np.log(prec[idiag])
        lower_prec = prec[np.tril_indices(self.dim)]

        # need an arbitrary, extra parameter (hence 99999)
        alpha = np.concatenate((np.array([99999]), mean.reshape(-1), lower_prec.reshape(-1)))
        super().__init__(alpha, rng=rng)

    def __call__(self, Z, U, log=False):
        miss_mask = self.get_miss_mask(Z)  # (n, k)
        collapsed_mask = miss_mask.sum(1)
        assert np.alltrue(collapsed_mask <= 1), 'expected at most 1 missing value per data point'

        Z_collapsed = np.sum(Z, axis=2)  # (nz, n)
        truncation_mask = Z_collapsed >= 0  # (nz, n)
        if np.sum(miss_mask) == 0:
            # no missing data, so log probabilities are all 0
            log_probs = np.zeros(Z_collapsed.shape)
        else:
            mean, var = self.get_conditional_pretruncated_params(U, miss_mask)
            std = var**0.5
            precision = np.divide(1, var, out=np.zeros_like(var), where=var != 0)
            std_inv = precision**0.5

            power = -(1/2) * precision * (Z_collapsed - mean)**2  # (nz, n)
            log_std = np.log(std, out=np.zeros_like(std), where=std != 0)
            log_norm_const_1 = (-1/2) * np.log(2 * np.pi) - log_std  # (n) - norm const for usual Gaussian
            log_norm_const_2 = - np.log(1 - norm.cdf(-mean * std_inv))  # (n) - norm const due to truncation
            log_probs = power + log_norm_const_1 + log_norm_const_2  # (nz, n)

        if log:
            val = truncation_mask * log_probs + (1 - truncation_mask) * -15  # should be -infty, but -15 avoids numerical issues
        else:
            val = truncation_mask * np.exp(log_probs)

        return val  # (nz, n)

    def sample(self, nz, U, miss_mask=None, nn_outputs=None):
        if miss_mask is None:
            Z = np.zeros((nz, ) + U.shape)
        else:
            E = self.sample_E(nz, miss_mask)  # (nz, n, 2)
            Z = self.get_Z_samples_from_E(nz, E, U, miss_mask)  # (nz, n, 2)
        return Z

    def sample_E(self, nz, miss_mask):
        return self.rng.uniform(0, 1, size=(nz, miss_mask.shape[0]))

    def get_Z_samples_from_E(self, nz, E, U, miss_mask):
        """Converts samples E from some simple base distribution to samples from posterior

        :param E: array (nz, n, d)
            random variable to be transformed into Z (via reparam trick)
        :param U: array (n, d)
        """
        mean, var = self.get_conditional_pretruncated_params(U, miss_mask)
        collapsed_mask = np.any(miss_mask != 0, axis=-1)
        std = var**0.5
        std_inv = np.divide(1, std, out=np.zeros_like(std), where=std != 0)

        one_minus_E = (1 - E) * collapsed_mask  # (nz, n)
        a = (norm.cdf(-mean * std_inv) * one_minus_E) + E  # (nz, n, d)
        a *= collapsed_mask
        Z = self.normal_cdf_inverse(a)  # (nz, n)
        Z *= collapsed_mask
        Z *= std
        Z += mean  # (nz, n)

        Z_uncollapsed = np.tile(miss_mask, (nz, 1)).reshape((nz,) + miss_mask.shape)
        Z_uncollapsed[np.nonzero(Z_uncollapsed)] = Z[np.nonzero(Z)]
        self.checkMask(Z_uncollapsed, miss_mask)

        return Z_uncollapsed  # (nz, n, d)

    def get_joint_pretruncated_params(self):
        """Get mean and precision of the joint multivariate normal"""
        mean, lprec = deepcopy(self.alpha[1:1+self.dim]), deepcopy(self.alpha[1+self.dim:])

        # lower-triangular cholesky decomposition of the precision
        lprecision = np.zeros((self.dim, self.dim))
        ilower = np.tril_indices(self.dim)
        lprecision[ilower] = lprec

        # exp the diagonal, to enforce positivity
        idiag = np.diag_indices_from(lprecision)
        lprecision[idiag] = np.exp(lprecision[idiag])

        precision = lprecision + lprecision.T - np.diag(np.diag(lprecision))

        return mean, precision, lprecision, precision[idiag]

    def get_conditional_pretruncated_params(self, U, miss_mask):
        """Get the means and variances of the d univariate conditional normals
        :return:
        """
        V = deepcopy(U * (1 - miss_mask))  # (n, d)
        mean, precision, lprec, prec_diag = self.get_joint_pretruncated_params()

        cond_mean, cond_var = np.zeros(U.shape[0]), np.zeros(U.shape[0])
        # get row and column indices of missing values
        row_i, col_i = np.nonzero(miss_mask)
        # get variance of the conditional for each data point
        cond_var[row_i] = 1 / prec_diag[col_i]
        # get mean of of the conditional for each data point
        K = np.zeros_like(V)
        K[row_i] = precision[col_i]
        diff = mean - V
        cond_mean = cond_var * np.sum(K * diff, axis=1)

        return cond_mean, cond_var

    def get_truncated_params(self, miss_mask):
        """returns mean and variance of TRUNCATED normal
        :param U: array (n, d)
        :return mean, std: arrays (n, d)
        """
        raise NotImplementedError
        # alpha = -mean * std_inv
        # erf_term = np.divide(1, erfcx(alpha / 2**0.5), out=np.zeros_like(alpha), where=erfcx(alpha / 2**0.5) != 0)
        # std = np.divide(1, std_inv, out=np.zeros_like(std_inv), where=std_inv != 0)
        # var = std**2
        # const = (2 / np.pi)**0.5
        #
        # trunc_mean = mean + const * erf_term * std
        # trunc_var = var * (1 + const * alpha * erf_term - (const * erf_term)**2)
        # return trunc_mean, trunc_var

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

    def __call__(self, U, Z=None, log=False, nn_outputs=None):
        if Z is not None:
            observed_mask = self.get_observed_mask(Z)  # (n, k) - 0's represent missing data
        else:
            observed_mask = np.ones_like(U)
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

    def sample(self, n, miss_mask=None):
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
        observed_mask = np.ones_like(E)  # all data sampled from noise is observed
        mean, chol = self.get_mean_and_chol(observed_mask)  # (n, k)  - cholesky of diagonal precision matrix
        a = (norm.cdf(-mean * chol) * (1 - E)) + E
        U = norm.ppf(a)  # (n, k)
        U *= (1 / chol)
        U += mean
        return U  # (n, k)

    def get_mean_and_chol(self, observed_mask):
        """
        :param U: array (n, d)
        :return mean, cov: arrays (n, d)
        """
        mean = deepcopy(self.alpha[:self.mean_len])
        chol = deepcopy(np.exp(self.alpha[self.mean_len:]))  # diagonal cholesky of precision
        return mean * observed_mask, chol * observed_mask

    def get_observed_mask(self, Z):
        """ Get 2d array of missing data mask
        :param U: array (n, k)
        :return: array (n, k)
        """
        observed_mask = np.zeros(Z.shape[1:])
        observed_mask[np.where(Z[0] == 0)] = 1
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

    def __call__(self, U, log=False):
        """evaluate probability of data U

        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        val = multivariate_normal.pdf(U, self.mean, self.cov)
        if log:
            val = np.log(val)
        return val

    def sample(self, num_noise_samples, miss_mask=None):
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

    def __call__(self, U, log=False):
        """evaluate probability of binary data U

        :param U: U: array (n, d)
            n data points with dimension d
        :return array of shape (n, )
            probability of each input datapoint
        """
        log_val = np.sum(np.log(U*self.p + (1-U)*(1-self.p)), axis=1)
        if log:
            return log_val
        else:
            return np.exp(log_val)

    def sample(self, num_noise_samples, miss_mask=None):
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
        from pomegranate import BayesianNetwork
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
        mask = self.get_missing_mask(U)
        V = mask * self.low  # fill in the missing vals with self.low (an arbitrary choice inside boundaries)
        in_boundaries = self.is_inside_boundaries(U+V)
        log_vols = np.sum((1 - mask) * np.log((self.high - self.low)), axis=-1)
        val = log_vols * in_boundaries - (15 * (1 - in_boundaries))
        if not log:
            val = np.exp(val)
        return val

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


# noinspection PyPep8Naming,PyMissingConstructor
class LearnedVariationalNoise(Distribution):
    def __init__(self, var_dist, rng=None):
        self.var_dist = var_dist
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
        d = U.shape[1]
        vals = np.zeros((1, ) + U.shape)  # (n, d)
        for i in range(d):
            V = deepcopy(U)
            V[:, i] = 0
            Z = np.zeros_like(vals)
            Z[0, :, i] = U[:, i]
            vals[0, :, i] = self.var_dist(Z, V, log=True)

        val = np.sum(vals, axis=-1)
        if not log:
            val = np.exp(val)
        return val

    # def __call__(self, U, log=False):
    #     """evaluate probability of data U
    #
    #     :param U: U: array (n, d)
    #         n data points with dimension d
    #     :return array of shape (n, )
    #         probability of each input datapoint
    #     """
    #     d = U.shape[1]
    #     vals = np.zeros((1,) + U.shape)  # (n, d)
    #     V = deepcopy(U)
    #     Z = np.zeros_like(vals)
    #
    #     for i in range(d):
    #         V[:, i] = 0
    #         Z[0, :, i] = U[:, i]
    #         vals[0, :, i] = self.var_dist(Z, V, log=True)
    #
    #     val = np.sum(vals)
    #     if not log:
    #         val = np.exp(val)
    #     return val

    def sample(self, U, num_gibbs_steps):
        """

        :return: array (num_noise_samples, d)
        """
        V = deepcopy(U)
        miss_mask = np.zeros_like(U)
        for _ in range(num_gibbs_steps):
            for i in range(U.shape[1]):
                miss_mask *= 0
                miss_mask[:, i] = 1
                W = V * (1 - miss_mask)
                Z = self.var_dist.sample(nz=1, U=W, miss_mask=miss_mask)
                V = Z[0, :, :] + W

        return V
