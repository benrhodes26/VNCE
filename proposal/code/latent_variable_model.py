"""Module provides classes for unnormalised latent-variable models
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from itertools import product
from numpy import random as rnd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity as kd
from utils import validate_shape, sigmoid
# noinspection PyPep8Naming
DEFAULT_SEED = 1083463236

# noinspection PyPep8Naming
class LatentVarModel(metaclass=ABCMeta):

    def __init__(self, theta, rng=None):
        """ Initialise parameters of model: theta.

        Theta should be a one-dimensional array or int/float"""
        if isinstance(theta, float) or isinstance(theta, int):
            theta = np.array([theta])
        assert theta.ndim == 1, 'Theta should have dimension 1, ' \
                                'not {}'.format(theta.ndim)
        self._theta = theta
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
        assert value.ndim == 1, 'Theta should have 1 dimension, ' \
                                'not {}'.format(value.ndim)
        assert value.shape == self.theta_shape, 'Tried to ' \
            'set theta to array with shape {}, should be {}'.format(value.shape,
                                                                    self.theta_shape)
        self._theta = value

    @abstractmethod
    def __call__(self, U, Z):
        """Evaluate model on data

        :param U: array (n, d)
            n input data points to evaluate model on
        :param Z: (nz, n, m)
            m dimensional latent variable samples for data U.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_log_wrt_params(self, U, Z):
        """Returns nabla_theta(log(phi(x, theta)))

        :param U: array (n, d)
            data sample we are training our model on
        :param Z: (nz, n, m)
            nz*n 1-dimensional latent variable samples for data U.
        """
        raise NotImplementedError


# noinspection PyPep8Naming,PyMissingConstructor,PyTypeChecker,PyArgumentList
class LatentMixtureOfTwoGaussians(LatentVarModel):
    """ *normalised* Mixture of two Gaussians.

    The model takes the form:
    phi(u; theta) = (1/2)*Z*N(u; 0, sigma(theta)) + (1/2)*(1-Z)N(u; 0, sigma1)
    where Z is a binary latent variable and sigma = exp(theta)
    """
    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
            (log) standard deviation of one of the two gaussians. Note that
            for optimisation we use the log, to enforce positivity.
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U, Z, log=False):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :param Z: (nz, n, 1)
            nz*n 1-dimensional latent variable samples for data U.
        :return array (nz, n)
        """
        Z = Z.reshape(Z.shape[0], Z.shape[1])

        first_term = (Z == 0)*self.marginal_z_0(U)
        second_term = (Z == 1)*self.marginal_z_1(U)

        if log:
            val = np.log(first_term + second_term)
        else:
            val = first_term + second_term

        return val

    def marginal_z_0(self, U):
        """ Return value of model for z=0
        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n,)
        """
        U = U.reshape(-1)
        sigma = np.exp(self.theta)
        return 0.5*norm.pdf(U.reshape(-1), 0, sigma)

    def marginal_z_1(self, U):
        """ Return value of model for z=1
        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n,)
        """
        U = U.reshape(-1)
        return 0.5*norm.pdf(U.reshape(-1), 0, self.sigma1)

    def marginalised_over_z(self, U):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param U: array (n, 1)
            N is number of data points
        :return array (n, )
            probabilities of datapoints under p(x) i.e with z marginalised out
            and the distribution has been normalised
        """
        return self.marginal_z_0(U) + self.marginal_z_1(U)

    def grad_log_wrt_params(self, U, Z):
        """ nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :param Z: array (nz, n, 1)
        :return grad: array (len(theta), nz, n)
        """
        nz, n = Z.shape[0], Z.shape[1]
        Z = Z.reshape(nz, n)

        a = (U**2 * np.exp(-2*self.theta)) - 1  # (n, 1)
        grad = (Z == 0) * a.reshape(-1)
        grad = grad.reshape(1, nz, n)  # (len(theta), nz, n)

        correct_shape = self.theta_shape + (nz, n)
        assert grad.shape == correct_shape, ' ' \
            'gradient should have shape {}, got {} instead'.format(correct_shape,
                                                                   grad.shape)
        return grad

    def grad_log_wrt_params_analytic(self, U):
        """ nabla_theta(log(model))
        :param U: array (n, 1)
            either data or noise for NCE
        :return array (1, n)
        """
        grad0 = (U**2 * np.exp(-2*self.theta)) - 1  # (n, 1)
        grad1 = 0
        return grad0.T, grad1  # (1, n)

    def sample(self, n):
        """ Sample n values from the model
        :param n: number of data points to sample
        :return: data array of shape (n, )
        """
        sigma = np.exp(self.theta)
        w = self.rng.uniform(0, 1, n) > 0.5
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
        fig = plt.figure(figsize=figsize)
        u = np.arange(-10, 10, 0.01)

        x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        plt.plot(u, x_density_samples, label='kde')

        px = self.marginalised_over_z(u)
        plt.plot(u, px, label='true', c='r', linestyle='--')

        plt.legend()
        plt.grid()


# noinspection PyPep8Naming,PyMissingConstructor,PyTypeChecker,PyArgumentList
class LatentMixtureOfTwoUnnormalisedGaussians(LatentVarModel):
    """ Mixture of two *unnormalised* Gaussians given by:
    phi(u; theta) = e^-(theta[0]) [exp(-x**2/2exp(theta[1])**2) +
                                   exp(-x**/2sigma1**2)]

    This class represents a sum of two gaussians, each of which has
    lost its normalising constant.
    """

    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise theta and stdevs of gaussians
        :param theta: array (2, )
            theta[0] is a scaling parameter. See the formula
            in the class docstring. theta[1] is (log) standard deviation
            of one of the two gaussians. We use the log to enforce positivity.
        :param sigma1: float/int
            standard deviation of one of the gaussians
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U, Z):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :param Z: (nz, n, 1)
            nz*n 1-dimensional latent variable samples for data U.
        :return array (nz, n)
        """
        Z = Z.reshape(Z.shape[0], Z.shape[1])

        first_term = (Z == 0)*self.marginal_z_0(U)  # (nz, n)
        second_term = (Z == 1)*self.marginal_z_1(U)  # (nz, n)

        return first_term + second_term

    def marginal_z_0(self, U):
        """ Return value of model for z=0
        :param U: array (n, 1)
            N is number of data points
        :return array (n,)
        """
        U = U.reshape(-1)
        a = np.exp(-self.theta[0])  # scaling parameter
        sigma = np.exp(self.theta[1])  # stdev parameter
        b = np.exp(-U**2 / (2*(sigma**2)))
        return a * b

    def marginal_z_1(self, U):
        """ Return value of model for z=1
        :param U: array (n, 1)
            N is number of data points
        :return array (n,)
        """
        U = U.reshape(-1)
        a = np.exp(-self.theta[0])  # scaling parameter
        b = np.exp(-U**2 / (2*(self.sigma1**2)))
        return a * b

    def grad_log_wrt_params(self, U, Z):
        """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :param Z: (nz, n, 1)
            1 dimensional latent variable samples for data U.
        :return grad: array (len(theta)=2, nz, n)
        """
        nz, n = Z.shape[0], Z.shape[1]
        Z = Z.reshape(nz, n)
        grad = np.zeros((len(self.theta), nz, n))

        grad[0] = -1  # grad w.r.t scaling param
        a = (U**2 * np.exp(-2*self.theta[1]))  # (n, 1)
        grad[1] = (Z == 0) * a.reshape(-1)  # (nz, n)

        correct_shape = self.theta_shape + (nz, n)
        assert grad.shape == correct_shape, ' ' \
            'gradient should have shape {}, got {} instead'.format(correct_shape,
                                                                   grad.shape)
        return grad

    def grad_log_wrt_params_analytic(self, U):
        """ nabla_theta[1] (log(model))
        :param U: array (n, 1)
            either data or noise for NCE
        :return array (2, n)
        """
        grad0 = np.zeros((self.theta.size, U.shape[0]))  # (2, n)
        grad1 = np.zeros((self.theta.size, U.shape[0]))  # (2, n)

        grad0[0] = -1
        grad1[0] = -1

        a = (U**2 * np.exp(-2*self.theta[1]))  # (n, 1)
        grad0[1] = a.reshape(-1)
        grad1[1] = 0

        return grad0, grad1  # (2, n)

    def sample(self, n):
        """ Sample n values from the model, with uniform(0,1) dist over latent vars

        :param n: number of data points to sample
        :return: data array of shape (n, 1)
        """
        sigma = np.exp(self.theta[1])
        a = self.sigma1 / (sigma + self.sigma1)
        w = self.rng.uniform(0, 1, n) < a
        x = (w == 0)*(self.rng.randn(n)*sigma) + (w == 1)*(self.rng.randn(n)*self.sigma1)
        return x.reshape(-1, 1)

    def normalised_and_marginalised_over_z(self, U):
        """Return values of p(U), where p is the (normalised) marginal over x

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
        fig = plt.figure(figsize=figsize)
        u = np.arange(-10, 10, 0.01)

        x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        plt.plot(u, x_density_samples, label='kde')

        px = self.normalised_and_marginalised_over_z(u)
        plt.plot(u, px, label='true', c='r', linestyle='--')

        plt.legend()
        plt.grid()

        return fig


# noinspection PyPep8Naming,PyMissingConstructor,PyTypeChecker,PyArgumentList
class LatentMixtureOfTwoUnnormalisedGaussians2(LatentVarModel):
    """ Mixture of two *unnormalised* Gaussians given by:
    phi(u, z; theta) =  [ (1-z) * e^-(theta[0])* exp(-x**2/2exp(theta[2])**2) + z * e^-(theta[1])* exp(-x**/2sigma1**2)]

    This class represents a sum of two gaussians, each of which has lost its normalising constant.
    """

    def __init__(self, theta, sigma1=1, rng=None):
        """Initialise theta and stdevs of gaussians
        :param theta: array (3, )
            theta[0] and theta[1] are scaling parameters. See the formula
            in the class docstring. theta[2] is (log) standard deviation
            of one of the two gaussians. We use the log to enforce positivity.
        :param sigma1: float/int
            standard deviation of one of the gaussians
        """
        self.sigma1 = sigma1
        super().__init__(theta, rng=rng)

    def __call__(self, U, Z):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 1)
             either data or noise for NCE
        :param Z: (nz, n, 1)
            nz*n 1-dimensional latent variable samples for data U.
        :return array (nz, n)
        """
        Z = Z.reshape(Z.shape[0], Z.shape[1])

        first_term = (Z == 0)*self.marginal_z_0(U)  # (nz, n)
        second_term = (Z == 1)*self.marginal_z_1(U)  # (nz, n)

        return first_term + second_term

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

    def grad_log_wrt_params(self, U, Z):
        """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, 1)
             either data or noise for NCE
        :param Z: (nz, n, 1)
            1 dimensional latent variable samples for data U.
        :return grad: array (len(theta)=3, nz, n)
        """
        nz, n = Z.shape[0], Z.shape[1]
        Z = Z.reshape(nz, n)
        grad = np.zeros((len(self.theta), nz, n))

        first_term = (Z == 0)*self.marginal_z_0(U)  # (nz, n)
        second_term = (Z == 1)*self.marginal_z_1(U)  # (nz, n)

        grad[0] = - first_term / (first_term + second_term)
        grad[1] = - second_term / (first_term + second_term)
        grad[2] = first_term * (U**2 * np.exp(-2 * self.theta[2])) / (first_term + second_term)

        correct_shape = self.theta_shape + (nz, n)
        assert grad.shape == correct_shape, ' ' \
            'gradient should have shape {}, got {} instead'.format(correct_shape, grad.shape)

        return grad

    def grad_log_wrt_params_analytic(self, U):
        """ nabla_theta[1] (log(model))
        :param U: array (n, 1)
            either data or noise for NCE
        :return array (3, n)
        """
        grad0 = np.zeros((self.theta.size, U.shape[0]))  # (3, n)
        grad1 = np.zeros((self.theta.size, U.shape[0]))  # (3, n)

        grad0[0] = -1
        grad0[1] = 0
        a = (U**2 * np.exp(-2*self.theta[2]))  # (n, 1)
        grad0[2] = a.reshape(-1)

        grad1[0] = 0
        grad1[1] = -1
        grad1[2] = 0

        return grad0, grad1  # (3, n)

    def sample(self, n):
        """ Sample n values from the model, with uniform(0,1) dist over latent vars

        :param n: number of data points to sample
        :return: data array of shape (n, 1)
        """
        # sigma = np.exp(self.theta[2])
        # a = self.sigma1 / (sigma + self.sigma1)
        # w = self.rng.uniform(0, 1, n) < a
        # x = (w == 0)*(self.rng.randn(n)*sigma) + (w == 1)*(self.rng.randn(n)*self.sigma1)
        # return x.reshape(-1, 1)
        raise NotImplementedError

    def normalised_and_marginalised_over_z(self, U):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n, )
            probabilities of datapoints under p(x) i.e with z marginalised out
            and the distribution has been normalised
        """
        # sigma = np.exp(self.theta[1])
        # a = sigma / (sigma + self.sigma1)
        # return a*norm.pdf(U.reshape(-1), 0, sigma) + (1 - a)*norm.pdf(U.reshape(-1), 0, self.sigma1)
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
        # fig = plt.figure(figsize=figsize)
        # u = np.arange(-10, 10, 0.01)
        #
        # x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        # x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        # plt.plot(u, x_density_samples, label='kde')
        #
        # px = self.normalised_and_marginalised_over_z(u)
        # plt.plot(u, px, label='true', c='r', linestyle='--')
        #
        # plt.legend()
        # plt.grid()
        #
        # return fig
        raise NotImplementedError


# noinspection PyPep8Naming,PyMissingConstructor,PyArgumentList,PyTypeChecker
class RestrictedBoltzmannMachine(LatentVarModel):

    """ Type of probabilistic graphical model that specifies an unnormalised
    joint distribution over data and latent variables. The model is given by:
             phi(u, z; W, a, b) = exp(uWz + au + bz)
    where
        - u, a are d-dimensional vectors
        - z, b are m-dimensional vectors
        - W is a (d, m) matrix
    a, b and W are all weights, u is the visible data and z is the latent vector.
    """

    def __init__(self, W, rng=None):
        """Initialise the parameters of the RBM.

        Typically, the formula of the RMB is given as:
            phi(u, z, W, a, b) = exp(uWz + au + bz)
        Here, we prepend u and z with a bias term (i.e '1')
        and enlarge W so that we can rewrite the 3 terms as 1:
            phi(u, z, W) = exp(uWz)
        Note that this gives us an extra W_11 term, which is useful,
        since it can serve as a scaling parameter for NCE!

        :param W: array (d+1, m+1)
            Weight matrix. d=num_visibles, m=num_hiddens
        """
        self.W_shape = W.shape  # (d+1, m+1)
        self.norm_const = None
        theta = W.reshape(-1)
        super().__init__(theta, rng=rng)

    def __call__(self, U, Z, normalise=False, reset_norm_const=True, log=False):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, d)
             either data or noise for NCE
        :param Z: (nz, n, m) or (n, m)
            nz*n m-dimensional latent variable samples for data U.
        :param normalise: bool
            if true, first normalise the distribution.
        :param reset_norm_const: bool
            if True, recalculate the normalisation constant using current
            parameters. If false, use already saved norm const. Note: this
            argument only matters if normalise=True.
        :return array (nz, n)
            probability of each datapoint & its corresponding latent under
            the joint distribution of the model
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)

        U, Z = self.add_bias_terms(U), self.add_bias_terms(Z)  # (n, d+1), (nz, n, m+1)

        uW = np.dot(U, W)  # (n, m+1)
        uWz = np.sum(Z * uW, axis=-1)  # (nz, n)

        if log:
            val = uWz
        else:
            val = np.exp(uWz)  # (nz, n)
        if normalise:
            if (not self.norm_const) or reset_norm_const:
                self.reset_norm_const()
            val /= self.norm_const

        validate_shape(val.shape, Z.shape[:-1])
        return val

    def add_bias_terms(self, V):
        """prepend 1s along final dimension
        :param V: array (..., k)
        :return: array (..., k+1)
        """
        # add bias terms
        V_bias = np.ones(V.shape[:-1] + (1, ))
        V = np.concatenate((V_bias, V), axis=-1)
        return V

    def grad_log_wrt_params(self, U, Z):
        """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, d)
             either data or noise for NCE
        :param Z: (nz, n, m) or (n, m)
            m-dimensional latent variable samples. nz per datapoint in U.
        :return grad: array (len(theta), nz, n)
        """
        if len(Z.shape) == 2:
            Z = Z.reshape((1, ) + Z.shape)
        U, Z = self.add_bias_terms(U), self.add_bias_terms(Z)  # (n, d+1), (nz, n, m+1)
        nz, n, m_add_1 = Z.shape
        d_add_1 = U.shape[1]

        U = np.repeat(U, m_add_1, axis=-1)  # (n, [d+1]*[m+1])
        Z = np.tile(Z, d_add_1)  # (nz, n, [d+1]*[m+1])

        grad = Z * U  # (nz, n, [d+1]*[m+1])
        grad = np.transpose(grad, (2, 0, 1))
        validate_shape(grad.shape, self.theta_shape + (nz, n))

        return grad

    def grad_log_visible_marginal_wrt_params(self, U):
        """ Nabla_theta(log(phi(u; theta))), where we have summed out the latents

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

    def p_visibles_given_latents(self, Z):
        """Return probability binary (visible) variable is 1 given latent Z
        :param Z: array (n, m)
            n (hidden) data points
        :return: (n, d)
            For each of the n latent variables that we condition on, calculate
            an d-length vector of probabilities, describing the chance
            that the corresponding visible binary variable is 1
        """
        Z = self.add_bias_terms(Z)  # (n, m+1)
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)

        a = np.dot(Z, W[1:, :].T)  # (n, d)
        p1 = sigmoid(a)

        return p1

    def p_latents_given_visibles(self, U):
        """Return probabilities that each binary latent variable is 1
        :param U: array (n, d)
            n (visible) data points that we condition on
        :return: array (n, m)
            For each of the n visibles that we condition on, calculate
            an m-length vector of probabilities, describing the chance
            that the corresponding latent binary variable is 1
        """
        U = self.add_bias_terms(U)  # (n, d+1)
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)

        a = np.dot(U, W[:, 1:])  # (n, m)
        p1 = sigmoid(a)

        return p1

    def p_latent_samples_given_visibles(self, Z, U):
        """
        :param Z: array (nz, n, m)
        :param U: array (n, d)
            n data points with dimension d
        :return: array (nz, n)
            probability of latent variables given data
        """
        p1 = self.p_latents_given_visibles(U)  # (n, m)

        posteriors = (Z == 0)*(1 - p1) + (Z == 1)*p1  # (nz, n, m)
        posterior = np.product(posteriors, axis=-1)  # (nz, n)

        return posterior

    def sample(self, n, num_iter=100):
        """ Sample n 'visible' and latent datapoints using gibbs sampling
        :param n: int
            number of data points to sample
        :param num_iter: int
            number of steps to take in gibbs sampling
        :return X:  array(n, d)
        """
        d, m = np.array(self.W_shape) - 1
        Z = self.rng.uniform(0, 1, (n, m)) < 0.5  # initialise gibbs sample

        for _ in range(num_iter):
            pu_given_z = self.p_visibles_given_latents(Z)  # (n, d)
            U = self.rng.uniform(0, 1, pu_given_z.shape) < pu_given_z  # (n, d)
            pz_given_u = self.p_latents_given_visibles(U)  # (n, m)
            Z = self.rng.uniform(0, 1, pz_given_u.shape) < pz_given_u  # (n, m)

        return U.astype(int), Z.astype(int)

    """
        # noinspection PyUnboundLocalVariable
        def sample_for_contrastive_divergence(self, U_0, num_iter=100):
            Sample the random variables needed for CD learning

            Contrastive divergence learning (see optimisers class)
            needs two things:
            - E(Z|U_0). The expected value of latents given data
            - (V_k, Z_k) sampled from the *model*, which requires gibbs
            sampling. In fact, it is recommended not to directly sample
            the final Z_k, but to return E(Z_k | V_k), so we do this.

            :param U_0 array (n, d)
                data vectors used to initialise gibbs sampling
            :param num_iter: int
                number of steps to take in gibbs sampling
            :return p_z0_given_u0 array (n, m)
                        expected value of latents given U_0
                    U: array (n, d)
                        binary sample of visibles after k gibbs steps
                    p_z_given_u: array (n, m)
                        expected value of latents after k gibbs steps

            p_z0_given_u0 = self.p_latents_given_visibles(U_0)  # (n, m)
            Z_0 = self.rng.uniform(0, 1, p_z0_given_u0.shape) < p_z0_given_u0  # (n, m)

            Z = Z_0
            for i in range(num_iter):
                p_u_given_z = self.p_visibles_given_latents(Z)  # (n, d)
                U = self.rng.uniform(0, 1, p_u_given_z.shape) < p_u_given_z  # (n, d)

                p_z_given_u = self.p_latents_given_visibles(U)  # (n, m)
                Z = self.rng.uniform(0, 1, p_z_given_u.shape) < p_z_given_u  # (n, m)

            return p_z0_given_u0, U.astype(int), p_z_given_u
    """

    # noinspection PyUnboundLocalVariable
    def sample_for_contrastive_divergence(self, U0, num_iter=100):
        """Sample the random variables needed for CD learning

        Contrastive divergence learning (see optimisers class)
        needs two things:
        - a sample Z0 ~ P(Z|U0). where U0 is the data
        - (V_k, Z_k) sampled from the *model*, which requires gibbs
        sampling with k gibbs steps.

        :param U0 array (n, d)
            data vectors used to initialise gibbs sampling
        :param num_iter: int
            number of steps to take in gibbs sampling
        :return Z0 array (n, m)
                    sample of latents given U0 (the data)
                U: array (n, d)
                    binary sample of visibles after k gibbs steps
                Z: array (n, m)
                    sample of latents after k gibbs steps
        """

        p_z0_given_u0 = self.p_latents_given_visibles(U0)  # (n, m)
        Z0 = self.rng.uniform(0, 1, p_z0_given_u0.shape) < p_z0_given_u0  # (n, m)

        Z = Z0
        for i in range(num_iter):
            p_u_given_z = self.p_visibles_given_latents(Z)  # (n, d)
            U = self.rng.uniform(0, 1, p_u_given_z.shape) < p_u_given_z  # (n, d)

            p_z_given_u = self.p_latents_given_visibles(U)  # (n, m)
            Z = self.rng.uniform(0, 1, p_z_given_u.shape) < p_z_given_u  # (n, m)

        return Z0.astype(int), U.astype(int), Z.astype(int)

    def sample_from_latents_given_visibles(self, nz, U):
        """Return latent samples conditioning on visibles.
        
        For many models, sampling latents given visibles requires MCMC. Here, we have
        access to the exact posterior over latents and sampling from it is easy.
        :param nz int
            number of latent samples per visible samples.
        :param U: array (n, d)
            n (visible) data points that we condition on
        :return: array (n, m)
            For each of the n visibles that we condition on, return nz latent samples#
        """
        p1 = self.p_latents_given_visibles(U)  # (n, m)
        Z_shape = (nz, ) + p1.shape

        Z = self.rng.uniform(0, 1, Z_shape) < p1  # (nz, n, m)

        return Z.astype(int)

    def reset_norm_const(self):
        """Reset normalisation constant using current theta"""
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        d, m = W.shape[0] - 1, W.shape[1] - 1
        assert d*(2**m) <= 10**9, "Calculating the normalisation" \
            "constant has O(d 2**m) cost. Assertion raised since d*2^m " \
            "equal to {}, which exceeds the current limit of 10^9,".format(d*(2**m))

        W_times_all_latents = self.get_z_marginalization_matrix(W)  # (d+1, 2**m)
        W_times_all_latents = np.exp(W_times_all_latents)
        W_times_all_latents += np.concatenate((np.zeros((1, 2**m)),
                                               np.ones((d, 2**m))), axis=0)
        self.norm_const = np.sum(np.product(W_times_all_latents, axis=0))

    def normalised_and_marginalised_over_z(self, U, reset_norm_const=True):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param U: array (n, d)
             either data or noise for NCE
        :param reset_norm_const bool
            If true, recalculate the normalisation constant for current params
        :return
        array (n, )
            probabilities of datapoints under p(u) i.e with z marginalised out
            and the distribution has been normalised
        norm_const: float
            the normalisation constant for current parameters

        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        d, m = W.shape[0] - 1, W.shape[1] - 1
        assert d*(2**m) <= 10**9, "Calculating the normalisation" \
            "constant has O(d 2**m) cost. Assertion raised since d*2^m " \
            "equal to {}, which exceeds the current limit of 10^9,".format(d*(2**m))

        phi_u = self.marginalised_over_z(U)  # (n, ), (d+1, 2**m)

        if reset_norm_const:
            self.reset_norm_const()

        return phi_u / self.norm_const, self.norm_const

    def marginalised_over_z(self, U):
        """
        :param U: array (n, d)
             either data or noise for NCE
        :return
        val: array (n, )
            probabilities of datapoints under phi(x) i.e with z marginalised out
        """
        W = self.theta.reshape(self.W_shape)
        d, m = np.array(W.shape) - 1
        U = self.add_bias_terms(U)  # (n, d+1)
        uW = np.dot(U, W)  # (n, m+1)
        exp_uW = np.exp(uW)
        exp_uW += np.concatenate((np.zeros((len(U), 1)), np.ones((len(U), m))), axis=1)
        val = np.product(exp_uW, axis=1)  # (n, )
        validate_shape(val.shape, (U.shape[0], ))

        return val

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

    def normalised_and_marginalised_over_u(self, Z, reset_norm_const=True):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param Z: array (n, m)
             latent variables
        :param reset_norm_const bool
            If true, recalculate the normalisation constant for current params
        :return
        array (n, )
            probabilities of datapoints under p(z) i.e with u marginalised out
            and the distribution has been normalised
        norm_const: float
            the normalisation constant for current parameters
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        d, m = W.shape[0] - 1, W.shape[1] - 1
        assert d*(2**m) <= 10**7, "Calculating the normalisation" \
            "constant has O(d 2**m) cost. Assertion raised since d*2^m " \
            "equal to {}, which exceeds the current limit of 10^7,".format(d*(2**m))

        phi_z, uW = self.marginalised_over_u(Z)  # (n, ), (2**d, m+1)

        if reset_norm_const:
            self.reset_norm_const()

        return phi_z / self.norm_const, self.norm_const

    def marginalised_over_u(self, Z):
        """
        :param Z: array (n, m)
             latent variables
        :return
        val: array (n, )
            probabilities of datapoints under p(z) i.e with u marginalised out
            and the distribution has been normalised
        marginalisation_matrix: array (d+1, 2**m)
            WZ, where Z contains all possible m-length binary vectors
            vertically stacked, and W is the RBM weight matrix
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)

        Z = self.add_bias_terms(Z)  # (n, m+1)
        uW = self.get_u_marginalization_matrix(W)  # (2**d, m+1)
        uWz = np.dot(uW, Z.T)  # (2**d, n)

        # For each datapoint and setting of the latent variable, compute value of RBM
        val = np.exp(uWz)  # (2**d, n)

        # For each datapoint, marginalize (i.e sum) over the visibles
        val = np.sum(val, axis=0)   # (n,)

        validate_shape(val.shape, (Z.shape[0], ))

        return val, uW

    def get_u_marginalization_matrix(self, W):
        """Stack each of the 2**m possible binary visible vectors into a
        (2**d, d+1) matrix Z, and then return the matrix multiply UW

        :param W: array (d+1, m+1)
            weight matrix of RBM
        :return: array (2**d, m+1)
            UW, where U contains all possible d-length binary vectors
        """
        d = W.shape[0] - 1
        U = self.get_all_binary_vectors(d)  # (2**d, d)
        U = self.add_bias_terms(U)  # (2**d, d+1)
        uW = np.dot(U, W)  # (2**d, m+1)

        return uW

    def evaluate_on_entire_domain(self, normalise=False, reset_norm_const=True):
        """return value of rbm on every combination of visible and hidden input

        :param normalise: bool
            if true, first normalise the distribution.
        :param reset_norm_const: bool
            if True, recalculate the normalisation constant using current
            parameters. If false, use already saved norm const. Note: this
            argument only matters if normalise=True.
        :return: array (2**d, 2**m)
            matrix containing all values of rbm
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        d, m = W.shape[0] - 1, W.shape[1] - 1
        assert d + m <= 20, "Evaluating over the whole domain " \
            "has O(2**(m+d)) cost. Assertion raised since m+d is " \
            "equal to {}, which exceeds the current limit of 20".format(d+m)

        U = self.get_all_binary_vectors(d)  # (2**d, d)
        U = self.add_bias_terms(U)  # (2**d, d+1)
        Wz = self.get_z_marginalization_matrix(W)  # (d+1, 2**m)
        uWz = np.dot(U, Wz)  # (2**d, 2**m)
        val = np.exp(uWz)
        if normalise:
            if (not self.norm_const) or reset_norm_const:
                self.reset_norm_const()
            val /= self.norm_const

        return val

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

    def set_alpha(self, alpha):
        """"Only useful when using the rbm as a variational noise distribution in adaptive VNCE """
        self.theta = deepcopy(alpha)


class StarsAndMoonsModel(LatentVarModel):

    def __init__(self, c, truncate_gaussian=False, rng=None):
        """Initialise the parameters of the model.

        the model is implicitly specified by:
            z ~ N(0, 1)
            x = w + e
            w = (z_1*z_2, z_1*z_2)
            e ~ N(0, c*I)
        where x, z and e are all 2-dimensional real vectors. e independent from z.
        Note that the model is normalised.

        If self.truncate_gaussian=True, then:
            e ~ N(0, c*I) * I(>= 0)
        where I(>= 0) is an indicator function, equal to one when both components of e are non-negative.
        :param c: array (2, )
            variance of each component of e
        :param truncate_gaussian: boolean
            if True, truncate distribution of e so its components are positive (see math in docstring)
        """
        self.c = c
        self.truncate_gaussian = truncate_gaussian
        super().__init__(c, rng=rng)

    def __call__(self, U, Z, log=False):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, 2)
             either data or noise for NCE
        :param Z: (nz, n, 2) or (n, 2)
            latent data
        :param log: boolean
            if True, return value of logpdf
        :return array (nz, n)
            probability of each datapoint & its corresponding latent under
            the joint distribution of the model
        """
        if Z.ndim == 2:
            new_shape = (1, ) + Z.shape
            Z = Z.reshape(new_shape)
        Z1_Z2 = np.product(Z, axis=-1)[..., np.newaxis]
        Z1_Z2 = np.tile(Z1_Z2, (1, 1, 2))  # (nz, n, 2)
        validate_shape(Z1_Z2.shape, Z.shape)

        c1 = -np.log(2*np.pi*self.c)
        first_term = c1 - (1 / (2 * self.c)) * np.sum((U - Z1_Z2) * (U - Z1_Z2), axis=-1)  # (nz, n)
        if self.truncate_gaussian:
            mask = np.all(U - Z1_Z2 > 0, axis=-1)
            first_term = mask * first_term + (1 - mask) * -10  # should be -infty, but -10 avoids numerical issues

        c2 = -np.log(2 * np.pi)
        second_term = c2 - 0.5 * np.sum(Z * Z, axis=-1)  # (nz, n)

        val = first_term + second_term
        if not log:
            val = np.exp(val)

        return val

    def grad_log_wrt_params(self, U, Z):
        pass

    def grad_log_wrt_nn_outputs(self, U, Z, grad_z_wrt_nn_outputs):
        """ Evaluate gradient of model w.r.t the variational parameters

        Note that the model depends on the variational parameters because we are using
        the reparameterisation trick.

        :param U: array (n, 2)
             either data or noise for NCE
        :param Z: (nz, n, 2) or (n, 2)
            latent data
        :param grad_z_wrt_nn_outputs: array (len(alpha), nz, n, 2)
        :return grad: array (4, nz, n)
        """
        grad_Z = grad_z_wrt_nn_outputs  # (4, nz, n, 2)

        Z1_Z2 = np.product(Z, axis=-1)[..., np.newaxis]
        Z1_Z2 = np.tile(Z1_Z2, (1, 1, 2))  # (nz, n, 2)
        validate_shape(Z1_Z2.shape, Z.shape)

        Z_swap = Z[:, :, [1, 0]]  # (nz, n, 2)
        A = grad_Z * Z_swap  # (4, nz, n, 2)

        grad_log_posterior_x_given_z = (1 / self.c) * np.sum(A, axis=-1) * np.sum((U - Z1_Z2), axis=-1)  # (4, nz, n)
        if self.truncate_gaussian:
            mask = np.all(U - Z1_Z2 > 0, axis=-1)
            grad_log_posterior_x_given_z *= mask
        validate_shape(grad_log_posterior_x_given_z.shape, grad_Z.shape[:-1])

        grad_log_prior_z = - np.sum(grad_Z * Z, axis=-1)  # (4, nz, n)
        validate_shape(grad_log_prior_z.shape, grad_Z.shape[:-1])

        return grad_log_posterior_x_given_z + grad_log_prior_z  # (4, nz, n)

    def sample(self, n):
        z_mean = np.zeros(2)
        z_cov = np.identity(2)
        z = rnd.multivariate_normal(mean=z_mean, cov=z_cov, size=n)
        z1_z2 = np.product(z, axis=1)
        z1_z2 = np.tile(z1_z2.reshape(-1, 1), (1, 2))

        noise_mean = np.zeros(2)
        noise_cov = self.c * np.identity(2)
        noise = rnd.multivariate_normal(mean=noise_mean, cov=noise_cov, size=n)

        if self.truncate_gaussian:
            noise = np.abs(noise)

        return z, z1_z2 + noise  # (n, 2)
