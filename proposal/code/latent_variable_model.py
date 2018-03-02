"""Module provides classes for unnormalised latent-variable models
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from numpy import random as rnd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity as kd
# noinspection PyPep8Naming



# noinspection PyPep8Naming
class LatentVarModel(metaclass=ABCMeta):

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
class MixtureOfTwoGaussians(LatentVarModel):
    """ *normalised* Mixture of two Gaussians.

    The model takes the form:
    phi(u; theta) = (1/2)*Z*N(u; 0, sigma(theta)) + (1/2)*(1-Z)N(u; 0, sigma1)
    where Z is a binary latent variable and sigma = exp(theta)
    """
    def __init__(self, theta, sigma1=1):
        """Initialise std deviations of gaussians

        :param theta: array of shape (1, )
            (log) standard deviation of one of the two gaussians. Note that
            for optimisation we use the log, to enforce positivity.
        :param sigma1: float
        """
        self.sigma1 = sigma1
        super().__init__(theta)

    def __call__(self, U, Z):
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

        return first_term + second_term

    def marginal_z_0(self, U):
        """ Return value of model for z=0
        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n,)
        """
        sigma = np.exp(self.theta)
        return 0.5*norm.pdf(U.reshape(-1), 0, sigma)

    def marginal_z_1(self, U):
        """ Return value of model for z=1
        :param U: array (n, 1)
             either data or noise for NCE
        :return array (n,)
        """
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
        grad = (U**2 * np.exp(-2*self.theta)) - 1  # (n, 1)
        return grad.T  # (1, n)

    def sample(self, n):
        """ Sample n values from the model
        :param n: number of data points to sample
        :return: data array of shape (n, )
        """
        sigma = np.exp(self.theta)
        w = rnd.uniform(0, 1, n) > 0.5
        x = (w == 0)*(rnd.randn(n)*sigma) + (w == 1)*(rnd.randn(n)*self.sigma1)
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
class MixtureOfTwoUnnormalisedGaussians(LatentVarModel):
    """ Mixture of two *unnormalised* Gaussians given by:
    phi(u; theta) = e^-(theta[0]) [exp(-x**2/2exp(theta[1])**2) +
                                   exp(-x**/2sigma1**2)]

    This class represents a sum of two gaussians, each of which has
    lost its normalising constant.
    """

    def __init__(self, theta, sigma1=1):
        """Initialise theta and stdevs of gaussians
        :param theta: array (2, )
            theta[0] is a scaling parameter. See the formula
            in the class docstring. theta[1] is (log) standard deviation
            of one of the two gaussians. We use the log to enforce positivity.
        :param sigma1: float/int
            standard deviation of one of the gaussians
        """
        self.sigma1 = sigma1
        super().__init__(theta)

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
        grad = np.zeros((self.theta.size, U.shape[0]))  # (2, n)
        grad[0] = -1
        a = (U**2 * np.exp(-2*self.theta[1]))  # (n, 1)
        grad[1] = a.reshape(-1)
        return grad  # (2, n)

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
        _ = plt.figure(figsize=figsize)
        u = np.arange(-10, 10, 0.01)

        x_density = kd(bandwidth=bandwidth).fit(X.reshape(-1, 1))
        x_density_samples = np.exp(x_density.score_samples(u.reshape(-1, 1)))
        plt.plot(u, x_density_samples, label='kde')

        px = self.normalised_and_marginalised_over_z(u)
        plt.plot(u, px, label='true', c='r', linestyle='--')

        plt.legend()
        plt.grid()


# noinspection PyPep8Naming,PyMissingConstructor,PyArgumentList,PyTypeChecker
class RestrictedBoltzmannMachine(LatentVarModel):
    # todo: complete docstring
    """
    """

    def __init__(self, W, a, b):
        """Initialise the parameters of the RBM.

        :param W: array (d, m)
            Weight matrix. d=num_visibles, m=num_hiddens
        :param a: array (d, )
            Visible weights
        :param b: array (m, )
            hidden weights
        """
        self.W_shape = W.shape
        self.a_shape = a.shape
        self.b_shape = b.shape
        theta = np.concatenate((W.reshape(-1), a, b))
        super().__init__(theta)

    def __call__(self, U, Z):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, d)
             either data or noise for NCE
        :param Z: (nz, n, m)
            nz*n 1-dimensional latent variable samples for data U.
        :return array (nz, n)
        """
        W, a, b = self.get_weights()
        # todo complete method
        Z = Z.reshape(Z.shape[0], Z.shape[1])

        first_term = (Z == 0)*self.marginal_z_0(U)  # (nz, n)
        second_term = (Z == 1)*self.marginal_z_1(U)  # (nz, n)

        return first_term + second_term

    def get_weights(self):
        """Return W, a, b with correct shapes"""
        W_size = np.product(self.W_shape)
        a_size = self.a_shape[0]
        b_size = self.b_shape[0]

        W = self.theta[:W_size].reshape(self.W_shape)
        a = self.theta[W_size:W_size + a_size]
        b = self.theta[W_size + a_size:W_size + a_size + b_size]

        return W, a, b

    def marginal_z_0(self, U):
        """ Return value of model for z=0
        :param U: array (n, d)
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
        :param U: array (n, d)
            N is number of data points
        :return array (n,)
        """
        U = U.reshape(-1)
        a = np.exp(-self.theta[0])  # scaling parameter
        b = np.exp(-U**2 / (2*(self.sigma1**2)))
        return a * b

    def grad_log_wrt_params(self, U, Z):
        """ Nabla_theta(log(phi(x,z; theta))) where phi is the unnormalised model

        :param U: array (n, d)
             either data or noise for NCE
        :param Z: (nz, n, m)
            m-dimensional latent variable samples. nz per datapoint in U.
        :return grad: array (len(theta), nz, n)
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
        """ nabla_theta(log(model))
        :param U: array (n, d)
            either data or noise for NCE
        :return array (len(theta), n)
        """
        grad = np.zeros((self.theta.size, U.shape[0]))  # (2, n)
        grad[0] = -1
        a = (U**2 * np.exp(-2*self.theta[1]))  # (n, 1)
        grad[1] = a.reshape(-1)
        return grad  # (2, n)

    def sample(self, n):
        """ Sample n values from the model

        :param n: number of data points to sample
        :return: data array of shape (n, d)
        """
        sigma = np.exp(self.theta[1])
        a = self.sigma1 / (sigma + self.sigma1)
        w = rnd.uniform(0, 1, n) < a
        x = (w == 0)*(rnd.randn(n)*sigma) + (w == 1)*(rnd.randn(n)*self.sigma1)
        return x.reshape(-1, 1)

    def normalised_and_marginalised_over_z(self, U):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param U: array (n, d)
             either data or noise for NCE
        :return array (n, )
            probabilities of datapoints under p(x) i.e with z marginalised out
            and the distribution has been normalised
        """
        sigma = np.exp(self.theta[1])
        a = sigma / (sigma + self.sigma1)
        return a*norm.pdf(U.reshape(-1), 0, sigma) + (1 - a)*norm.pdf(U.reshape(-1), 0, self.sigma1)
