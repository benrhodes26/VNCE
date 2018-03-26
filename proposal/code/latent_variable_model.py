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
    """ Type of probabilistic graphical model that specifies an unnormalised
    joint distribution over data and latent variables. The model is given by:
             phi(u, z; W, a, b) = exp(uWz + au + bz)
    where
        - u, a are d-dimensional vectors
        - z, b are m-dimensional vectors
        - W is a (d, m) matrix
    a, b and W are all weights, u is the visible data and z is the latent vector.
    """

    def __init__(self, W):
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
        theta = W.reshape(-1)
        super().__init__(theta)

    def __call__(self, U, Z):
        """ Evaluate model for each data point U[i, :]

        :param U: array (n, d)
             either data or noise for NCE
        :param Z: (nz, n, m) or (n, m)
            nz*n m-dimensional latent variable samples for data U.
        :return array (nz, n)
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)

        U, Z = self.add_bias_terms(U), self.add_bias_terms(Z)
        # (n, d+1), (nz, n, m+1)

        uW = np.dot(U, W)  # (n, m+1)
        uWz = np.sum(Z * uW, axis=-1)  # (nz, n)

        val = np.exp(uWz)  # (nz, n)
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
        """Return probability that the binary latent variable is 1
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

    def sample(self, n, num_iter=100):
        """ Sample n 'visible' and latent datapoints using gibbs sampling
        :param n: int
            number of data points to sample
        :param num_iter: int
            number of steps to take in gibbs sampling
        :return X:  array(n, d)
        """
        d, m = np.array(self.W_shape) - 1
        Z = rnd.uniform(0, 1, (n, m)) < 0.5  # initialise gibbs sample

        for _ in range(num_iter):
            pu_given_z = self.p_visibles_given_latents(Z)  # (n, d)
            U = rnd.uniform(0, 1, pu_given_z.shape) < pu_given_z  # (n, d)
            pz_given_u = self.p_latents_given_visibles(U)  # (n, m)
            Z = rnd.uniform(0, 1, pz_given_u.shape) < pz_given_u  # (n, m)

        return U.astype(int), Z.astype(int)

    # noinspection PyUnboundLocalVariable
    def sample_for_contrastive_divergence(self, U_0, num_iter=100):
        """ Sample the random variables needed for CD learning

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
        """
        p_z0_given_u0 = self.p_latents_given_visibles(U_0)  # (n, m)
        Z_0 = rnd.uniform(0, 1, p_z0_given_u0.shape) < p_z0_given_u0  # (n, m)

        Z = Z_0
        for i in range(num_iter):
            p_u_given_z = self.p_visibles_given_latents(Z)  # (n, d)
            U = rnd.uniform(0, 1, p_u_given_z.shape) < p_u_given_z  # (n, d)

            p_z_given_u = self.p_latents_given_visibles(U)  # (n, m)
            Z = rnd.uniform(0, 1, p_z_given_u.shape) < p_z_given_u  # (n, m)

        return p_z0_given_u0, U.astype(int), p_z_given_u

    def normalised_and_marginalised_over_z(self, U):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param U: array (n, d)
             either data or noise for NCE
        :return array (n, )
            probabilities of datapoints under p(u) i.e with z marginalised out
            and the distribution has been normalised
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        assert np.sum(W.shape) <= 22, "Won't normalize when the sum of latent and " \
            "and visible dimensions is {}. Maximum is 20, since this operation has " \
            "O(2**(d+m)) cost".format(np.sum(W.shape))

        # get values of RBM marginalized over z. During the marginalisation,
        # we construct a matrix WZ, where Z is a (m+1, 2**m) matrix containing
        # all possible m-length binary vectors (with bias terms added). We reuse
        # this matrix when computing the normalisation constant
        phi_u, Wz = self.marginalised_over_z(U)  # (n, ), (d+1, 2**m)

        # compute the normalisation constant
        all_visibles = self.get_all_binary_vectors(U.shape[1])  # (2**d, d)
        all_visibles = self.add_bias_terms(all_visibles)  # (2**d, d+1)
        uWz = np.dot(all_visibles, Wz)  # (2**d, 2**m)
        exp_uWz = np.exp(uWz)
        norm_constant = np.sum(exp_uWz)

        return phi_u / norm_constant, norm_constant

    def marginalised_over_z(self, U):
        """
        :param U: array (n, d)
             either data or noise for NCE
        :return
        val: array (n, )
            probabilities of datapoints under p(x) i.e with z marginalised out
            and the distribution has been normalised
        marginalisation_matrix: array (d+1, 2**m)
            WZ, where Z contains all possible m-length binary vectors
            vertically stacked, and W is the RBM weight matrix
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)

        U = self.add_bias_terms(U)  # (n, d+1)
        Wz = self.get_z_marginalization_matrix(W)  # (d+1, 2**m)
        uWz = np.dot(U, Wz)  # (n, 2**m)

        # For each datapoint and setting of the latent variable, compute value of RBM
        val = np.exp(uWz)  # (n, 2**m)

        # For each datapoint, marginalize (i.e sum) over the latents
        val = np.sum(val, axis=-1)   # (n,)

        validate_shape(val.shape, (U.shape[0], ))

        return val, Wz

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

    def normalised_and_marginalised_over_u(self, Z):
        """Return values of p(U), where p is the (normalised) marginal over x

        :param Z: array (n, m)
             latent variables
        :return array (n, )
            probabilities of datapoints under p(z) i.e with u marginalised out
            and the distribution has been normalised
        """
        W = self.theta.reshape(self.W_shape)  # (d+1, m+1)
        assert np.sum(W.shape) <= 22, "Won't normalize when the sum of latent and " \
            "and visible dimensions is {}. Maximum is 20, since this operation has " \
            "O(2**(d+m)) cost".format(np.sum(W.shape))

        # get values of RBM marginalized over u. During the marginalisation,
        # we construct a matrix UW, where U is a (d+1, 2**d) matrix containing
        # all possible d-length binary vectors (with bias terms added). We reuse
        # this matrix when computing the normalisation constant
        phi_z, uW = self.marginalised_over_u(Z)  # (n, ), (2**d, m+1)

        # compute the normalisation constant
        all_hiddens = self.get_all_binary_vectors(Z.shape[1])  # (2**m, m)
        all_hiddens = self.add_bias_terms(all_hiddens)  # (2**m, m+1)
        uWz = np.dot(uW, all_hiddens.T)  # (2**d, 2**m)
        exp_uWz = np.exp(uWz)
        norm_constant = np.sum(exp_uWz)

        return phi_z / norm_constant, norm_constant

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
