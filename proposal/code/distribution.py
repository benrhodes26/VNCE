"""Module provides classes for probability distributions.
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from numpy import random as rnd
from pandas import DataFrame
from pomegranate import BayesianNetwork
from scipy.stats import norm, multivariate_normal
from utils import sigmoid

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
        return self.rng.uniform(0, 1, (num_noise_samples, len(self.p))) < self.p


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

