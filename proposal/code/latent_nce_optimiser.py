""" Optimisers for an unnormalised latent variable probability distribution.

These classes wraps together a set of methods for optimising an unnormalised
probabilistic model with latent variables using a novel variant of NCE.
See the following for an introduction to NCE:
(http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)
"""

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import random as rnd
from scipy.optimize import minimize
from utils import validate_shape

DEFAULT_SEED = 22012018

# noinspection PyPep8Naming,PyTypeChecker,PyMethodMayBeStatic
class LatentNCEOptimiser:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model with latent variables. This Optimiser estimates all expectations
    with respect to the variational distribution using monte-carlo estimates.
    If you have access to analytic expressions of these expectations, consider
    using LatentNCEOptimiserWithAnalyticExpectations.

    This class has two methods for optimisation: fit() and fit_with_analytic_q().
    The latter assumes that you have an analytic expression for the true posterior
    of the latent variables given data. In general, this posterior is intractable,
    and so you will need

    """
    def __init__(self, model, noise, variational_dist, sample_size, nu=1,
                 latent_samples_per_datapoint=1,  eps=1e-15):
        """ Initialise unnormalised model and distributions.

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, Z, theta) where:
            - U is (n, d) array of data
            - Z is a (nz, n, m) collection of m-dimensional latent variables
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param variational_dist: Distribution
            variational distribution with parameters alpha. we use it
            to optimise a lower bound on the log likelihood of phi_model.
            Has arguments (Z, U) where:
            - Z is a (nz, n, m) dimensional latent variable
            - U is a (-1, d) dimensional array (either data or noise)
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.q = variational_dist
        self.nu = nu
        self.n = sample_size
        self.nz = latent_samples_per_datapoint
        self.eps = eps
        self.Y = self.pn.sample(int(sample_size * nu))  # generate noise

    def r(self, U, Z):
        """ compute ratio phi / (pn*q).
        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :param Z: array of shape (nz, ?, m)
            ? is either n or n*nu
        :return: array of shape (nz, ?)
            ? is either n or n*nu
        """
        phi = self.phi(U, Z)
        q = self.q(Z, U)
        val = phi / (q*self.pn(U) + self.eps)
        validate_shape(val.shape, (Z.shape[0], Z.shape[1]))

        return val

    def compute_J1(self, X, ZX=None, ZY=None):
        """Return MC estimate of Lower bound of NCE objective

        :param X: array (N, d)
            data sample we are training our model on
        :param ZX: array (N, nz, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: array (nz, N*nu)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :return: float
            Value of objective for current parameters
        """
        Y, nu = self.Y, self.nu
        if ZX is None:
            ZX = self.q.sample(self.nz, X)
        if ZY is None:
            ZY = self.q.sample(self.nz, Y)

        r_x = self.r(X, ZX)
        a = nu/(r_x + self.eps)  # (nz, n)
        validate_shape(a.shape, (self.nz, self.n))
        first_term = -np.mean(np.log(1 + a))

        r_y = self.r(Y, ZY)
        b = (1/nu) * np.mean(r_y, axis=0)  # (n*nu, )
        validate_shape(b.shape, (self.n*self.nu, ))
        second_term = -nu*np.mean(np.log(1 + b))

        return first_term + second_term

    def compute_J1_grad(self, X, ZX=None, ZY=None):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :param ZX: (N, nz, m)
            latent variable samples for data x. for each
            x there are nz samples of shape (m, )
        :param ZY: (nz, N*nu)
            latent variable samples for noise y. for each
            y there are nz samples of shape (m, )
        :return grad: array of shape (len(phi.theta), )
        """
        Y = self.Y
        if ZX is None:
            ZX = self.q.sample(self.nz, X)
        if ZY is None:
            ZY = self.q.sample(self.nz, Y)

        gradX = self.phi.grad_log_wrt_params(X, ZX)  # (len(theta), nz, n)
        a = (self._psi_1(X, ZX) - 1)/self._psi_1(X, ZX)  # (nz, n)
        # take double expectation over X and Z
        term_1 = np.mean(gradX * a, axis=(1, 2))  # (len(theta), )

        gradY = self.phi.grad_log_wrt_params(Y, ZY)  # (len(theta), nz, n)
        r = self.r(Y, ZY)  # (nz, n)
        # Expectation over ZY
        E_ZY = np.mean(gradY * r, axis=1)  # (len(theta), n)
        one_over_psi2 = 1/self._psi_2(Y, ZY)  # (n, )
        # Expectation over Y
        term_2 = - np.mean(E_ZY * one_over_psi2, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        validate_shape(grad.shape, self.phi.theta_shape)

        return grad

    def _psi_1(self, U, Z):
        """Return array (nz, n)"""
        return 1 + (self.nu / self.r(U, Z))

    def _psi_2(self, U, Z):
        """Return array (n, )"""
        return 1 + (1/self.nu) * np.mean(self.r(U, Z), axis=0)

    def fit_using_analytic_q(self, X, theta_inds=None, theta0=np.array([0.5]), disp=True, plot=True,
                             stop_threshold=10**-6, max_num_em_steps=20, gtol=1e-4,
                             ftol=1e-4):
        # todo: update second paragraph, currently not true
        """ Fit the parameters of the model to the data X with an
        EM-type algorithm that switches between optimising the model
        params to produce theta_k and then resetting the variational distribution
        q to be the analytic posterior q(z) =  p(z|x; theta_k). We assume that
        q is parametrised such that setting alpha = theta_k achieves this.

        In general, this posterior is intractable. If this is the case,
        then use self.fit, which takes a parametrised q and then optimises
        its variational parameters alpha.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta_inds: array
            indices of elements of theta we want to optimise. By default
            this get populated with all indices.
        :param theta0: array of shape (1, )
            initial value of theta used for optimisation
        :param disp: bool
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :param stop_threshold: float
            stop EM-type optimisation when the value of the lower
            bound J1 changes by less than stop_threshold
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :param ftol: float
            parameter passed to scipy.minimize. An iteration stops
            when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
        :return
            thetas_after_em_step: (?, num_em_steps),
            J1s: list of lists
                each inner list contains the fevals computed for a
                single step (loop) in the EM-type optimisation
            J1_grads: list of lists
                each inner list contains the grad fevals computed for a
                single step (loop) in the EM-type optimisation
        """
        # by default, optimise all elements of theta
        if theta_inds is None:
            theta_inds = np.arange(len(self.phi.theta))
        # initialise parameters
        self.phi.theta, self.q.alpha = deepcopy(theta0), deepcopy(theta0)

        # Sample latent variables
        # ZX, ZY = self.q.sample(self.nz, X), self.q.sample(self.nz, self.Y)

        thetas_after_em_step = [theta0]  # save theta_k after each em step
        J1s = []  # save fevals for each individual optimisation step
        J1_grads = []   # save grad evals for each individual optimisation step

        num_em_steps = 0
        prev_J1, current_J1 = -999, -9999  # arbitrary negative numbers
        while np.abs(prev_J1 - current_J1) > stop_threshold \
                and num_em_steps < max_num_em_steps:
            prev_J1 = self.compute_J1(X)

            # optimise w.r.t to theta
            self.maximize_J1_wrt_theta(theta_inds, X, J1s, J1_grads,
                                       thetas_after_em_step, disp=disp,
                                       gtol=gtol, ftol=ftol)

            # reset the variational distribution to new posterior
            self.q.alpha = deepcopy(self.phi.theta)

            current_J1 = self.compute_J1(X)
            # store results
            thetas_after_em_step.append(self.phi.theta)
            num_em_steps += 1

        if plot:
            self.plot_loss_curve(J1s)

        return np.array(thetas_after_em_step), np.array(J1s), np.array(J1_grads)

# def maximize_J1_wrt_theta(self, X, ZX, ZY, J1s, J1_grads, disp, gtol=1e-4, ftol=1e-4):
    def maximize_J1_wrt_theta(self, theta_inds,  X, J1s, J1_grads,
                              thetas_after_em_step, disp=True,
                              gtol=1e-4, ftol=1e-4):
        """Return theta that maximises J1

        :param theta_inds: array
            indices of elements of theta we want to optimise
        :param X: array (n, 1)
            data
        :param J1s: list
            list of function evaluations of J1 during optimisation
        :param J1_grads: list
            list of function evaluations of J1_grad during optimisation
        :param disp: bool
            display optimisation results for each iteration
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :param ftol: float
            parameter passed to scipy.minimize. An iteration stops
            when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
        :return new_theta: array
            new theta that maximises the J1 defined by input theta
        """
        J1_vals, J1_grad_vals = [], []

        def J1_k_neg(theta_subset):
            self.phi.theta[theta_inds] = theta_subset
            self.q.alpha[theta_inds] = theta_subset
            thetas_after_em_step.append(self.q.alpha)
            val = -self.compute_J1(X)
            J1_vals.append(-val)
            # print(-val)
            return val

        def J1_k_grad_neg(theta_subset):
            self.phi.theta[theta_inds] = theta_subset
            self.q.alpha[theta_inds] = theta_subset
            grad_val = -self.compute_J1_grad(X)[theta_inds]
            J1_grad_vals.append(-grad_val)
            return grad_val

        res = minimize(J1_k_neg, self.phi.theta[theta_inds], method='CG', jac=J1_k_grad_neg,
                       options={'gtol': gtol, 'ftol': ftol, 'disp': disp})
        #res = minimize(J1_k_neg, self.phi.theta[theta_inds], method='Nelder-Mead',
        #               options={'gtol': gtol, 'ftol': ftol, 'disp': disp})
        self.phi.theta[theta_inds] = res.x
        J1_vals.append(self.compute_J1(X))
        J1s.append(J1_vals)
        J1_grads.append(J1_grad_vals)

    def plot_loss_curve(self, J1s, plot_reset_points=True,
                        endpoints=None, true_theta=None, X=None):
        """plot of objective function during optimisation

        :param J1s: list of lists
            returned by self.fit_using_analytic_q. Each inner list
            contains the fevals computed for a single step (loop) in
            the EM-type optimisation
        :param plot_reset_points bool
            plot vertical line at time-points at each E step of EM algorithm
        :param endpoints: tuple
            (start, end) to specify interval over which to plot curve.
        :param true_theta: array
            true value of parameters that generated the data (if known).
            Used to plot value of J1 at the true parameters, which we
            expect to upper bound the values in J1.
        :param X: array (n, d)
            needed to compute objective function at true_theta.
        :return:
            fig, ax
                plot of objective function during optimisation
        """
        num_fevals_per_step = [len(i) for i in J1s]
        cum_fevals = [sum(num_fevals_per_step[:i + 1]) for i in range(len(num_fevals_per_step))]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        J1s = [i for sublist in J1s for i in sublist]

        if endpoints is not None:
            t = np.arange(endpoints[0], endpoints[1])
            J1 = J1s[endpoints[0]:endpoints[1]]
        else:
            t = np.arange(len(J1s))
            J1 = J1s

        # plot optimisation curve
        ax.plot(t, J1, c='k')

        if plot_reset_points:
            for i in range(len(cum_fevals)):
                ax.plot(cum_fevals[i] * np.array([1, 1]), np.array([np.min(J1), 0]), 'r--',
                        label='resetting variational distribution')

        if (true_theta is not None) and (X is not None):
            current_theta = deepcopy(self.phi.theta)
            current_nz = deepcopy(self.nz)
            self.phi.theta, self.q.alpha = true_theta, true_theta
            self.nz = 100  # increase accuracy of approximation
            optimal_J1 = self.compute_J1(X)
            self.phi.theta, self.q.alpha = current_theta, current_theta
            self.nz = current_nz

            ax.plot((t[0], t[-1]), (optimal_J1, optimal_J1), 'b--',
                    label='J1 evaluated at true theta')

        # Remove duplicates from the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        plt.yticks(np.arange(round(min(J1)) - 1, 1, 1.0))
        return fig, ax

    def __repr__(self):
        return "LatentNCEOptimiser"


# noinspection PyPep8Naming,PyTypeChecker,PyMethodMayBeStatic
class LatentNCEOptimiserWithAnalyticExpectations:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model with latent variables. This Optimiser estimates all expectations
    with respect to the variational distribution using analytic expressions
    provided at initialisation.

    There are 5 expectations required to make this work:

    1) E(r(u, z))
    2) E(log(psi_1(u,z)))
    3) E(grad_theta(log(phi(u,z)) (psi_1(u, z) - 1) / psi_1(u, z))
    4) E(grad_theta(log(phi(u,z)) r(u, z) )
    5) grad_alpha(E(log(1 + nu/r(u, z))))

    Where each expectation is over z ~ q(z | u; alpha).
    r and psi_1 are given by:

    r(u, z) = phi(u, z; theta) / q(z| u; alpha)*pn(u)
    psi_1(u, z) = 1 + (nu/r(u, z))
    """

    def __init__(self, model, noise, variational_dist, sample_size,
                 E1, E2, E3, E4, E5, nu=1, latent_samples_per_datapoint=1,
                 eps=1e-15):
        """ Initialise unnormalised model and distributions.

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, Z, theta) where:
            - U is (n, d) array of data
            - Z is a (nz, n, m) collection of m-dimensional latent variables
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param variational_dist: Distribution
            variational distribution with parameters alpha. we use it
            to optimise a lower bound on the log likelihood of phi_model.
            Has arguments (Z, U) where:
            - Z is a (nz, n, m) dimensional latent variable
            - U is a (-1, d) dimensional array (either data or noise)
        :param E1 function
            expectation E(r(u, z)) w.r.t var_dist. Takes args:
            - (U, phi, q, pn, eps)
        :param E2 function
            expectation E(log(psi_1(x,z))) w.r.t var_dist. Takes args:
            - (X, phi, q, pn, nu, eps)
        :param E3 function
            expectation E((psi_1(x, z) - 1) / psi_1(x, z)) w.r.t var_dist. Takes args:
            - (X, phi, q, pn, nu)
        :param E4 function
            expectation E( grad_theta(log(phi(y,z)) r(y, z) ) w.r.t var_dist. Takes args:
            - (Y, phi, q, pn, nu)
        :param E5 function
            gradient_alpha(E(log(1 + nu/r(x, z)))) w.r.t var_dist. Takes args:
            - (X, phi, q, pn, nu, eps)
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.q = variational_dist
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.E4 = E4
        self.E5 = E5

        self.nu = nu
        self.sample_size = sample_size
        self.nz = latent_samples_per_datapoint
        self.eps = eps
        self.Y = self.pn.sample(int(sample_size * nu))  # generate noise

    def compute_J1(self, X):
        """Return analytic expression of Lower bound of NCE objective

        :param X: array (N, d)
            data sample we are training our model on
        :return: float
            Value of objective for current parameters
        """
        E2 = self.E2(X, self.phi, self.q, self.pn, self.nu, self.eps)
        first_term = -np.mean(E2)

        E1 = self.E1(self.Y, self.phi, self.q, self.pn, self.eps)
        psi_2 = 1 + (1/self.nu)*E1
        second_term = -self.nu * np.mean(np.log(psi_2))

        return first_term + second_term

    def compute_J1_grad_theta(self, X):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :return grad: array of shape (len(phi.theta), )
        """
        E3 = self.E3(X, self.phi, self.q, self.pn, self.nu, self.eps)  # (len(theta), n)
        term_1 = np.mean(E3, axis=1)

        Y = self.Y
        E4 = self.E4(Y, self.phi, self.q, self.pn, self.nu, self.eps)  # (len(theta), n)
        a = 1/(self._psi_2(Y) + self.eps)  # (n, )
        term_2 = - np.mean(a * E4, axis=1)

        grad = term_1 + term_2  # (len(theta), )

        # need theta to be an array for optimisation
        if isinstance(grad, float):
            grad = np.array(grad)
        assert grad.shape == self.phi.theta_shape, 'Expected grad ' \
            'to be shape {}, got {} instead'.format(self.phi.theta_shape,
                                                    grad.shape)
        return grad

    def _psi_2(self, U):
        """Return array (n, )"""
        E1 = self.E1(U, self.phi, self.q, self.pn, self.eps)
        return 1 + (1/self.nu) * E1

    def compute_J1_grad_alpha(self, X):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (n, d)
            data sample we are training our model on.
        :return grad: array of shape (len(q.alpha), )
        """
        E5 = self.E5(X, self.phi, self.q, self.pn, self.nu, self.eps)
        grad = np.mean(E5, axis=1)

        assert grad.shape == self.q.alpha_shape, 'Expected grad ' \
            'to be shape {}, got {} instead'.format(self.phi.alpha_shape,
                                                    grad.shape)
        return grad

    def fit(self, X, theta0, alpha0, disp=True, plot=True, stop_threshold=10**-6,
            max_num_em_steps=20, gtol=1e-4):
        """ Fit the parameters theta to the data X with a variational EM-type algorithm

        The algorithm alternates between optimising the model
        params to produce theta_k (holding the parameter alpha of q fixed)
        and then optimising the variational distribution q to produce alpha_k
        (holding theta_k fixed).

        To optimise w.r.t alpha, we have to differentiate an expectation
        w.r.t to q(.|alpha). In general, this is a difficult stochastic optimisation
        problem requiring clever techniques to solve (see the reparameterisation trick').

        This class assumes you have already provided the analytic expressions for
        the necessary expectations.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta0: array
            initial value of theta used for optimisation
        :param alpha0: array
            initial value of alpha used for optimisation
        :param disp:
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :param stop_threshold: float
            stop EM-type optimisation when the value of the lower
            bound J1 changes by less than stop_threshold
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return
            thetas_after_em_step: (?, num_em_steps),
            J1s: list of lists
                each inner list contains the fevals computed for a
                single E or M step in the EM-type optimisation
            J1_grads: list of lists
                each inner list contains the grad fevals computed for a
                single E or M step in the EM-type optimisation
        """
        # initialise parameters
        self.phi.theta, self.q.alpha = deepcopy(theta0), deepcopy(alpha0)

        alphas_after_em_step = [alpha0]
        thetas_after_em_step = [theta0]  # save theta_k after for each EM step
        J1s = []  # save fevals for each individual E or M step
        J1_grads = []  # # save grad evals for each individual E or M step

        num_em_steps = 0
        prev_J1, current_J1 = -999, -9999  # arbitrary negative numbers
        while np.abs(prev_J1 - current_J1) > stop_threshold\
                and num_em_steps < max_num_em_steps:
            prev_J1 = self.compute_J1(X)

            # optimise w.r.t theta
            self.maximize_J1_wrt_theta(X, J1s, J1_grads, disp, gtol)

            # optimise w.r.t alpha
            self.maximize_J1_wrt_alpha(X, J1s, J1_grads, disp, gtol)

            # store results
            alphas_after_em_step.append(self.q.alpha)
            thetas_after_em_step.append(self.phi.theta)

            current_J1 = self.compute_J1(X)
            num_em_steps += 1

        if plot:
            self.plot_loss_curve(J1s)

        return np.array(thetas_after_em_step), np.array(alphas_after_em_step), J1s, J1_grads

    def maximize_J1_wrt_theta(self, X, J1s, J1_grads, disp, gtol=1e-4):
        """Return theta that maximises J1

        :param X: array (n, 1)
            data
        :param J1s: list
            list of function evaluations of J1 during optimisation
        :param J1_grads: list
            list of function evaluations of J1_grad during optimisation
        :param disp:
            display output from scipy.minimize
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return new_theta: array
            new theta that maximises the J1 defined by input theta
        """
        J1_vals, J1_grad_vals = [], []

        def J1_k_neg(theta):
            self.phi.theta = theta
            val = -self.compute_J1(X)
            J1_vals.append(-val)
            return val

        def J1_k_grad_neg(theta):
            self.phi.theta = theta
            grad_val = -self.compute_J1_grad_theta(X)
            J1_grad_vals.append(-grad_val)
            return grad_val

        _ = minimize(J1_k_neg, self.phi.theta, method='L-BFGS-B', jac=J1_k_grad_neg,
                     options={'gtol': gtol, 'disp': disp})

        J1s.append(J1_vals)
        J1_grads.append(J1_grad_vals)

    def maximize_J1_wrt_alpha(self, X, J1s, J1_grads, disp, gtol=1e-4):
        """Return alpha that maximises J1

        :param X: array (n, 1)
            data
        :param J1s: list
            list of function evaluations of J1 during optimisation
        :param J1_grads: list
            list of function evaluations of J1_grad during optimisation
        :param disp:
            display output from scipy.minimize
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return new_alpha: array
            new alpha that maximises the J1 defined by input alpha
        """
        J1_vals, J1_grad_vals = [], []

        def J1_k_neg(alpha):
            self.q.alpha = alpha
            val = -self.compute_J1(X)
            J1_vals.append(-val)
            return val

        def J1_k_grad_neg(alpha):
            self.q.alpha = alpha
            grad_val = -self.compute_J1_grad_alpha(X)
            J1_grad_vals.append(-grad_val)
            return grad_val

        _ = minimize(J1_k_neg, self.q.alpha, method='L-BFGS-B', jac=J1_k_grad_neg,
                     options={'gtol': gtol, 'disp': disp})

        J1s.append(J1_vals)
        J1_grads.append(J1_grad_vals)

    def plot_loss_curve(self, J1s):
        num_fevals_per_step = [len(i) for i in J1s]
        cum_fevals = [sum(num_fevals_per_step[:i + 1]) for i in range(len(num_fevals_per_step))]
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        J1s = [i for sublist in J1s for i in sublist]
        t = np.arange(len(J1s))
        axs.plot(t, J1s, c='k')
        for i in range(len(cum_fevals)):
            if i % 2 == 0:
                axs.plot(cum_fevals[i] * np.array([1, 1]), plt.get(axs, 'ylim'), 'r--')
            else:
                axs.plot(cum_fevals[i] * np.array([1, 1]), plt.get(axs, 'ylim'), 'b--')

        return fig, axs

    def __repr__(self):
        return "LatentNCEOptimiserWithAnalyticExpectations"


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyTypeChecker
class NCEOptimiser:
    """ Optimiser for estimating/learning the parameters of an unnormalised
    model as proposed in http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf

    This class wraps together the NCE objective function, its gradients with respect
    to the parameters of the unnormalised model and a method for performing the
    optimisation: self.fit().

    """
    def __init__(self, model, noise, sample_size, nu=1, eps=1e-15):
        """ Initialise unnormalised model and noise distribution

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, theta) where:
            - U is (n, d) array of data
        :param noise: Distribution
            noise distribution required for NCE. Has argument U where:
            - U is (n*nu, d) array of noise
        :param nu: ratio of noise to model samples
        :param eps: small constant needed to avoid division by zero
        """
        self.phi = model
        self.pn = noise
        self.nu = nu
        self.sample_size = sample_size
        self.eps = eps
        self.Y = self.pn.sample(int(sample_size * nu))  # generate noise

    def h(self, U):
        """ Compute log(phi(U)) - log(pn(U).
        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :return: array of shape (?)
            ? is either n or n*nu
        """
        return np.log(self.phi(U)) - np.log(self.pn(U))

    def compute_J1(self, X):
        """Return MC estimate of Lower bound of NCE objective

        :param X: array (N, d)
            data sample we are training our model on
        :return: float
            Value of objective for current parameters
        """
        Y, nu = self.Y, self.nu

        a0 = 1 + nu*np.exp(-self.h(X))
        a1 = 1 + (1 / nu)*np.exp(self.h(Y))

        first_term = - np.mean(np.log(a0))
        second_term = - nu*np.mean(np.log(a1))

        return first_term + second_term

    def compute_J1_grad(self, X):
        """Computes J1_grad w.r.t theta using Monte-Carlo

        :param X: array (N, d)
            data sample we are training our model on.
        :return grad: array of shape (len(phi.theta), )
        """
        Y, nu = self.Y, self.nu

        gradX = self.phi.grad_log_wrt_params(X)  # (len(theta), n)
        # a0 = 1 / (1 + (1 / nu)*np.exp(self.h(X)))  # (n,)
        a0 = nu*self.pn(X) / (nu*self.pn(X) + self.phi(X))  # (n,)
        # expectation over X
        term_1 = np.mean(gradX*a0, axis=1)  # (len(theta), )

        gradY = self.phi.grad_log_wrt_params(Y)  # (len(theta), nu*n)
        a1 = self.phi(Y) / (nu*self.pn(Y) + self.phi(Y))  # (n)
        # Expectation over Y
        term_2 = - nu * np.mean(gradY*a1, axis=1)  # (len(theta), )

        grad = term_1 + term_2

        # If theta is 1-dimensional, grad will be a float.
        if isinstance(grad, float):
            grad = np.array(grad)
        assert grad.shape == self.phi.theta_shape, ' ' \
            'Expected grad to be shape {}, got {} instead'.format(self.phi.theta_shape,
                                                                  grad.shape)
        return grad

    def fit(self, X, theta0=np.array([0.5]), disp=True, plot=True, gtol=1e-4):
        """ Fit the parameters of the model to the data X by optimising
        the objective function defined in self.compute_J1(). To do this,
        we use a gradient-based optimiser and define the gradient of J1
        with respect to its parameters theta in self.compute_J1_grad().

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta0: array of shape (1, )
            initial value of theta used for optimisation
        :param disp: bool
            display output from scipy.minimize
        :param plot: bool
            plot optimisation loss curve
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return
            thetas_after_em_step: (?, num_em_steps),
            J1s: list of lists
                each inner list contains the fevals computed for a
                single step (loop) in the EM-type optimisation
            J1_grads: list of lists
                each inner list contains the grad fevals computed for a
                single step (loop) in the EM-type optimisation
        """
        # initialise parameters
        self.phi.theta = deepcopy(theta0)

        # optimise w.r.t to theta
        J1s, J1_grads = self.maximize_J1_wrt_theta(X, disp, gtol)

        if plot:
            self.plot_loss_curve(J1s)

        return np.array(J1s), np.array(J1_grads)

    def maximize_J1_wrt_theta(self, X, disp, gtol=1e-4):
        """Return theta that maximises J1

        :param X: array (n, 1)
            data
        :param disp: bool
            display optimisation results for each iteration
        :param gtol: float
            parameter passed to scipy.minimize. An iteration will stop
            when max{|proj g_i | i = 1, ..., n} <= gtol where
            pg_i is the i-th component of the projected gradient.
        :return J1s: list
            function evaluations during optimisation
                J1_grads: list
            gradient evaluations during optimisation
        """
        J1s, J1_grads = [], []

        def J1_k_neg(theta):
            self.phi.theta = theta
            val = -self.compute_J1(X)
            J1s.append(-val)
            return val

        def J1_k_grad_neg(theta):
            self.phi.theta = theta
            grad_val = -self.compute_J1_grad(X)
            J1_grads.append(-grad_val)
            return grad_val

        _ = minimize(J1_k_neg, self.phi.theta, method='L-BFGS-B', jac=J1_k_grad_neg,
                     options={'gtol': gtol, 'disp': disp})

        return J1s, J1_grads

    def plot_loss_curve(self, J1s):
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        t = np.arange(len(J1s))
        axs.plot(t, J1s, c='k')

        return fig, axs

    def __repr__(self):
        return "NCEOptimiser"


# noinspection PyMethodMayBeStatic,PyPep8Naming,PyTypeChecker
class CDOptimiser:
    """ Contrastive divergence Optimiser for estimating/learning the
    parameters of an unnormalised model as proposed in:
    http://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf

    For a simple, practical guide that informed this implementation, see:
    https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

    """
    def __init__(self, model, rng=None):
        """ Initialise unnormalised model and noise distribution

        :param model: LatentVariableModel
            unnormalised model whose parameters theta we want to optimise.
            Has arguments (U, theta) where:
            - U is (n, d) array of data
        :param rng: random number generator
        """
        self.phi = model
        if not rng:
            self.rng = np.random.RandomState(DEFAULT_SEED)
        else:
            self.rng = rng

    def fit(self, X, theta0, num_gibbs_steps, learning_rate, batch_size, num_epochs=10):
        """ Fit the parameters of the model to the data X.

        We fit the data by using the contrastive divergence gradient updates:

        W_ij =  W_ij + learning_rate*( E(v_i*h_j) - E(v_i*h_j) )

        The first expectation is with respect to p_emp(x). p(z| x)
        where p_emp is the empirical distribution of the data.
        The second expectation is with respect to p_model(x, z), which
        we must use gibbs sampling to sample from.

        Following https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf,
        This implementation directly calculates the expected latents Z, given data
        (as opposed to first sampling from P(Z | X)).
        It also directly calculates the expected value of latents on the *final*
        step of gibbs sampling. for every step before that, we sample
        latents.

        :param X: Array of shape (num_data, data_dim)
            data that we want to fit the model to
        :param theta0: array
            initial values for model's parameters
        :param num_gibbs_steps: int
            number of gibbs steps to perform during learning, when trying
            to sample from the model's distribution
        :param learning rate: float
            learning rate for gradient ascent
        :param batch_size: int
            size of a mini-batch
        :param num_epochs: int
            number of times we loop through training data
        :return thetas list of array
            values of model parameters (theta) after each gradient step
        """
        # todo: make it easy to access biases and non-bias weights separately
        # todo: write a get_learning_rate method using decay
        # initialise parameters
        self.phi.theta = deepcopy(theta0)
        n = X.shape[0]
        thetas = []  # store model params after each gradient step
        for j in range(num_epochs):
            # shuffle data
            perm = self.rng.permutation(n)
            X = X[perm]
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                Z, X_model, Z_model = self.phi.sample_for_contrastive_divergence(
                    X_batch, num_iter=num_gibbs_steps)

                data_grad = self.phi.grad_log_wrt_params(X_batch, Z)  # (len(theta), 1, n)
                data_grad = np.mean(data_grad, axis=(1, 2))  # (len(theta), )

                model_grad = self.phi.grad_log_wrt_params(X_model, Z_model)
                model_grad = np.mean(model_grad, axis=(1, 2))  # (len(theta), )

                grad = data_grad - model_grad  # (len(theta), )
                self.phi.theta += learning_rate * grad
                thetas.append(self.phi.theta)

        return thetas

    def __repr__(self):
        return "CDOptimiser"
