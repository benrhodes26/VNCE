""" Collection of analytic expressions for expectations
required by LatentNCEOptimiserWithAnalyticExpectations class
defined in latent_nce_optimiser.py

There are 5 expectations required to make this work:

1) E(r(u, z))
2) E(log(psi_1(u,z)))
3) E( grad_theta(log(phi(u,z)) (psi_1(u, z) - 1) / psi_1(u, z))
4) E( grad_theta(log(phi(u,z)) r(u, z) )
5) grad_alpha (E(log(1 + nu/r(u, z))))

Where each expectation is over z ~ q(z | u; alpha).

r and psi_1 are given by:

r(u, z) = phi(u, z; theta) / q(z| u; alpha)*pn(u)
psi_1(u, z) = 1 + (nu/r(u, z))

Below are the 5 expectations for the MixtureOfTwoGaussians (normalised or unnormalised)
model with a bernoulli variational distribution (e.g PolynomialSigmoidBernoulli)
"""
import numpy as np
from distribution import PolynomialSigmoidBernoulli as PolySig


# noinspection PyPep8Naming,PyUnusedLocal
def E_r(U, phi, q, pn, eps=10**-15):
    """ E(r(u, z))
    :param U: array (n, 1)
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param nu: float
        ratio of noise to data samples in NCE
    :param pn: Distribution
    :return array (n, )
    """
    term0 = (1 - q.calculate_p(U))*_r0(U, phi, q, pn, eps)
    term1 = q.calculate_p(U)*_r1(U, phi, q, pn, eps)

    return term0 + term1


# noinspection PyPep8Naming
def E_log_psi_1(X, phi, q, pn, nu, eps=10**-15):
    """ E(psi_1(x, z)) = E(log(q + (nu/r(x,z)))

    :param X: array (n, 1)
        data
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param nu: float
        ratio of noise to data samples in NCE
    :param pn: Distribution
    :return array (n, )
    """
    a0 = np.log(_psi_1_0(X, phi, q, pn, nu, eps))
    a1 = np.log(_psi_1_1(X, phi, q, pn, nu, eps))

    term0 = (1 - q.calculate_p(X))*a0
    term1 = q.calculate_p(X)*a1

    return term0 + term1


# noinspection PyPep8Naming
def E_psi_1_ratio_times_grad_log_theta(X, phi, q, pn, nu, eps=10**-15):
    """ E( grad_theta(log(phi(u,z)) (psi_1(u, z) - 1) / psi_1(u, z))
    :param X: array (n, 1)
        data
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param nu: float
        ratio of noise to data samples in NCE
    :param pn: Distribution
    :return array (len(theta), n)
    """
    psi_1_0 = _psi_1_0(X, phi, q, pn, nu, eps)  # (n, )
    a = (psi_1_0 - 1)/psi_1_0  # (n, )
    b = phi.grad_log_wrt_params_analytic(X)  # (len(theta), n)
    E = (1 - q.calculate_p(X)) * a * b  # (len(theta), n)

    return E  # (len(theta), n)


# noinspection PyPep8Naming,PyUnusedLocal
def E_r_times_grad_log_theta(Y, phi, q, pn, nu, eps=10 ** 15):
    """ E( grad_theta(log(phi(u,z)) r(u, z) )

    :param Y: array (n*nu, 1)
        noise for NCE
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param nu: float
        ratio of noise to data samples in NCE
    :param pn: Distribution
    :return array (len(theta), n)
    """
    a = _r0(Y, phi, q, pn, eps)  # (n, )
    b = phi.grad_log_wrt_params_analytic(Y)  # (len(theta), n)
    E = (1 - q.calculate_p(Y)) * a * b  # (len(theta), n)

    return E  # (len(theta), n)


# noinspection PyPep8Naming
def grad_wrt_alpha_of_E_log_psi_1(X, phi, q, pn, nu, eps=10**-15):
    """ grad_alpha (E(log(1 + nu/r(u, z))))
    :param X: array (n, 1)
        data
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param nu: float
        ratio of noise to data samples in NCE
    :param pn: Distribution
    :return: array (len(alpha), n)
    """
    a0 = nu * pn(X)/(phi.marginal_z_0(X) + eps)  # (n, )
    a1 = nu * pn(X)/(phi.marginal_z_1(X) + eps)  # (n, )
    q0 = 1 - q.calculate_p(X)  # (n, )
    q1 = q.calculate_p(X)  # (n, )

    b = np.log((1 + a0*q0)/(1 + a1*q1))
    c = a0*q0/(1 + a0*q0)
    d = a1*q1/(1 + a1*q1)
    e = b + c - d  # (n, )

    q1_grad = q.grad_p_wrt_alpha(X)  # (len(alpha), n)

    correct_shape = (q.alpha_shape[0], X.shape[0])
    assert q1_grad.shape == correct_shape, 'Expected grad ' \
        'to be shape {}, got {} instead'.format(correct_shape,
                                                q1_grad.shape)

    return q1_grad*e


# noinspection PyPep8Naming
def _r0(U, phi, q, pn, eps):
    """
    :param U: array (n, 1)
        data
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param pn: Distribution
    :return: array (n, )

    """
    phi0 = phi.marginal_z_0(U)  # (n, )
    q0 = 1 - q.calculate_p(U)  # (n, )

    return phi0 / (q0*pn(U) + eps)


# noinspection PyPep8Naming
def _r1(U, phi, q, pn, eps):
    """
    :param U: array (n, 1)
        data
    :param phi: LatentVariableModel
        unnormalised model
    :param q: Distribution
        bernoulli distribution
    :param pn: Distribution
    :return: array (n, )

    """
    phi1 = phi.marginal_z_1(U)  # (n, )
    q1 = q.calculate_p(U)  # (n, )

    return phi1 / (q1*pn(U) + eps)


# noinspection PyPep8Naming
def _psi_1_0(U, phi, q, pn, nu, eps):
    """See docstring of _r0"""
    return 1 + (nu/(_r0(U, phi, q, pn, eps) + eps))


# noinspection PyPep8Naming
def _psi_1_1(U, phi, q, pn, nu, eps):
    """See docstring of _r1"""
    return 1 + (nu/(_r1(U, phi, q, pn, eps) + eps))
