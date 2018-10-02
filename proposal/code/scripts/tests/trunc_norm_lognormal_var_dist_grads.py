import os
import sys
CODE_DIRS = ['~/masters-project/ben-rhodes-masters-project/proposal/code', '/home/s1771906/code']
CODE_DIRS_2 = [d + '/neural_network' for d in CODE_DIRS] + CODE_DIRS
CODE_DIRS_2 = [os.path.expanduser(d) for d in CODE_DIRS_2]
for code_dir in CODE_DIRS_2:
    if code_dir not in sys.path:
        sys.path.append(code_dir)
import numpy as np

from copy import deepcopy
from numpy import random as rnd
from scipy.optimize import check_grad
from scipy.stats import norm, multivariate_normal

from data_provider import DataProvider
from distribution import MissingDataLogNormalPosterior, MissingDataProductOfTruncNormNoise
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm, MissingDataUnnormalisedTruncNormSymmetric
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, MonteCarloVnceLoss


def make_model(d):
    # init the model p(x, z)
    scale0 = np.array([0.])
    # mean0 = np.ones(d, dtype='float64') * 1.2
    # a = np.diag(np.ones(d)) * 1.2
    mean0 = np.ones(d, dtype='float64') * rnd.uniform(0.5, 4)
    a = np.diag(np.ones(d)) * rnd.uniform(1, 4)
    b = np.diag(rnd.uniform(0.3, 0.5, d - 1), 1)
    prec0 = a + b + b.T
    # model = MissingDataUnnormalisedTruncNorm(scaling_param=scale0, mean=mean0, precision=prec0)
    model = MissingDataUnnormalisedTruncNormSymmetric(scaling_param=scale0, mean=mean0, precision=prec0)
    print('model mean: {}'.format(mean0))
    print('model prec diag: {}'.format(a))

    return model, scale0, mean0, prec0


def make_var_dist(d):
    # init the variational distribution
    # var_mean0 = np.ones(d) * 1.2
    # a = np.diag(np.ones(d)) * 1.2
    var_mean0 = np.ones(d) * rnd.uniform(0, 2)
    a = np.diag(np.ones(d)) * rnd.uniform(1, 5)
    print('var_mean0: {}'.format(var_mean0))
    print('var prec diag: {}'.format(a))
    b = np.diag(rnd.uniform(0.3, 0.5, d - 1), 1)
    var_prec0 = a + b + b.T
    var_dist = MissingDataLogNormalPosterior(mean=var_mean0, precision=var_prec0)
    return var_dist


def make_noise(n, nu, d, X_mask):
    noise_mean = np.ones(d, dtype='float64') * rnd.uniform(0.5, 4)
    noise_chol = np.ones(d, dtype='float64') * rnd.uniform(1, 2)
    print('noise parameters: \n mean: {} \n chol: {}'.format(noise_mean, noise_chol))

    # make noise distribution and noise samples for vnce
    noise_dist = MissingDataProductOfTruncNormNoise(mean=noise_mean, chol=noise_chol)
    Y = noise_dist.sample(int(n * nu))
    noise_miss_mask = np.repeat(X_mask, nu, axis=0)

    return noise_dist, Y, noise_miss_mask


def make_dp(X_train, Y, n, noise_miss_mask, X_mask, nu, nz, var_dist):
    data_provider = DataProvider(train_data=X_train,
                                 val_data=X_train,
                                 noise_samples=Y,
                                 noise_to_data_ratio=nu,
                                 num_latent_per_data=nz,
                                 variational_dist=var_dist,
                                 train_missing_data_mask=X_mask,
                                 val_missing_data_mask=X_mask,
                                 noise_miss_mask=noise_miss_mask,
                                 use_cdi=False,
                                 use_minibatches=True,
                                 batch_size=n,
                                 use_reparam_trick=True)
    return data_provider


def generate_data(n, d):
    true_mean = np.ones(d, dtype='float64') * 1.
    true_prec = np.identity(d) * 0.5  # choleksy of precision
    data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=true_mean, precision=true_prec)

    X_train = data_dist.sample(n)  # (n, d)
    old_X_mask = rnd.uniform(0, 1, X_train.shape) < 0.5
    X_mask = old_X_mask[~np.all(old_X_mask == 1, axis=1)].astype('int')
    X_train = X_train[~np.all(old_X_mask == 1, axis=1)]
    # X_mask = np.array([[1, 1, 0]])
    n = len(X_train)
    print("num data points remaining: {}".format(n))

    return X_train, X_mask, n


def make_loss(data_provider, model, nu, use_numeric_stable_approx_second_term, var_dist, noise_dist):
    vnce_loss = MonteCarloVnceLoss(data_provider=data_provider,
                                   model=model,
                                   noise=noise_dist,
                                   variational_dist=var_dist,
                                   noise_to_data_ratio=nu,
                                   use_neural_model=False,
                                   use_neural_variational_dist=False,
                                   use_minibatches=True,
                                   use_reparam_trick=True,
                                   separate_terms=True,
                                   use_numeric_stable_approx_second_term=use_numeric_stable_approx_second_term)
    return vnce_loss


def check_model_wrt_alpha_grad(alph, inds, model, var_dist, X_train, X_mask, E, nz, eps):

    def get_alpha_from_subset(a_subset):
        alpha = deepcopy(var_dist.alpha)
        alpha[inds] = a_subset
        return alpha

    def eval_log_model_wrt_alpha(alpha_subset):
        alpha = get_alpha_from_subset(alpha_subset)
        var_dist.alpha = deepcopy(alpha)

        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask)
        val = model(X_train, Z_from_E, X_mask, log=True)  # (nz, n)
        return np.mean(val)

    def grad_log_model_wrt_alpha(alpha_subset):
        alpha = get_alpha_from_subset(alpha_subset)
        var_dist.alpha = deepcopy(alpha)

        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask)
        grad_logmodel_wrt_z = model.grad_log_wrt_z(X_train, Z_from_E, X_mask)
        grad_log_model = var_dist.grad_log_model_wrt_alpha(X_train, E, grad_logmodel_wrt_z, X_mask)  # (len(alpha), nz, n)
        grad = np.mean(grad_log_model, axis=(1, 2))  # (len(alpha), )
        return grad[inds]

    diff = check_grad(eval_log_model_wrt_alpha, grad_log_model_wrt_alpha, alph, epsilon=eps)
    print('{} grad finite diff: {}'.format(inds, diff))


def check_var_dist_wrt_alpha_grad(alph, inds, model, var_dist, X_train, X_mask, E, nz, eps):

    def get_alpha_from_subset(a_subset):
        alpha = deepcopy(var_dist.alpha)
        alpha[inds] = a_subset
        return alpha

    def eval_log_var_dist_wrt_alpha(alpha_subset):
        alpha = get_alpha_from_subset(alpha_subset)
        var_dist.alpha = deepcopy(alpha)

        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask, alpha)
        val = var_dist(Z_from_E, X_train, log=True)  # (n, nz)
        return np.mean(val)

    def grad_log_var_dist_wrt_alpha(alpha_subset):
        alpha = get_alpha_from_subset(alpha_subset)
        var_dist.alpha = deepcopy(alpha)

        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask)
        grad_logmodel_wrt_z = model.grad_log_wrt_z(X_train, Z_from_E, X_mask)
        grad_log_var_dist = var_dist.grad_log_wrt_alpha(X_train, E, grad_logmodel_wrt_z, X_mask)
        grad = np.mean(grad_log_var_dist, axis=(1, 2))  # (len(alpha), )
        return grad[inds]

    diff = check_grad(eval_log_var_dist_wrt_alpha, grad_log_var_dist_wrt_alpha, alph, epsilon=eps)
    print('{} grad finite diff: {}'.format(inds, diff))


def check_log_model_wrt_z_grad(z0, model, X_train, X_mask):

    def get_args(z0):
        Z = z0.reshape((1, 1) + z0.shape)
        return Z

    def eval_log_model_wrt_z(z0):
        Z = get_args(z0)
        val = model(X_train, Z, X_mask, log=True)  # (nz, n)
        return np.sum(val)

    def grad_log_var_dist_wrt_z(z0):
        Z = get_args(z0)
        grad_logmodel_wrt_z = model.grad_log_wrt_z(X_train, Z, X_mask)[0][0]
        grad = np.zeros(d, dtype='float64')
        i_miss = np.nonzero(X_mask[0])
        grad[i_miss] = grad_logmodel_wrt_z
        return grad

    diff = check_grad(eval_log_model_wrt_z, grad_log_var_dist_wrt_z, z0)
    print('log_model w.r.t z grad finite diff: {}'.format(diff))


def check_model_grad(theta0, model, X_train, X_mask, Z, d, check=None):
    if check == 'scale':
        ind = slice(1)
    elif check == 'mean':
        ind = slice(1, 1+d)
    elif check == 'chol':
        ind = slice(1+d, 1+d+len(theta0))
    else:
        ind = slice(-1)

    def eval_log_model(theta):
        model.theta[ind] = deepcopy(theta)
        return np.mean(model(X_train, Z, X_mask, log=True))

    def grad_log_model(theta):
        model.theta[ind] = deepcopy(theta)
        return np.mean(model.grad_log_wrt_params(X_train, Z, X_mask)[ind], axis=(1, 2))

    print('{} grad finite diff: {}'.format(check, check_grad(eval_log_model, grad_log_model, theta0)))


def check_vnce_loss_grad_wrt_theta(theta0, loss, term=1):
    loss.dp.next_minibatch()
    loss.dp.resample_latents_if_necessary()

    def eval_loss(theta):
        loss.model.theta = deepcopy(theta)
        term1, term2 = loss()
        if term == 1:
            return term1
        elif term == 2:
            return term2
        else:
            print('need to specify term = 1 or term = 2!')
            raise ValueError

    def grad_loss_wrt_theta(theta):
        loss.model.theta = deepcopy(theta)
        if term == 1:
            grad = loss.first_term_of_grad_wrt_theta()
        else:
            grad = loss.second_term_grad_wrt_theta()
        return grad

    diff = check_grad(eval_loss, grad_loss_wrt_theta, theta0)
    print('grad finite diff: {}'.format(diff))


def check_vnce_loss_grad_wrt_alpha(alpha0, loss, nz):
    loss.dp.next_minibatch()
    loss.dp.E_ZX = loss.variational_dist.sample_E(nz, loss.dp.X_mask)

    def eval_loss_wrt_alpha(alpha):
        loss.variational_dist.alpha = deepcopy(alpha)
        loss.dp.ZX = loss.variational_dist.get_Z_samples_from_E(nz, loss.dp.E_ZX, loss.dp.X, loss.dp.X_mask)
        term1, term2 = loss()
        return term1

    def grad_loss_wrt_alpha(alpha):
        loss.variational_dist.alpha = deepcopy(alpha)
        loss.dp.ZX = loss.variational_dist.get_Z_samples_from_E(nz, loss.dp.E_ZX, loss.dp.X, loss.dp.X_mask)
        grad = loss.grad_wrt_alpha()
        return grad

    diff = check_grad(eval_loss_wrt_alpha, grad_loss_wrt_alpha, alpha0)
    print('grad finite diff: {}'.format(diff))


def main():
    # eps = np.sqrt(np.finfo(float).eps)
    eps = 1e-8
    n = 100
    nu = 5
    nz = 5
    d = 5
    use_numeric_stable_approx_second_term = False

    # generate synthetic data
    X_train, X_mask, n = generate_data(n, d)

    model, scale0, mean0, prec0 = make_model(d)
    var_dist = make_var_dist(d)
    noise_dist, Y, noise_miss_mask = make_noise(n, nu, d, X_mask)
    alpha0 = deepcopy(var_dist.alpha)
    E = var_dist.sample_E(nz, X_mask)
    Z = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask)

    data_provider = make_dp(X_train, Y, n, noise_miss_mask, X_mask, nu, nz, var_dist)
    loss = make_loss(data_provider, model, nu, use_numeric_stable_approx_second_term, var_dist, noise_dist)

    print('-----------------------------------')
    print("Checking model grads w.r.t alpha")
    print('-----------------------------------')
    # check_model_wrt_alpha_grad(alpha0, np.arange(len(alpha0)))
    # check_model_wrt_alpha_grad(alpha0[:1], [0])
    # check_model_wrt_alpha_grad(alpha0[1:2], [1])
    # check_model_wrt_alpha_grad(alpha0[2:3], [2])
    # check_model_wrt_alpha_grad(alpha0[3:4], [3], model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[4:5], [4], model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[5:6], [5], model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[6:7], [6], model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[7:8], [7], model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[8:9], [8], model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[:d], np.arange(d), model, var_dist, X_train, X_mask, E, nz, eps)
    # check_model_wrt_alpha_grad(alpha0[d:], np.arange(d, len(alpha0)), model, var_dist, X_train, X_mask, E, nz, eps)

    print('-----------------------------------')
    print("Checking Var dist grads w.r.t alpha")
    print('-----------------------------------')
    # check_var_dist_wrt_alpha_grad(alpha0, np.arange(len(alpha0)))
    # check_var_dist_wrt_alpha_grad(alpha0[0:1], [0])
    # check_var_dist_wrt_alpha_grad(alpha0[1:2], [1])
    # check_var_dist_wrt_alpha_grad(alpha0[2:3], [2])
    # check_var_dist_wrt_alpha_grad(alpha0[[3, 4, 5]], [3, 4, 5])
    # check_var_dist_wrt_alpha_grad(alpha0[[6, 7]], [6, 7])
    # check_var_dist_wrt_alpha_grad(alpha0[8:], [8])
    # check_var_dist_wrt_alpha_grad(alpha0[:d], np.arange(d), model, var_dist, X_train, X_mask, E, nz, eps)
    # check_var_dist_wrt_alpha_grad(alpha0[d:], np.arange(d, len(alpha0)), model, var_dist, X_train, X_mask, E, nz, eps)

    # print("Checking log_model w.r.t z grads")
    # # z0 = rnd.uniform(0, 1, d) < 0.5
    # z0 = np.array([1, 1, 0])
    # check_log_model_wrt_z_grad(z0)

    print('-----------------------------------')
    print('checking grad of model w.r.t theta')
    print('-----------------------------------')
    check_model_grad(deepcopy(scale0), model, X_train, X_mask, Z, d, 'scale')
    check_model_grad(deepcopy(mean0) * rnd.uniform(-0.5, 2, mean0.shape), model, X_train, X_mask, Z, d, 'mean')
    check_model_grad(deepcopy(model.theta[1 + d:]), model, X_train, X_mask, Z, d, 'chol')

    print('-----------------------------------')
    print('checking grad of vnce objective w.r.t theta')
    print('-----------------------------------')
    check_vnce_loss_grad_wrt_theta(deepcopy(model.theta), loss, term=1)
    check_vnce_loss_grad_wrt_theta(deepcopy(model.theta), loss, term=2)

    print('-----------------------------------')
    print('checking grad of vnce objective w.r.t alpha')
    print('-----------------------------------')
    check_vnce_loss_grad_wrt_alpha(deepcopy(var_dist.alpha), loss, nz)


if __name__ == '__main__':
    main()
