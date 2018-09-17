import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_3 = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dir_4 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dirs = [code_dir, code_dir_2, code_dir_3, code_dir_4]
for code_dir in code_dirs:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import numpy as np

from copy import deepcopy
from numpy import random as rnd
from scipy.optimize import check_grad
from scipy.stats import norm, multivariate_normal

from distribution import MissingDataLogNormal, MissingDataProductOfTruncNormNoise
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, MonteCarloVnceLoss

n = 2
nz = 3
d = 5
len_alpha = int(d * (d + 3) / 2)

true_mean = np.ones(d, dtype='float64') * 1.
true_prec = np.identity(d) * 0.5  # choleksy of precision
data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=true_mean, precision=true_prec)

# generate synthetic data
X_train = data_dist.sample(n)  # (n, d)
# old_X_mask = rnd.uniform(0, 1, X_train.shape) < 0.5
# X_mask = old_X_mask[~np.all(old_X_mask == 1, axis=1)]
# X_train = X_train[~np.all(old_X_mask == 1, axis=1)]
# n = len(X_train)
# print("num data points remaining: {}".format(n))
X_mask = np.zeros((n, d), dtype='float64')
X_mask[:int(d/2), 0] = 1
X_mask[:int(d/2), 1] = 1
X_mask[int(d/2):, 2] = 1
X_mask[int(d/2):, 4] = 1

# init the model p(x, z)
scale0 = np.array([0.])
mean0 = np.ones(d, dtype='float64') * 2.
a = np.diag(np.ones(d))
b = np.diag(rnd.uniform(0.3, 0.5, d - 1), 1)
prec0 = a + b + b.T
model = MissingDataUnnormalisedTruncNorm(scaling_param=scale0, mean=mean0, precision=prec0)

# init the variational distribution
var_mean0 = np.ones(d) * 0.5
a = np.diag(np.ones(d)) * 1.2
b = np.diag(rnd.uniform(0.3, 0.5, d - 1), 1)
var_prec0 = a + b + b.T
var_dist = MissingDataLogNormal(mean=var_mean0, precision=var_prec0)
alpha0 = deepcopy(var_dist.alpha)
E = var_dist.sample_E(nz, X_mask)


def check_model_wrt_alpha_grad(alph, inds):

    def get_alpha_from_subset(a_subset):
        alpha = deepcopy(alpha0)
        alpha[inds] = a_subset
        return alpha

    def eval_log_model_wrt_alpha(alpha_subset):
        alpha = get_alpha_from_subset(alpha_subset)
        var_dist.alpha = deepcopy(alpha)

        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask)
        # Z_from_E = var_dist._test_get_Z_samples_from_E(1, E, X_train, X_mask)
        val = model(X_train, Z_from_E, X_mask, log=True)  # (nz, n)
        return np.mean(val)

    def grad_log_model_wrt_alpha(alpha_subset):
        alpha = get_alpha_from_subset(alpha_subset)
        var_dist.alpha = deepcopy(alpha)

        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask)
        # Z_from_E = var_dist._test_get_Z_samples_from_E(1, E, X_train, X_mask)
        grad_logmodel_wrt_z = model.grad_log_wrt_z(X_train, Z_from_E, X_mask)
        grad_log_model = var_dist.grad_log_model_wrt_alpha(X_train, E, grad_logmodel_wrt_z, X_mask)  # (len(alpha), nz, n)
        # grad_log_model = var_dist._test_grad_wrt_alpha(X_train, E, grad_logmodel_wrt_z, X_mask)  # (len(alpha), nz, n)
        grad = np.mean(grad_log_model, axis=(1,2))  # (len(alpha), )
        return grad[inds]

    diff = check_grad(eval_log_model_wrt_alpha, grad_log_model_wrt_alpha, alph)
    print('{} grad finite diff: {}'.format(inds, diff))


def check_var_dist_wrt_alpha_grad(alph, inds):

    def get_alpha_from_subset(a_subset):
        alpha = deepcopy(alpha0)
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
        grad = np.mean(grad_log_var_dist, axis=(1,2))  # (len(alpha), )
        return grad[inds]

    print('{} grad finite diff: {}'.format(inds, check_grad(eval_log_var_dist_wrt_alpha, grad_log_var_dist_wrt_alpha, alph)))


def check_log_model_wrt_z_grad(z0):

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

print("Checking model grads")
# check_model_wrt_alpha_grad(alpha0, np.arange(len(alpha0)))
# check_model_wrt_alpha_grad(alpha0[:1], [0])
# check_model_wrt_alpha_grad(alpha0[1:2], [1])
# check_model_wrt_alpha_grad(alpha0[2:3], [2])
# check_model_wrt_alpha_grad(alpha0[3:4], [3])
# check_model_wrt_alpha_grad(alpha0[4:5], [4])
# check_model_wrt_alpha_grad(alpha0[5:6], [5])
# check_model_wrt_alpha_grad(alpha0[[3, 4, 5]], [3, 4, 5])
# check_model_wrt_alpha_grad(alpha0[[6, 7]], [6, 7])
# check_model_wrt_alpha_grad(alpha0[8:], [8])
check_model_wrt_alpha_grad(alpha0[:d], np.arange(d))
check_model_wrt_alpha_grad(alpha0[d:], np.arange(d, len(alpha0)))

# print("Checking Var dist grads")
# check_var_dist_wrt_alpha_grad(alpha0, np.arange(len(alpha0)))
# check_var_dist_wrt_alpha_grad(alpha0[0:1], [0])
# check_var_dist_wrt_alpha_grad(alpha0[1:2], [1])
# check_var_dist_wrt_alpha_grad(alpha0[2:3], [2])
# check_var_dist_wrt_alpha_grad(alpha0[[3, 4, 5]], [3, 4, 5])
# check_var_dist_wrt_alpha_grad(alpha0[[6, 7]], [6, 7])
# check_var_dist_wrt_alpha_grad(alpha0[8:], [8])
check_var_dist_wrt_alpha_grad(alpha0[:d], np.arange(d))
check_var_dist_wrt_alpha_grad(alpha0[d:], np.arange(d, len(alpha0)))

# print("Checking log_model w.r.t z grads")
# # z0 = rnd.uniform(0, 1, d) < 0.5
# z0 = np.array([1, 1, 0])
# check_log_model_wrt_z_grad(z0)
