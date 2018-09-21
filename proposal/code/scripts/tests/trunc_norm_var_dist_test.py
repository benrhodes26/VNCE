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

from distribution import MissingDataProductOfTruncNormsPosterior, MissingDataProductOfTruncNormNoise
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, MonteCarloVnceLoss

n = 10
nz = 2
d = 4
len_alpha = d * (d + 3) / 2

true_mean = np.ones(d) * 1.2
# true_chol = np.identity(d) * 0.1  # choleksy of precision
# data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=true_mean, chol=true_chol)
true_prec = np.identity(d) * 0.1  # choleksy of precision
data_dist = MissingDataUnnormalisedTruncNorm(scaling_param=np.array([0.]), mean=true_mean, precision=true_prec)

# generate synthetic data
X_train = data_dist.sample(n)  # (n, d)
# X_mask = rnd.uniform(0, 1, X_train.shape) < 0.5
X_mask = np.zeros((n, d))
X_mask[:, 0] = 1
Z = np.ones((nz, ) + X_train.shape)
Z *= X_mask * 1.5

# init the model p(x, z)
scale0 = np.array([0.])
mean0 = np.ones(d) * 5.
# chol0 = np.identity(d)  # cholesky of precision
# chol0 = np.array([[1, 0], [-1, 1]])
# model = MissingDataUnnormalisedTruncNorm(scaling_param=scale0, mean=mean0, chol=chol0)
a = np.diag(np.ones(d)) * rnd.uniform(1, 4)
b = np.diag(rnd.uniform(0.3, 0.5, d - 1), 1)
prec0 = a + b + b.T
model = MissingDataUnnormalisedTruncNorm(scaling_param=scale0, mean=mean0, precision=prec0)

# init the variational distribution
var_dist = MissingDataProductOfTruncNormsPosterior(nn=None, data_dim=d)
E = var_dist.sample_E(nz, np.ones_like(X_train))

nn_output = rnd.uniform(0, 1, (n, 2 * d))
nn_output = np.zeros((n, 2*d))

def check_model_grad(theta0, check=None):
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


def check_model_wrt_nn_out_grad(nn_out, check=None):

    def get_nn_out(nn_out):
        nn_out = nn_out.reshape(n, -1)
        new_nn_out = nn_output
        # new_nn_out = np.zeros((n, 2 * d))
        if check == 'mean':
            new_nn_out[:, :d] = nn_out
        elif check == 'chol':
            new_nn_out[:, d:] = nn_out
        else:
            new_nn_out = nn_out
        return new_nn_out

    def eval_log_model_wrt_nn_out(nn_out):
        nn_out = get_nn_out(nn_out)
        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask, nn_out)
        val = model(X_train, Z_from_E, X_mask, log=True)
        return np.sum(np.mean(val, axis=0))


    def grad_log_model_wrt_nn_out(nn_out):
        nn_out = get_nn_out(nn_out)
        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask, nn_out)
        grad_z_wrt_nn_out = var_dist.grad_of_Z_wrt_nn_outputs(nn_out, E)
        grad = model.grad_log_wrt_nn_outputs(X_train, Z_from_E, grad_z_wrt_nn_out, X_mask)
        grad = np.mean(grad, axis=1).T  # (n, 2*d)

        if check == 'mean':
            grad = grad[:, :d]
        elif check == 'chol':
            grad = grad[:, d:]
        return grad.reshape(-1)

    print('{} grad finite diff: {}'.format(check, check_grad(eval_log_model_wrt_nn_out, grad_log_model_wrt_nn_out, nn_out.reshape(-1))))


def check_var_wrt_nn_out_grad(nn_out, check=None):

    def get_nn_out(nn_out):
        nn_out = nn_out.reshape(n, -1)
        new_nn_out = nn_output
        if check == 'mean':
            new_nn_out[:, :d] = nn_out
        elif check == 'chol':
            new_nn_out[:, d:] = nn_out
        else:
            new_nn_out = nn_out
        return new_nn_out

    def eval_log_var(nn_out):
        nn_out = get_nn_out(nn_out)
        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask, nn_out)
        val = var_dist(Z_from_E, X_train, log=True, nn_outputs=nn_out)
        return np.sum(np.mean(val, axis=0))

    def grad_log_var(nn_out):
        nn_out = get_nn_out(nn_out)
        Z_from_E = var_dist.get_Z_samples_from_E(nz, E, X_train, X_mask, nn_out)
        grad_z_wrt_nn_out = var_dist.grad_of_Z_wrt_nn_outputs(nn_out, E)
        grad = var_dist.grad_log_wrt_nn_outputs(nn_out, grad_z_wrt_nn_out, Z_from_E)
        grad = np.mean(grad, axis=1).T  # (n, 2*d)

        if check == 'mean':
            grad = grad[:, :d]
        elif check == 'chol':
            grad = grad[:, d:]
        return grad.reshape(-1)

    print('{} grad finite diff: {}'.format(check, check_grad(eval_log_var, grad_log_var, nn_out.reshape(-1))))


check_model_grad(deepcopy(scale0), 'scale')
check_model_grad(deepcopy(mean0) * rnd.uniform(-0.5, 2, mean0.shape), 'mean')
check_model_grad(deepcopy(model.theta[1+d:]), 'chol')

# check_model_wrt_nn_out_grad(np.zeros((n, 2 * d)))
check_model_wrt_nn_out_grad(rnd.uniform(0, 1, (n, d)), check='mean')
check_model_wrt_nn_out_grad(rnd.uniform(0, 1, (n, d)), check='chol')

# check_var_wrt_nn_out_grad(np.zeros((n, 2 * d)))
check_var_wrt_nn_out_grad(rnd.uniform(0, 1, (n, d)), check='mean')
check_var_wrt_nn_out_grad(rnd.uniform(0, 1, (n, d)), check='chol')
