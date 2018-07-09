import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

import numpy as np
import pickle

# my code
from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, ScipyMinimiseEmStep, ExactEStep, MonteCarloVnceLoss, AdaptiveMonteCarloVnceLoss

from numpy import random as rnd
from scipy.optimize import check_grad

rng = rnd.RandomState(1083463236)

# generate weights of RBM that we want to learn
true_theta = rng.randn(9 + 1, 8 + 1) * 1
true_theta[0, 0] = 0

# generate synthetic training and test sets
true_data_dist = RestrictedBoltzmannMachine(true_theta, rng=rng)
X, Z = true_data_dist.sample(10000, num_iter=1000)
X_mean = np.mean(X, axis=0)

# initialise random weights, that depend on the data
theta0 = np.asarray(
    rng.uniform(
        low=-4 * np.sqrt(6. / (9 + 8)),
        high=4 * np.sqrt(6. / (9 + 8)),
        size=(9 + 1, 8 + 1)
    ))
noise_dist = MultivariateBernoulliNoise(X_mean, rng=rng)
Y = noise_dist.sample(int(10000 * 1))

model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)

nce_optimiser = NCEOptimiser(model=nce_model, noise=noise_dist, noise_samples=Y, nu=1)


def J1(theta):
    nce_optimiser.model.theta = theta
    return nce_optimiser.compute_J(X)


def J1_grad(theta):
    nce_optimiser.model.theta = theta
    return nce_optimiser.compute_J_grad(X)


print(check_grad(J1, J1_grad, theta0.reshape(-1)))
