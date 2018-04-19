import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np
import pdb
from copy import deepcopy

# my code
from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_nce_optimiser import LatentNCEOptimiser
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from utils import plot_log_likelihood_training_curves

from numpy import random as rnd

rng = rnd.RandomState(1083463236)

n = 1000  # number of datapoints
nz = 1  # number latent samples per datapoint
nu = 1

d = 2  # visible dimension
m = 1  # latent dimension
true_W = rng.randn(d + 1, m + 1) * 1
true_W[0, 0] = 0

ftol = 1e-4
stop_threshold = 1e-01
max_num_em_steps = 20

# hyperparameters used in CDOptimiser (contrastive divergence optimiser)
cd_num_steps = 1
cd_learn_rate = 0.01
cd_batch_size = 10
cd_num_epochs = 100

true_data_dist = RestrictedBoltzmannMachine(true_W)
X, Z = true_data_dist.sample(n, num_iter=1000)
X_mean = np.mean(X, axis=0)

# random initial weights, that depend on the data
theta0 = np.asarray(
    rng.uniform(
        low=-4 * np.sqrt(6. / (d + m)),
        high=4 * np.sqrt(6. / (d + m)),
        size=(d + 1, m + 1)
    ))
theta0[1:, 0] = np.log(X_mean / (1 - X_mean))  # visible biases
theta0[0, 0] = -m * np.log(2) + np.sum(np.log(1 - X_mean))  # scaling parameter
theta0[0, 1:] = 0  # hidden biases

noise = MultivariateBernoulliNoise(X_mean, rng=rng)
Y = noise.sample(int(n * nu))  # generate noise

model = RestrictedBoltzmannMachine(theta0, rng=rng)
cd_model = RestrictedBoltzmannMachine(theta0, rng=rng)
random_init_model = RestrictedBoltzmannMachine(theta0, rng=rng)
nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)

var_dist = RBMLatentPosterior(theta0, rng=rng)

optimiser = LatentNCEOptimiser(model, noise, noise_samples=Y, variational_dist=var_dist, sample_size=n, nu=nu, latent_samples_per_datapoint=nz, rng=rng)
cd_optimiser = CDOptimiser(cd_model, rng=rng)
nce_optimiser = NCEOptimiser(model=nce_model, noise=noise, noise_samples=Y, sample_size=n, nu=nu)

final_theta = np.array([-3.04247897, -1.30884324, -1.32626975, 3.02416787, 1.88164527, -0.17250398])
for j, nz in enumerate([1, 5, 50, 500, 1000]):
    last_J1 = optimiser.evaluate_J1_at_param(final_theta, X, nz)
    last_J = nce_optimiser.evaluate_J_at_param(final_theta, X)
    print(last_J1 - last_J)

# thetas_after_EM_step, J1s, times = optimiser.fit_using_analytic_q(X,
#                                                                   theta0=theta0.reshape(-1),
#                                                                   stop_threshold=stop_threshold,
#                                                                   max_num_em_steps=max_num_em_steps,
#                                                                   ftol=ftol,
#                                                                   plot=False,
#                                                                   disp=True)
#
#
# thetas_after_each_minibatch, cd_times = cd_optimiser.fit(X, theta0=theta0.reshape(-1),
#                                                          num_gibbs_steps=cd_num_steps,
#                                                          learning_rate=cd_learn_rate,
#                                                          batch_size=cd_batch_size,
#                                                          num_epochs=cd_num_epochs)
#
# plot_log_likelihood_training_curves(X, optimiser, cd_optimiser, true_data_dist, random_init_model)
