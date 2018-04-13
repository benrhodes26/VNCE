import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np
import pdb

# my code
from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise
from latent_nce_optimiser import LatentNCEOptimiser
from latent_variable_model import RestrictedBoltzmannMachine
from utils import plot_log_likelihood_training_curves

from numpy import random as rnd

rng = rnd.RandomState(1083463236)

n = 1000  # number of datapoints
nz = 1  # number latent samples per datapoint
nu = 1

d = 3  # visible dimension
m = 2  # latent dimension
true_W = np.array([[0.2, 0.1, 0.2],
                  [-1, -0.3, 0.4],
                  [1, -0.5, 0.6],
                  [0, -0.7, 0.8]])
# theta0 = rnd.uniform(-0.1, 0.1, (d+1, m+1))
theta0 = true_W
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

random_init_model = RestrictedBoltzmannMachine(theta0)
model = RestrictedBoltzmannMachine(theta0)
noise = MultivariateBernoulliNoise(X_mean)  # uniform dist over binary vectors i.e p(u) = 2**-d
var_dist = RBMLatentPosterior(theta0)
optimiser = LatentNCEOptimiser(model, noise, var_dist, n, nu=nu, latent_samples_per_datapoint=nz)

cd_model = RestrictedBoltzmannMachine(theta0)
cd_optimiser = CDOptimiser(cd_model)

thetas_after_EM_step, J1s, times = optimiser.fit_using_analytic_q(X,
                                                                  theta0=theta0.reshape(-1),
                                                                  stop_threshold=stop_threshold,
                                                                  max_num_em_steps=max_num_em_steps,
                                                                  ftol=ftol,
                                                                  plot=False,
                                                                  disp=True)


thetas_after_each_minibatch, cd_times = cd_optimiser.fit(X, theta0=theta0.reshape(-1),
                                                         num_gibbs_steps=cd_num_steps,
                                                         learning_rate=cd_learn_rate,
                                                         batch_size=cd_batch_size,
                                                         num_epochs=cd_num_epochs)

plot_log_likelihood_training_curves(X, optimiser, cd_optimiser, true_data_dist, random_init_model)
