import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np

# my code
from distribution import RBMLatentPosterior,
from latent_nce_optimiser import LatentNCEOptimiser
from latent_variable_model import RestrictedBoltzmannMachine

from numpy import random as rnd

def test_rbm_model(model):
    W = model.theta.reshape(3, 2)
    hidden_vals = [[[[0]]], [[[1]]]]
    visible_vals = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
    answers = [1, np.exp(1), np.exp(1), np.exp(1),
               np.exp()]

    for z in hidden_vals:
        for u in visible_vals:
            model_val = model(u, z)


def test_rbm_log_grads():
    hidden_vals = [0, 1]
    visible_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]

def test_rbm_posterior():
    raise NotImplementedError