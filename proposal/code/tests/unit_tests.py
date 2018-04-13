import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np

# my code
from distribution import RBMLatentPosterior
from latent_nce_optimiser import LatentNCEOptimiser
from latent_variable_model import RestrictedBoltzmannMachine

from numpy import random as rnd

def test_rbm_model(model):

    hidden_vals = [[[[0]]],
                   [[[1]]]]  # (2, 1, 1, 1)
    visible_vals = [[[0, 0]],
                    [[0, 1]],
                    [[1, 0]],
                    [[1, 1]]]  # (4, 1, 2)
    log_answers = [2, 3, 1, 2, 3, 6, 1.5, 4.5]

    all_correct = True
    # 8 different combinations of inputs
    i = 0
    for z in hidden_vals:
        for u in visible_vals:
            model_val = model(np.array(u), np.array(z))[0][0]
            correct = (np.allclose(model_val, np.exp(log_answers[i])))
            all_correct = all_correct and correct
            i += 1

    return all_correct

def test_rbm_log_grads(model):
    visible_vals = np.array([[0, 1],
                            [1, 0]])  # (2 2)
    hidden_vals = np.array([[[0],
                             [1]],

                            [[1],
                             [0]]])  # (2, 2, 1)

    correct_grad = np.array([[1,    0.5],
                             [0.5, 0.25],
                             [0.5, 0.25]]).reshape(-1)
    model_grad = model.grad_log_wrt_params(visible_vals, hidden_vals)  # (6, 2, 2)
    mean_model_grad = np.mean(model_grad, axis=(1, 2))

    return np.allclose(correct_grad, mean_model_grad)

def test_rbm_posterior():
    raise NotImplementedError

if __name__ == '__main__':
    rbm = RestrictedBoltzmannMachine(np.array([[2,    1],
                                              [-1, -0.5],
                                              [1,    2]]))
    if test_rbm_model(rbm):
        print('Passed model definition test!')
    else:
        'Incorrect model definition!'
        raise Exception
    if test_rbm_log_grads(rbm):
        print('Passed model log gradients test!')
    else:
        print('Incorrect log grads!')
        raise Exception
