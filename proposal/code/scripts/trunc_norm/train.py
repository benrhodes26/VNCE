import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
code_dir_3 = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dirs = [code_dir, code_dir_2, code_dir_3]
for code_dir in code_dirs:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import numpy as np
import pickle

from copy import deepcopy
from itertools import combinations
from scipy import stats as st

from distribution import MissingDataProductOfTruncNormsPosterior, MissingDataProductOfTruncNormNoise
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from latent_variable_model import MissingDataUnnormalisedTruncNorm
from layers import AffineLayer, ReluLayer, TanhLayer
from models import MultipleLayerModel
from vnce_optimiser import VemOptimiser, SgdEmStep, MonteCarloVnceLoss

# Seed a random number generator
seed = 10102016
rng = np.random.RandomState(seed)
START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--data_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data/',
                    help='Path to directory where data is loaded and saved')
parser.add_argument('--save_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/trunc_norm', help='Path to directory where model will be saved')

parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')

# Data arguments
parser.add_argument('--which_dataset', default='usps_4by4patches', help='options: usps and synthetic')
parser.add_argument('--n', type=int, default=10000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=float, default=1.0, help='ratio of noise to data samples in NCE')
parser.add_argument('--num_gibbs_steps', type=int, default=1000,
                    help='number of gibbs steps used to generate as synthetic dataset (if using synthetic)')

# Model arguments
parser.add_argument('--theta0_path', type=str, default=None, help='path to pre-trained weights')
parser.add_argument('--theta0scale', type=float, default=1.0, help='multiplier on initial weights of RBM.')
parser.add_argument('--true_sd', type=float, default=1.0, help='standard deviation of gaussian rv for synthetic ground truth weights')
parser.add_argument('--d', type=int, default=9, help='dimension of visibles for synthetic dataset')
parser.add_argument('--m', type=int, default=8, help='dimension of hiddens')

# Latent NCE optimisation arguments
parser.add_argument('--loss', type=str, default='MonteCarloVnceLoss', help='loss function class to use. See vnce_optimisers.py for options')
parser.add_argument('--noise', type=str, default='marginal', help='type of noise distribution for latent NCE. Currently, this can be either marginals or chow-liu')
parser.add_argument('--opt_method', type=str, default='SGD', help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter', type=int, default=10, help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=0, help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_em_steps', type=int, default=10000, help='Maximum number of EM steps to perform')
parser.add_argument('--learn_rate', type=float, default=0.05, help='if opt_method=SGD, this is the learning rate used')
parser.add_argument('--batch_size', type=int, default=100, help='if opt_method=SGD, this is the size of a minibatch')
parser.add_argument('--num_batch_per_em_step', type=int, default=1, help='if opt_method=SGD, this is the number of batches per EM step')
parser.add_argument('--num_em_steps_per_save', type=int, default=1, help='Every X EM steps save the current params, loss and time')
parser.add_argument('--num_gibbs_steps_for_adaptive_vnce', type=int, default=1, help='needed when sampling from joint noise distribution in adaptive vnce')
parser.add_argument('--track_loss', dest='track_loss', action='store_true', help='track VNCE loss in E & M steps')
parser.add_argument('--no-track_loss', dest='track_loss', action='store_false')
parser.set_defaults(track_loss=True)

# nce optimisation arguments
parser.add_argument('--nce_opt_method', type=str, default='SGD', help='nce optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter_nce', type=int, default=500, help='number of iterations inside scipy.minimize')
parser.add_argument('--nce_num_epochs', type=int, default=1, help='if nce_opt_method=SGD, this is the number of passes through data set')
parser.add_argument('--nce_learn_rate', type=float, default=0.05, help='if nce_opt_method=SGD, this is the learning rate used')
parser.add_argument('--nce_batch_size', type=int, default=100, help='if nce_opt_method=SGD, this is the size of a minibatch')

# Other arguments
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=True)
parser.add_argument('--random_seed', type=int, default=1083463236, help='seed for np.random.RandomState')

args = parser.parse_args()
save_dir = os.path.join(args.save_dir, args.exp_name, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def compose_layers(p):
    """Return the layers for a multi-layer neural network according to the specified hyperparameters

    PARAMETERS
    ----------------------
    p: dict
        Contains the hyperparameters: num_layers, input_dim, output_dim, hidden_dim, batch_size,
        learning_rate, num_epochs, stats_interval, activation_layer, weights_init, biases_init
    """

    first_layer = [AffineLayer(p['input_dim'], p['hidden_dim'], p['weights_init'], p['biases_init']),
                   p['activation_layer']]

    hidden_layers = []
    for i in range(p['num_layers'] - 2):
        hidden_layers.append(AffineLayer(p['hidden_dim'], p['hidden_dim'], p['weights_init'], p['biases_init']))
        hidden_layers.append(p['activation_layer'])

    final_layer = [AffineLayer(p['hidden_dim'], p['output_dim'], p['weights_init'], p['final_biases_init'])]

    return first_layer + hidden_layers + final_layer


def init_training(args):
    """ """
    # Create variational distribution, which uses a multi-layer nn
    layers = compose_layers(args)
    nn = MultipleLayerModel(layers)
    var_dist = MissingDataProductOfTruncNormsPosterior(nn=nn, data_dim=args.d, rng=args.rng)

    # initialise the model p(x, z)
    model = MissingDataUnnormalisedTruncNorm()
    X_train = model.sample(args.n)
    X_test = model.sample(args.n)
    data_mean = X.mean(axis=0)
    noise = MissingDataProductOfTruncNormNoise(mean=data_mean, cov=data_cov)

    e_step = SgdEmStep()
    loss = MonteCarloVnceLoss()
    optimiser = VemOptimiser()

    return model, var_dist


def main(args):
    # NOTE: I include the output layer when counting num_layers, so num_layers=3 implies 2 hidden layers.
    nn_config = {'num_layers': 3,
                 'hidden_dim': 100,
                 'weights_init': UniformInit(-0.05, 0.05, rng=rng),
                 'biases_init': ConstantInit(0.),
                 'final_biases_init': ConstantVectorInit(np.array([0., 0., -1, -1])),
                 'activation_layer': TanhLayer()}
    args.update(nn_config)
    model, var_dist = init_training(args)

    with open(os.path.join(SAVE_DIR, "{}_config.txt".format(exp_name)), 'w') as f:
        for key, value in nn_config.items():
            f.write("{}: {}\n".format(key, value))

    pickle.dump(model, open(os.path.join(args.save_dir, "model.p"), "wb"))
    pickle.dump(var_dist, open(os.path.join(args.save_dir, "var_dist.p"), "wb"))


if __name__ == "__main__":
    main(args)
