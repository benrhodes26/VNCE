import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
code_dir_3 = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code/neural_network'
code_dirs = [code_dir, code_dir_2, code_dir_3]
for code_dir in code_dirs:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle

from copy import deepcopy
from itertools import combinations
from data_providers import StarsAndMoonsDataProvider
from layers import AffineLayer, ReluLayer, TanhLayer
from errors import FreeEnergyLoss, VnceLoss
from models import MultipleLayerModel
from initialisers import GlorotUniformInit, ConstantInit, UniformInit, ConstantVectorInit
from learning_rules import GradientDescentLearningRule
from optimisers import Optimiser
from scipy import stats as st

from latent_variable_model import StarsAndMoonsModel
from distribution import StarsAndMoonsPosterior, GaussianNoise

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

SAVE_DIR = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project-non-code/experiments/stars-and-moons/'

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]


def plot_training_stats(stats, keys, stats_interval):
        # Plot the change in the validation and training set error over training.
        fig = plt.figure(figsize=(12, 8))

        ax_1 = fig.add_subplot(221)
        for k in ['error(train)', 'error(valid)']:
            ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                      stats[1:, keys[k]], label=k)
        ax_1.legend(loc=0)
        ax_1.set_xlabel('Epoch number')

        fig.savefig(SAVE_DIR + '/figs/' 'free-energy-training-curve.pdf')
        plt.show()
        
        return fig


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


def train(train_data, valid_data, config, log=True, plot=True):
    """Specify a feedforward neural network architecture, train it and plot training statistics
    
    PARAMETERS
    ----------------------
    params: dict
        Contains the following hyperparameters: num_layers, input_dim, output_dim
        hidden_dim, batch_size, learning_rate, num_epochs, stats_interval                  
    """
    # Reset random number generator and data provider states on each run to ensure reproducibility of results
    rng.seed(seed)
    train_data.reset()
    valid_data.reset()

    # Alter data-provider batch size
    train_data.batch_size = config['batch_size']
    valid_data.batch_size = config['batch_size']

    # Create multi-layer nn
    layers = compose_layers(config)
    nn = MultipleLayerModel(layers)

    # initialise the model p(x, z) and the variational distribution
    model = StarsAndMoonsModel(config['c'], truncate_gaussian=config['truncate_gaussian'], rng=rng)
    var_dist = StarsAndMoonsPosterior(nn=nn, rng=rng)

    if config['loss'] == 'VnceLoss':
        if config['noise'] == 'good_noise':
            X = (train_data.inputs * train_data.std) + train_data.mean
            data_mean = X.mean(axis=0)
            data_cov = np.dot((X-data_mean).T, X-data_mean) / len(X)
            noise = GaussianNoise(mean=data_mean, cov=data_cov)
        elif config['noise'] == 'bad_noise':
            noise = GaussianNoise(mean=np.ones(2), cov=30*np.identity(2))
        pickle.dump(noise, open(os.path.join(SAVE_DIR, "{}.p".format(config['noise'])), "wb"))
        error = VnceLoss(model=model,
                         noise=noise,
                         var_dist=var_dist,
                         num_z_per_x=config['nz'],
                         noise_to_data_ratio=config['nu'],
                         data_mean=train_data.mean,
                         data_std=train_data.std)

    elif config['loss'] == 'FreeEnergyLoss':
        error = FreeEnergyLoss(model=model, var_dist=var_dist, num_z_per_x=config['nz'], data_mean=train_data.mean, data_std=train_data.std)

    # Use a basic gradient descent learning rule
    learning_rule = GradientDescentLearningRule(learning_rate=config['learning_rate'])

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(nn, error, learning_rule, train_data, valid_data, data_monitors={})

    # Train nn, tracking speed, error & accuracy per epoch and plotting these after training
    logger.handlers = [logging.StreamHandler()] if log else [logging.NullHandler()]

    # Run the optimiser for 5 epochs (full passes through the training set) printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=config['num_epochs'], stats_interval=config['stats_interval'])

    if plot:
        _ = plot_training_stats(stats, keys, config['stats_interval'])
    
    return model, var_dist


def main():
    # NOTE: I include the output layer when counting num_layers, so num_layers=3 implies 2 hidden layers.
    config = {'loss': 'VnceLoss',
              'truncate_gaussian': False,
              'num_layers': 3,
              'input_dim': 2,
              'output_dim': 4,
              'hidden_dim': 100,
              'learning_rate': 0.01,
              'num_data': 10000,
              'batch_size': 100,
              'num_epochs': 100,
              'stats_interval': 1,
              'weights_init': UniformInit(-0.05, 0.05, rng=rng),
              'biases_init': ConstantInit(0.),
              'final_biases_init': ConstantVectorInit(np.array([0., 0., -1, -1])),
              'activation_layer': TanhLayer(),
              'c': 0.3,
              'nz': 1,
              'noise': 'bad_noise',
              'nu': 10}
    if config['noise']:
        exp_name = config['loss'] + '_' + 'truncate_gaussian=' + str(config['truncate_gaussian']) + '_' + config['noise'] + '_' + 'nu' + str(config['nu'])
    else:
        exp_name = config['loss'] + '_' + 'truncate_gaussian=' + str(config['truncate_gaussian'])

    train_data = StarsAndMoonsDataProvider(config['c'],
                                           'train',
                                           size=config['num_data'],
                                           batch_size=config['batch_size'],
                                           truncate_gaussian=config['truncate_gaussian'],
                                           rng=rng)
    valid_data = StarsAndMoonsDataProvider(config['c'],
                                           'valid',
                                           batch_size=config['batch_size'],
                                           truncate_gaussian=config['truncate_gaussian'],
                                           rng=rng)

    model, var_dist = train(train_data, valid_data, config, log=False, plot=True)

    with open(os.path.join(SAVE_DIR, "{}_config.txt".format(exp_name)), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))

    pickle.dump(model, open(os.path.join(SAVE_DIR, "truncate={}_model.p".format(str(config['truncate_gaussian']))), "wb"))
    pickle.dump(var_dist, open(os.path.join(SAVE_DIR, "{}_var_dist.p".format(exp_name)), "wb"))


if __name__ == "__main__":
    main()
