import os
import sys
CODE_DIRS = ['~/masters-project/ben-rhodes-masters-project/proposal/code', '/home/s1771906/code']
CODE_DIRS_2 = [d + '/neural_network' for d in CODE_DIRS] + CODE_DIRS
CODE_DIRS_2 = [os.path.expanduser(d) for d in CODE_DIRS_2]
for code_dir in CODE_DIRS_2:
    if code_dir not in sys.path:
        sys.path.append(code_dir)

import numpy as np
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from numpy import random as rnd

from latent_variable_model import MissingDataUnnormalisedTruncNorm, MissingDataUnnormalisedTruncNormSymmetric
from project_statics import *

parser = ArgumentParser(description='Generate datasets for experiments for learning the parameters of a truncated normal from incomplete data with VNCE',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_save_dir', type=str, default=EXPERIMENT_OUTPUTS +'/trunc_norm/data', help='Path to directory where data will be saved')
parser.add_argument('--data_num_datasets', type=int, default=10, help='number of independent train + val sets')
parser.add_argument('--data_sample_size', type=int, default=100, help='number of datapoints (train)')
parser.add_argument('--data_random_seed', type=int, default=346322, help='seed for np.random.RandomState')

args = parser.parse_args()
args.data_rng = rnd.RandomState(args.data_random_seed)


def generate_data(args):
    # initialise ground-truth parameters for data generating distribution
    args.true_mean = np.zeros(args.d, dtype='float64')
    args.data_dist = MissingDataUnnormalisedTruncNormSymmetric(scaling_param=np.array([0.]), mean=args.true_mean, prec_type=args.prec_type, rng=args.data_rng)
    args.theta_true = deepcopy(args.data_dist.theta)

    # simulate data
    args.complete_X_train = args.data_dist.sample(args.data_sample_size)
    args.complete_X_val = args.data_dist.sample(int(args.data_sample_size / 4))  # val set is quarter size of train set


def generate_mask(args):
    X = deepcopy(args.complete_X_train)
    train_miss_mask = args.data_rng.uniform(0, 1, X.shape) < args.frac_missing

    # discard any data points that are all-zero
    args.train_missing_data_mask = train_miss_mask[~np.all(train_miss_mask == 1, axis=1)]
    args.X_train = X[~np.all(train_miss_mask == 1, axis=1)]
    args.n = len(args.X_train)
    print("There are {} remaining data points after discarding".format(args.n))

    X_val = deepcopy(args.complete_X_val)
    args.val_missing_data_mask = args.train_missing_data_mask[:len(X_val)]
    args.X_val = X_val[:len(args.val_missing_data_mask)]

    # calculate the sample mean and variance of non-missing data
    observed_train_data = args.X_train * (1 - args.train_missing_data_mask)
    masked_train = np.ma.masked_equal(observed_train_data, 0)
    observed_val_data = args.X_val * (1 - args.val_missing_data_mask)
    masked_val = np.ma.masked_equal(observed_val_data, 0)

    args.X_train_sample_mean = np.array(masked_train.mean(0))
    args.X_train_sample_diag_var = np.array(masked_train.var(0))  # ignore covariances
    args.X_val_sample_mean = np.array(masked_val.mean(0))
    args.X_val_sample_diag_var = np.array(masked_val.var(0))  # ignore covariances


def main(args):
    dimensions = [20, 50]
    prec_types = ['circular', 'hub']
    # sim_nums = np.arange(args.data_num_datasets)
    sim_nums = np.arange(100, 100 + args.data_num_datasets)  # arbitrary numbers for small (n = 50) datasets
    fracs = np.arange(10) / 10
    for d in dimensions:
        args.d = d
        for prec_type in prec_types:
            args.prec_type = prec_type
            for i in sim_nums:
                generate_data(args)
                for frac in fracs:
                    args.frac_missing = frac
                    generate_mask(args)

                    save = os.path.join(args.data_save_dir, "dim{}".format(args.d), prec_type, "{}".format(i), "frac{}".format(frac))
                    if not os.path.exists(save):
                        os.makedirs(save, exist_ok=True)
                    pickle.dump(args, open(os.path.join(save, "data.p"), "wb"))


if __name__ == "__main__":
    main(args)
