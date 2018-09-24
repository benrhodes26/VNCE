# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

import numpy as np
import time

from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import random as rnd
from plot import *
from scipy.optimize import minimize, check_grad
from utils import validate_shape, average_log_likelihood, take_closest

DEFAULT_SEED = 1083463236


class DataProvider(object):
    """Data provider for VNCE experiments"""

    def __init__(self,
                 train_data,
                 val_data,
                 noise_samples,
                 noise_to_data_ratio,
                 num_latent_per_data,
                 variational_dist,
                 train_missing_data_mask=None,
                 val_missing_data_mask=None,
                 noise_miss_mask=None,
                 use_cdi=False,
                 use_reparam_trick=False,
                 X_means=None,
                 Y_means=None,
                 use_minibatches=False,
                 batch_size=None,
                 rng=None):
        """Create a new data provider object.
        """
        self.nu = noise_to_data_ratio
        self.nz = num_latent_per_data
        self.variational_dist = variational_dist
        self.use_cdi = use_cdi
        self.use_reparam_trick = use_reparam_trick

        # Masks for when data is missing (1 = missing, 0 = observed)
        self.train_miss_mask = train_missing_data_mask
        self.val_miss_mask = val_missing_data_mask
        self.noise_miss_mask = noise_miss_mask

        # save 'global' (i.e not minibatch) data
        self.set_train_val_and_noise_data(train_data, val_data, noise_samples, use_cdi, X_means)

        # samples from a simple 'base' distribution needed if using reparametrisation trick
        if self.use_reparam_trick:
            self.global_E_ZX = self.variational_dist.sample_E(self.nz, self.train_miss_mask)
            self.global_E_ZX_val = self.variational_dist.sample_E(self.nz, self.val_miss_mask)
        else:
            self.global_E_ZX = None
            self.global_E_ZX_val = None
        self.ZX = None
        self.ZY = None

        self.use_minibatches = use_minibatches
        # initialise the variables that will hold all minibatch data/masks, if using minibatches.
        # Otherwise, initialise these variables to be copies of the global data/masks.
        if self.use_minibatches:
            self.set_data_and_masks(save_current=False)
        else:
            self.set_data_and_masks(which_set='train', save_current=False)

        self.batch_size = batch_size
        if batch_size:
            self.batches_per_epoch = int(len(self.train_data) / self.batch_size)
        self.current_batch_id = 0
        self.current_epoch = 0
        self.current_loss = None
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(DEFAULT_SEED)

    def set_train_val_and_noise_data(self, train_data, val_data, noise_samples, use_cdi, X_means):
        if self.train_miss_mask is not None:
            self.val_data = deepcopy(val_data * (1 - self.val_miss_mask))
            if use_cdi:
                # we are using the CDI algorithm, and so fill in all missing vals with global means
                self.train_data = deepcopy(train_data * (1 - self.train_miss_mask) + X_means * self.train_miss_mask)
                self.noise_samples = deepcopy(noise_samples)  # could use global noise means, but this works too
            else:
                self.train_data = deepcopy(train_data * (1 - self.train_miss_mask))
                self.noise_samples = deepcopy(noise_samples * (1 - self.noise_miss_mask))
        else:
            self.train_data = deepcopy(train_data)
            self.val_data = deepcopy(val_data)
            self.noise_samples = deepcopy(noise_samples)

    def set_data_and_masks(self, which_set=None, save_current=True):
        """Swap in a new set of data, noise and masks (e.g for evaluating on a validation set)"""
        # the following set of attributes will typically contain data/noise/masks etc. for each minibatch.
        # But If we are not using minibatches, then they will just refer to the whole dataset

        self.eval_mode = False  # by default, ensure this off. Turn on if which_set is 'train' or 'val'
        self.resample_from_variational_noise = True
        if save_current:
            self.prev_X = deepcopy(self.X)
            self.prev_X_mask = deepcopy(self.X_mask)
            self.prev_E_ZX = deepcopy(self.E_ZX)
            self.prev_Y = deepcopy(self.Y)
            self.prev_Y_mask = deepcopy(self.Y_mask)

        if which_set == 'train':
            self.eval_mode = True  # required when using the cdi algorithm, to avoid updating global data
            self.X = deepcopy(self.train_data)
            self.X_mask = deepcopy(self.train_miss_mask)
            self.E_ZX = self.global_E_ZX
            self.Y = deepcopy(self.noise_samples)
            self.Y_mask = deepcopy(self.noise_miss_mask)
        elif which_set == 'val':
            self.eval_mode = True  # required when using the cdi algorithm, to avoid updating global data
            num_val_noise = int(self.nu * len(self.val_data))
            self.X = deepcopy(self.val_data)
            self.X_mask = deepcopy(self.val_miss_mask)
            self.E_ZX = self.global_E_ZX_val
            self.Y = deepcopy(self.noise_samples[:num_val_noise])
            self.Y_mask = deepcopy(self.noise_miss_mask[:num_val_noise])
        elif which_set == 'prev':
            self.X = deepcopy(self.prev_X)
            self.X_mask = deepcopy(self.prev_X_mask)
            self.E_ZX = deepcopy(self.prev_E_ZX)
            self.Y = deepcopy(self.prev_Y)
            self.Y_mask = deepcopy(self.prev_Y_mask)
        elif which_set is None:
            self.X = None
            self.X_mask = None
            self.E_ZX = None
            self.Y = None
            self.Y_mask = None
        else:
            print('Must specify a set of data (or explicity state None)')
            raise ValueError

    def next_minibatch(self):
        if self.current_batch_id % self.batches_per_epoch == 0:
            self.new_epoch()
            self.current_batch_id = 0

        batch_start = self.current_batch_id * self.batch_size
        self.batch_slice = slice(batch_start, batch_start + self.batch_size)
        noise_batch_start = int(batch_start * self.nu)
        self.noise_batch_slice = slice(noise_batch_start, noise_batch_start + int(self.batch_size * self.nu))

        self.X = deepcopy(self.train_data[self.batch_slice])
        self.Y = deepcopy(self.noise_samples[self.noise_batch_slice])

        if self.train_miss_mask is not None:
            self.set_masks_for_batch()

        if self.use_reparam_trick:
            self.E_ZX = self.get_minibatch_E()

        self.current_batch_id += 1
        self.resample_from_variational_noise = True

    def set_masks_for_batch(self):
        global_X_mask = deepcopy(self.train_miss_mask[self.batch_slice])
        global_Y_mask = deepcopy(self.noise_miss_mask[self.noise_batch_slice])

        if not self.use_cdi:
            self.X_mask = global_X_mask
            self.Y_mask = global_Y_mask
        else:
            # For each row of X & Y, if the row contains at least one zero, randomly choose one of these zeros to keep,
            # and fill in the rest with the 'global mean' from the observed data/noise
            self.X_mask = np.zeros_like(self.X)
            self.Y_mask = np.zeros((self.X.shape[0], self.nu, self.X.shape[1]))
            for i in range(len(self.X_mask)):
                row_i = global_X_mask[i]
                # get indices of all missing dimensions except 1 (randomly chosen)
                miss_indices = np.where(row_i == 1)[0]
                if miss_indices.size > 0:
                    if miss_indices.size > 1:
                        throw = np.random.randint(0, len(miss_indices))
                        throw_index = miss_indices[throw]
                    else:
                        throw_index = miss_indices[0]
                    self.X_mask[i, throw_index] = 1
                    self.Y_mask[i, :, throw_index] = 1
            self.Y_mask = self.Y_mask.reshape(-1, self.X.shape[1])

    def get_minibatch_E(self):
        """ get a minibatch of E (samples from a base distribution in reparam trick)

        There are two cases:
        1) global_E_ZX is a n-long list containing (nz, k) arrays, where k varies
        2) global_E_ZX is an array of shape (nz, n, d), where d is fixed
        """
        if isinstance(self.global_E_ZX, list):
            E = deepcopy(self.global_E_ZX[self.batch_slice])
        else:
            E = deepcopy(self.global_E_ZX[:, self.batch_slice, :])
        return E

    def shuffle_global_E(self, data_perm):
        """shuffle the global data E (samples from a base distribution in reparam trick)

        There are two cases:
        1) global_E_ZX is a n-long list containing (nz, k) arrays, where k varies
        2) global_E_ZX is an array of shape (nz, n, d), where d is fixed
        """
        if isinstance(self.global_E_ZX, list):
            self.global_E_ZX = [self.global_E_ZX[i] for i in data_perm]
        else:
            self.global_E_ZX = deepcopy(self.global_E_ZX[:, data_perm, :])

    def new_epoch(self):
        """Shuffle data X and noise Y and print current loss"""
        n, d = self.train_data.shape
        data_perm = rnd.permutation(n)
        self.train_data = deepcopy(self.train_data[data_perm])

        if self.train_miss_mask is not None:
            self.train_miss_mask = deepcopy(self.train_miss_mask[data_perm])
            # noise samples are grouped in nu-sized chunks that are missing the same features
            # we need to keep this grouping when we shuffle the data
            new_samples = deepcopy(self.noise_samples)
            new_mask = deepcopy(self.noise_miss_mask)
            new_samples = new_samples.reshape(n, self.nu, d)
            new_mask = new_mask.reshape(n, self.nu, d)
            self.noise_samples = new_samples[data_perm].reshape(self.noise_samples.shape)
            self.noise_miss_mask = new_mask[data_perm].reshape(self.noise_samples.shape)
        else:
            noise_perm = rnd.permutation(len(self.noise_samples))
            self.noise_samples = deepcopy(self.noise_samples[noise_perm])

        if self.use_reparam_trick:
            self.shuffle_global_E(data_perm)

        # print('epoch {}: J1 = {}'.format(self.current_epoch, self.current_loss))
        self.current_epoch += 1

    def resample_latents_if_necessary(self):
        if (self.ZX is None) or (self.ZY is None) or self.resample_from_variational_noise:
            if self.use_reparam_trick:
                # self.E_ZX = self.variational_dist.sample_E(self.nz, self.X_mask)
                self.ZX = self.variational_dist.get_Z_samples_from_E(self.nz, self.E_ZX, self.X, self.X_mask)
            else:
                self.ZX = self.variational_dist.sample(self.nz, self.X, self.X_mask)
            # todo: change this to use a global E_ZY, to reduce stochasticity
            self.ZY = self.variational_dist.sample(self.nz, self.Y, self.Y_mask)
            self.resample_from_variational_noise = False

            if self.use_cdi and (not self.eval_mode):
                # for the missing dimensions we just sampled, replace their value in the global dataset with the mean
                # of their respective conditional distribution
                self.train_data[self.batch_slice] = deepcopy(self.ZX.mean(0) + (1 - self.X_mask) * self.train_data[self.batch_slice])
                self.noise_samples[self.noise_batch_slice] = deepcopy(self.ZY.mean(0) + (1 - self.Y_mask) * self.noise_samples[self.noise_batch_slice])
