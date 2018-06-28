# -*- coding: utf-8 -*-
"""Parameter initialisers.

This module defines classes to initialise the parameters in a layer.
"""

import numpy as np

DEFAULT_SEED = 10102016

class ConstantInit(object):
    """Constant parameter initialiser."""

    def __init__(self, value):
        """Construct a constant parameter initialiser.

        Args:
            value: Value to initialise parameter to.
        """
        self.value = value

    def __call__(self, shape):
        return np.ones(shape=shape) * self.value


class ConstantVectorInit(object):
    """Constant parameter initialiser where the constants per hidden unit can be different."""

    def __init__(self, value):
        """Construct a constant parameter initialiser.

        Args:
            value: Value to initialise parameter to.
        """
        self.value = value

    def __call__(self, shape):
        return self.value


class UniformInit(object):
    """Random uniform parameter initialiser."""

    def __init__(self, low, high, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            low: Lower bound of interval to sample from.
            high: Upper bound of interval to sample from.
            rng (RandomState): Seeded random number generator.
        """
        self.low = low
        self.high = high
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.uniform(low=self.low, high=self.high, size=shape)


class NormalInit(object):
    """Random normal parameter initialiser."""

    def __init__(self, mean, std, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            mean: Mean of distribution to sample from.
            std: Standard deviation of distribution to sample from.
            rng (RandomState): Seeded random number generator.
        """
        self.mean = mean
        self.std = std
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.normal(loc=self.mean, scale=self.std, size=shape)


class FanInUniformInit(object):
    """Random uniform weights initialiser.
    Initialises an two-dimensional parameter array using the 'normalized
    initialisation' scheme which attempts to maintain a
    roughly constant variance in the activations of a multi-layer model 
    consisting of interleaved affine and logistic sigmoidal transformation layers.
    Weights are sampled from a zero-mean uniform distribution with standard
    deviation `sqrt(1 / input_dim)` where `input_dim` is the input dimension of 
    the weight matrix.
    """

    def __init__(self, gain=1., rng=None):
        """Construct a normalised initilisation random initialiser object.
        Args:
            gain: Multiplicative factor to scale initialised weights by.
                Recommended values is 1 for affine layers followed by
                logistic sigmoid layers (or another affine layer).
            rng (RandomState): Seeded random number generator.
        """
        self.gain = gain
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        assert len(shape) == 2, (
            'Initialiser should only be used for two dimensional arrays.')
        std = self.gain * (1. / (shape[1]))**0.5
        half_width = 3.**0.5 * std
        return self.rng.uniform(low=-half_width, high=half_width, size=shape)
    
    def __repr__(self):
        return 'Uniform Fan-in'

class FanOutUniformInit(object):
    """Random uniform weights initialiser.
    Initialises an two-dimensional parameter array using the 'normalized
    initialisation' scheme which attempts to maintain a
    roughly constant variance in the output gradients of a multi-layer model 
    consisting of interleaved affine and logistic sigmoidal transformation layers.
    Weights are sampled from a zero-mean uniform distribution with standard
    deviation `sqrt(1 / output_dim)` where `output_dim` is the output dimension of 
    the weight matrix.
    """

    def __init__(self, gain=1., rng=None):
        """Construct a normalised initilisation random initialiser object.
        Args:
            gain: Multiplicative factor to scale initialised weights by.
                Recommended values is 1 for affine layers followed by
                logistic sigmoid layers (or another affine layer).
            rng (RandomState): Seeded random number generator.
        """
        self.gain = gain
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        assert len(shape) == 2, (
            'Initialiser should only be used for two dimensional arrays.')
        std = self.gain * (1. / (shape[0]))**0.5
        half_width = 3.**0.5 * std
        return self.rng.uniform(low=-half_width, high=half_width, size=shape)
    
    def __repr__(self):
        return 'Uniform Fan-out'


class FanInGaussianInit(object):
    """Random Gaussian weights initialiser.
    Initialises an two-dimensional parameter array using the 'normalized
    initialisation' scheme which attempts to maintain a
    roughly constant variance in the activations of a multi-layer model 
    consisting of interleaved affine and logistic sigmoidal transformation layers.
    Weights are sampled from a zero-mean Gaussian distribution with standard
    deviation `sqrt(1 / input_dim)` where `input_dim` is the input dimension of 
    the weight matrix.
    """

    def __init__(self, gain=1., rng=None):
        """Construct a normalised initilisation random initialiser object.
        Args:
            gain: Multiplicative factor to scale initialised weights by.
                Recommended values is 1 for affine layers followed by
                logistic sigmoid layers (or another affine layer).
            rng (RandomState): Seeded random number generator.
        """
        self.gain = gain
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        assert len(shape) == 2, (
            'Initialiser should only be used for two dimensional arrays.')
        std = self.gain * (1. / (shape[1]))**0.5
        return self.rng.normal(loc=0., scale=std, size=shape)
        
    def __repr__(self):
        return 'Gaussian Fan-in'

class FanOutGaussianInit(object):
    """Random Gaussian weights initialiser.
    Initialises an two-dimensional parameter array using the 'normalized
    initialisation' scheme which attempts to maintain a
    roughly constant variance in the output gradients of a multi-layer model 
    consisting of interleaved affine and logistic sigmoidal transformation layers.
    Weights are sampled from a zero-mean Gaussian distribution with standard
    deviation `sqrt(1 / output_dim)` where `output_dim` is the output dimension of 
    the weight matrix.
    """

    def __init__(self, gain=1., rng=None):
        """Construct a normalised initilisation random initialiser object.
        Args:
            gain: Multiplicative factor to scale initialised weights by.
                Recommended values is 1 for affine layers followed by
                logistic sigmoid layers (or another affine layer).
            rng (RandomState): Seeded random number generator.
        """
        self.gain = gain
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        assert len(shape) == 2, (
            'Initialiser should only be used for two dimensional arrays.')
        std = self.gain * (1. / (shape[0]))**0.5
        return self.rng.normal(loc=0., scale=std, size=shape)
            
    def __repr__(self):
        return 'Gaussian Fan-out'

class GlorotUniformInit(object):
    """Glorot and Bengio (2010) random uniform weights initialiser.
    Initialises an two-dimensional parameter array using the 'normalized
    initialisation' scheme suggested in [1] which attempts to maintain a
    roughly constant variance in the activations and backpropagated gradients
    of a multi-layer model consisting of interleaved affine and logistic
    sigmoidal transformation layers.
    Weights are sampled from a zero-mean uniform distribution with standard
    deviation `sqrt(2 / (input_dim * output_dim))` where `input_dim` and
    `output_dim` are the input and output dimensions of the weight matrix
    respectively.
    References:
      [1]: Understanding the difficulty of training deep feedforward neural
           networks, Glorot and Bengio (2010)
    """

    def __init__(self, gain=1., rng=None):
        """Construct a normalised initilisation random initialiser object.
        Args:
            gain: Multiplicative factor to scale initialised weights by.
                Recommended values is 1 for affine layers followed by
                logistic sigmoid layers (or another affine layer).
            rng (RandomState): Seeded random number generator.
        """
        self.gain = gain
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        assert len(shape) == 2, (
            'Initialiser should only be used for two dimensional arrays.')
        std = self.gain * (2. / (shape[0] + shape[1]))**0.5
        half_width = 3.**0.5 * std
        return self.rng.uniform(low=-half_width, high=half_width, size=shape)
            
    def __repr__(self):
        return 'Uniform Glorot'


class GlorotNormalInit(object):
    """Glorot and Bengio (2010) random normal weights initialiser.
    Initialises a two-dimensional parameter array using the 'normalized
    initialisation' scheme suggested in [1] which attempts to maintain a
    roughly constant variance in the activations and backpropagated gradients
    of a multi-layer model consisting of interleaved affine and logistic
    sigmoidal transformation layers.
    Weights are sampled from a zero-mean normal distribution with standard
    deviation `sqrt(2 / (input_dim * output_dim))` where `input_dim` and
    `output_dim` are the input and output dimensions of the weight matrix
    respectively.
    References:
      [1]: Understanding the difficulty of training deep feedforward neural
    """

    def __init__(self, gain=1., rng=None):
        """Construct a normalised initilisation random initialiser object.
        Args:
            gain: Multiplicative factor to scale initialised weights by.
                Recommended values is 1 for affine layers followed by
                logistic sigmoid layers (or another affine layer).
            rng (RandomState): Seeded random number generator.
        """
        self.gain = gain
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        std = self.gain * (2. / (shape[0] + shape[1]))**0.5
        return self.rng.normal(loc=0., scale=std, size=shape)
                
    def __repr__(self):
        return 'Gaussian Glorot'
