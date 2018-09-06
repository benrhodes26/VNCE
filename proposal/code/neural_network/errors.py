import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

import numpy as np
from utils import validate_shape


class FreeEnergyLoss(object):

    def __init__(self, model, var_dist, num_z_per_x, data_mean, data_std):
        self.model = model
        self.var_dist = var_dist
        self.nz = num_z_per_x
        self.data_mean = data_mean
        self.data_std = data_std

    def __call__(self, nn_outputs, inputs):
        """Evaluate (negative) free energy loss when the variational posterior has parameters given by a neural net

        :param outputs: Array of neural network outputs of shape (batch_size, output_dim).
        :param targets: Array of inputs of shape (batch_size, input_dim).

        :return loss: float
        """
        X = (inputs * self.data_std) + self.data_mean
        Z = self.var_dist.sample(nz=self.nz, U=inputs, nn_outputs=nn_outputs)  # (nz, n, m)
        log_model = self.model(U=X, Z=Z, log=True)  # (nz, n)
        var_dist_entropy = self.var_dist.entropy(U=inputs, outputs=nn_outputs)

        free_energy = np.mean(log_model, axis=0) + var_dist_entropy  # (n, )
        av_free_energy = np.mean(free_energy)

        return - av_free_energy

    def grad(self, nn_outputs, inputs):
        """ grad of (negative) free energy w.r.t variational parameters (belonging to a neural network)
        :param nn_outputs:
        :param inputs:
        :return:
        """
        X = (inputs * self.data_std) + self.data_mean
        E = self.var_dist.sample_E(self.nz, inputs.shape[0])
        Z = self.var_dist.get_Z_samples_from_E(E=E, U=inputs, outputs=nn_outputs)

        grad_z_wrt_nn_outputs = self.var_dist.grad_of_Z_wrt_nn_outputs(nn_outputs, E)  # (output_dim, nz, n, m)
        grad_of_log_model = self.model.grad_log_wrt_nn_outputs(X, Z, grad_z_wrt_nn_outputs)  # (output_dim, nz, n)
        av_grad_of_log_model = np.mean(grad_of_log_model, axis=1).T  # (n, output_dim)

        grad_of_var_dist_entropy = self.var_dist.grad_entropy_wrt_nn_outputs(outputs=nn_outputs)  # (n, output_dim)

        return - av_grad_of_log_model - grad_of_var_dist_entropy

    def __repr__(self):
        return 'FreeEnergyLoss'


class VnceLoss(object):

    def __init__(self, model, noise, var_dist, num_z_per_x, noise_to_data_ratio, data_mean, data_std):
        self.model = model
        self.noise = noise
        self.var_dist = var_dist
        self.nz = num_z_per_x
        self.nu = noise_to_data_ratio
        self.data_mean = data_mean
        self.data_std = data_std

    def __call__(self, nn_outputs, inputs):
        """Evaluate (negative) VNCE loss when the variational posterior has parameters given by a neural net

        :param outputs: Array of neural network outputs of shape (batch_size, output_dim).
        :param targets: Array of inputs of shape (batch_size, input_dim).

        :return loss: float
        """
        X = (inputs * self.data_std) + self.data_mean  # inputs to the nn were scaled and centred, so undo this
        ZX = self.var_dist.sample(nz=self.nz, U=inputs, nn_outputs=nn_outputs)  # (nz, n, m)

        Y = self.noise.sample(self.nu * len(X))
        Y2 = (Y - self.data_mean) / self.data_std
        ZY = self.var_dist.sample(nz=self.nz, U=Y2, nn_outputs=None)

        term1 = self.first_term_of_loss(X, ZX)  # (nz, n)
        term2 = self.second_term_of_loss(Y, ZY)  # (nz, n*nu)

        return - (term1 + term2)

    def first_term_of_loss(self, X, ZX):
        nu = self.nu

        # use a numerically stable implementation of the cross-entropy sigmoid
        h_x = self.h(X, ZX)
        a = (h_x > 0) * np.log(1 + nu * np.exp(-h_x))
        b = (h_x < 0) * (-h_x + np.log(nu + np.exp(h_x)))
        first_term = -np.mean(a + b)

        validate_shape(a.shape, (self.nz, len(X)))
        return first_term

    def second_term_of_loss(self, Y, ZY):
        nu = self.nu

        # We could use the same cross-entropy sigmoid trick as above, BEFORE using importance sampling.
        # Currently not doing this - not sure which way round is better.
        h_y = self.h(Y, ZY)
        expectation = np.mean(np.exp(h_y), axis=0)
        c = (1 / nu) * expectation  # (n*nu, )
        second_term = -nu * np.mean(np.log(1 + c))

        validate_shape(c.shape, (len(Y), ))
        return second_term

    def h(self, U, Z):
        """Compute the ratio: model / (noise*q).

        :param U: array of shape (?, d)
            U can be either data or noise samples, so ? is either n or n*nu
        :param Z: array of shape (nz, ?, m)
            ? is either n or n*nu
        :return: array of shape (nz, ?)
            ? is either n or n*nu
        """
        if len(Z.shape) == 2:
            Z = Z.reshape((1, ) + Z.shape)

        log_model = self.model(U, Z, log=True)
        q = self.var_dist(Z, U)
        val = log_model - np.log((q * self.noise(U)))
        validate_shape(val.shape, (Z.shape[0], Z.shape[1]))

        return val

    def grad(self, nn_outputs, inputs):
        """ grad of (negative) VNCE loss w.r.t to variational parameters (belonging to a neural network)
        :param nn_outputs:
        :param inputs:
        :return:
        """
        X = (inputs * self.data_std) + self.data_mean  # inputs to the nn were scaled and centred, so undo this
        E = self.var_dist.sample_E(self.nz, inputs.shape[0])
        Z = self.var_dist.get_Z_samples_from_E(E=E, U=inputs, outputs=nn_outputs)

        grad_z_wrt_nn_outputs = self.var_dist.grad_of_Z_wrt_nn_outputs(nn_outputs, E)  # (output_dim, nz, n, m)
        grad_of_log_model = self.model.grad_log_wrt_nn_outputs(X, Z, grad_z_wrt_nn_outputs)  # (output_dim, nz, n)
        # todo: compute grad wrt. var_dist correctly...
        grad_of_log_var_dist = self.var_dist.grad_log_wrt_nn_outputs(outputs=nn_outputs)  # (n, output_dim)

        joint_noise = (self.nu * self.noise(X) * self.var_dist(Z, X))
        a = joint_noise / (self.model(X, Z) + joint_noise)  # (nz, n)

        term1 = np.mean(a * grad_of_log_model, axis=1).T  # (n, output_dim)
        term2 = (np.mean(a, axis=0) * grad_of_log_var_dist.T).T  # (n, output_dim)
        return - (term1 + term2)

    def __repr__(self):
        return 'VnceLoss'
