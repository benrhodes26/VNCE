import numpy as np
import os
import pickle

from bisect import bisect_left
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt


def plot_nce_minus_vnce_loss(nce_loss_for_vnce_params, vnce_losses, times, e_step_ids):
    """plot J (NCE objective function) minus J1 (lower bound to NCE objective)"""

    fig, ax = plt.subplots(1, 1, figsize=(15, 20))

    diff = np.sum(nce_loss_for_vnce_params, axis=1) - np.sum(vnce_losses, axis=1)
    ax.plot(times, diff, c='k', label='J - J1')
    diff1 = nce_loss_for_vnce_params[:, 0] - vnce_losses[:, 0]
    ax.plot(times, diff1, c='r', label='term1: J - J1')
    diff2 = nce_loss_for_vnce_params[:, 1] - vnce_losses[:, 1]
    ax.plot(times, diff2, c='b', label='term2: J - J1')

    max_y_val = max(diff1.max(), diff2.max())
    min_y_val = min(diff1.min(), diff2.min())
    for ax in axs:
        for time_id in e_step_ids:
            time = times[time_id]
            ax.plot((time, time), (min_y_val, max_y_val), c='0.5')
        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.legend()

    return fig


def plot_vnce_loss(vnce_losses, times, vnce_val_losses, val_times, m_step_start_ids=None, e_step_start_ids=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.7, 2.5))

    if vnce_losses.ndim == 2:
        # first and second terms of vnce loss were saved separately, but combine them for the plot
        vnce_losses = np.sum(vnce_losses, axis=1)
        vnce_val_losses = np.sum(vnce_val_losses, axis=1)

    ax.plot(times, vnce_losses, c='k', label='J1 train')
    ax.plot(val_times, vnce_val_losses, c='green', label='J1 val')

    max_y_val = vnce_losses.max()
    min_y_val = vnce_losses.min()
    if e_step_start_ids is not None:
        for time_id in e_step_start_ids:
            time = times[time_id]
        ax.plot((time, time), (min_y_val, max_y_val), c='0.3', label='start of E step')
    if m_step_start_ids is not None:
        for time_id in m_step_start_ids:
            time = times[time_id]
            ax.plot((time, time), (min_y_val, max_y_val), c='0.7', label='start of M step')

    ax.set_xlabel('time (seconds)', fontsize=16)
    remove_duplicate_legends(ax)

    return fig


def plot_rbm_parameters(params, titles, d, m, with_bias=False, figsize=(15, 25)):
    """plot heatmaps of restricted boltzmann machine weights

    :param params: list arrays
        list of (flattened) params of different models
    :param titles: list of strings
        list of titles for each model
    :param d: int
        dimension of visibles
    :parm m: int
        dimension of latents
    """
    if with_bias:
        fig, axs = plt.subplots(len(params), 2, figsize=figsize)
        axs = axs.ravel()
        for i in range(0, len(axs), 2):
            j = int(i/2)
            params[j] = np.array(params[j].reshape(d+1, m+1).T)
            params[j][0, 0] = 0  # ignore scaling param

            axs[i].imshow(params[j], cmap='Blues')
            axs[i].set_title('{} with biases'.format(titles[j]), fontsize=16)
            axs[i].set_xlabel('visible weights', fontsize=16)
            axs[i].set_ylabel('hidden weights', fontsize=16)

            axs[i+1].imshow(params[j][1:, 1:], cmap='Blues')
            axs[i+1].set_title('{} without biases'.format(titles[j]), fontsize=16)
            axs[i+1].set_xlabel('visible weights', fontsize=16)
            axs[i+1].set_ylabel('hidden weights', fontsize=16)
    else:
        fig, axs = plt.subplots(len(params), 1, figsize=figsize)
        axs = axs.ravel()
        for i in range(0, len(axs)):
            params[i] = np.array(params[i].reshape(d+1, m+1).T)
            params[i][0, 0] = 0  # ignore scaling param

            axs[i].imshow(params[i], cmap='Blues')
            axs[i].set_title('{} with biases'.format(titles[i]), fontsize=16)
            axs[i].set_xlabel('visible weights', fontsize=16)
            axs[i].set_ylabel('hidden weights', fontsize=16)
    plt.tight_layout()

    return fig


def plot_log_likelihood_learning_curves(training_curves,
                                        static_lines,
                                        save_dir,
                                        x_lim=None,
                                        y_lim=None,
                                        file_name='train',
                                        title=None,
                                        logx=False,
                                        ax=None):
    """plot log-likelihood training curves

    :param training_curves: list of lists
        each inner list is of the form: [times, log-likelihoods, label, color]
        where times & log-likelihoods are arrays for plotting
    :param static_lines: list of lists
        each inner list is of the form [log-likelihood, label] where
        log-likelihood is a single value that will plotted as a
        horizontal line
    """
    create_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        create_fig = True

    for triple in training_curves:
        if logx:
            ax.semilogx(triple[0], triple[1], label=triple[2], color=triple[3])
        else:
            ax.plot(triple[0], triple[1], label=triple[2])
        ax.annotate(r"{}".format(round(triple[1][-1], 2)), xy=(triple[1][-1], triple[1][-1] + 1), fontsize=5, color=triple[3])
    for pair in static_lines:
        ax.plot(plt.get(ax, 'xlim'), (pair[0], pair[0]), label=pair[1])

    if title:
        ax.set_title(title)
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('log likelihood')
    ax.legend(loc='lower right')
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    if create_fig:
        save_fig(fig, save_dir, '{}_likelihood_optimisation_curve'.format(file_name))


def plot_2d_density(ax, f, cmap, low_x_lim, up_x_lim, low_y_lim, up_y_lime):
    nbins = 300
    z1_mesh, z2_mesh = np.mgrid[low_x_lim:up_x_lim:nbins * 1j, -low_y_lim:up_y_lime:nbins * 1j]
    mesh = np.vstack([z1_mesh.flatten(), z2_mesh.flatten()]).T
    f_mesh = f(mesh).reshape(z1_mesh.shape)
    ax.pcolormesh(z1_mesh, z2_mesh, f_mesh, cmap=cmap)


def save_fig(fig, save_dir, title):
    save_path = os.path.join(save_dir, title)
    fig.savefig(save_path + '.png', bbox_inches="tight", dpi=300)
    fig.savefig(save_path + '.pdf', bbox_inches="tight")
    pickle.dump(fig, open(save_path + '.p', "wb"))


def change_fig_fontsize(fig, new_size):
    for ax in fig.axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.legend().get_texts()):
            item.set_fontsize(new_size)


def remove_duplicate_legends(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
