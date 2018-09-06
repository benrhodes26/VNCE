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
import os
import pickle
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd
from scipy.integrate import dblquad
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from sklearn.neighbors import KernelDensity as kd

from plot import save_fig

rc('lines', linewidth=1)
rc('font', size=10)
rc('legend', fontsize=10)
rc('text', usetex=True)


def joint_distribution(model, z1, z2, x):
    """
    param z1: float or array (n, 1)
    param z1: float or array (n, 1)
    param x: array (2, ) or (n, 2)
    """
    z = np.array([z1, z2])
    z = z.reshape(1, 1, 2)
    x = x.reshape(1, 2)
    return model(x, z)


def compute_true_landmark_marginals(model, x_landmarks):
    p_x = np.zeros(len(x_landmarks))
    integral_limit = 10

    for i, x_i in enumerate(x_landmarks):
        res = dblquad(lambda z1, z2: joint_distribution(model, z1, z2, x_i), -integral_limit, integral_limit,
                      lambda z1: -integral_limit, lambda z2: integral_limit)
        p_x[i] = res[0]
        
    return p_x


def plot_contours(ax, f, lim, num_contours, levels=None):
    delta = 0.05
    x = np.arange(-lim, lim, delta)
    y = np.arange(-lim, lim, delta)
    X, Y = np.meshgrid(x, y)
    mesh = np.vstack([X.flatten(), Y.flatten()]).T
    Z = f(mesh).reshape(X.shape)
    if levels:
        ax.contour(X, Y, Z, num_contours, colors='black', alpha=0.3, levels=levels, linewidths=2)
    else:
        ax.contour(X, Y, Z, num_contours, colors='black', alpha=0.3)


def plot_prior(ax, z):
    ax.set_title(r'$P(z): \mathcal{N}(0, \textbf{I})$', fontsize=16)
    sns.regplot(x=z[:, 0], y=z[:, 1], fit_reg=False, color='grey', ax=ax, scatter_kws={'s': 1})
    plot_contours(ax, lambda x: multivariate_normal.pdf(x, np.zeros(2), np.identity(2)), 10, 10)


def plot_p_x(ax, x, x_landmarks):
    # ax.set_title(r'$P(x): \mathcal{N}(\textbf{w}, c \textbf{I})$', fontsize=8)
    ax.set_title(r'$P(x)$', fontsize=16)
    sns.regplot(x=x[:, 0], y=x[:, 1], fit_reg=False, color='grey', ax=ax, scatter_kws={'s': 1})
    landmark_cols = ['red', 'green', 'blue']
    # landmark_cols = ['red', 'orange', 'green', 'blue', 'purple']
    for i, x_i in enumerate(x_landmarks):
        ax.scatter(x_i[0], x_i[1], color=landmark_cols[i], s=35, edgecolors='k')


def plot_noise(ax, noise, sample_size, noise_num):
    if noise_num == 1:
        ax.set_title(r'$P_y^1(Y): \mathcal{N}(\bar{\textbf{x}}, \bar{\Sigma})$', fontsize=16)
    if noise_num == 2:
        ax.set_title(r'$P_y^2(Y): \mathcal{N}(0, 10 \textbf{I})$', fontsize=16)
    y = noise.sample(sample_size)
    sns.regplot(x=y[:, 0], y=y[:, 1], fit_reg=False, color='grey', ax=ax, scatter_kws={'s': 1})
    plot_contours(ax, lambda x: noise(x), 10, 10)


def plot_marginals(x, z, x_landmarks, noise, bad_noise, sample_size, title, save_dir):
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.5), sharex=True, sharey=True)
    axs = axs.ravel()
    plot_prior(axs[0], z)
    plot_p_x(axs[1], x, x_landmarks)
    plot_noise(axs[2], noise, sample_size, 1)
    plot_noise(axs[3], bad_noise, sample_size, 2)

    for ax in axs:
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
    fig.tight_layout()
    save_fig(fig, save_dir + 'figs/', title)


def plot_true_posterior(ax, model, z1_mesh, z2_mesh, x, p_x, cmap):
    mesh = np.vstack([z1_mesh.flatten(), z2_mesh.flatten()]).T  # (gridsize, 2)
    p_mesh = model(x, mesh) / p_x
    p_mesh = p_mesh.reshape(z1_mesh.shape)
    # ax.pcolormesh(z1_mesh, z2_mesh, p_mesh, cmap=cmap)
    ax.contourf(z1_mesh, z2_mesh, p_mesh, cmap=cmap)


def plot_approx_posterior(ax, z1_mesh, z2_mesh, x, p_x, cmap, posterior, model):
    mesh = np.vstack([z1_mesh.flatten(), z2_mesh.flatten()]).T
    p_mesh = posterior(mesh, x)
    p_mesh = p_mesh.reshape(z1_mesh.shape)
    # ax.pcolormesh(z1_mesh, z2_mesh, p_mesh, cmap=cmap)
    ax.contourf(z1_mesh, z2_mesh, p_mesh, cmap=cmap)
    plot_contours(ax, lambda z: model(x, z) / p_x, 5, 1, levels=[0.05])


def plot_landmark_posteriors(x_landmarks,
                             p_x, model,
                             free_energy_posterior,
                             vnce_posterior,
                             bad_vnce_posterior,
                             bad_vnce_posterior_nu50,
                             title,
                             save_dir):
    nbins = 300
    axis_lim = 5
    z1_mesh, z2_mesh = np.mgrid[-axis_lim:axis_lim:nbins*1j, -axis_lim:axis_lim:nbins*1j]
    # cmaps = [plt.cm.YlOrRd, plt.cm.YlGn, plt.cm.GnBu]
    cmaps = [plt.cm.YlOrRd_r, plt.cm.YlGn_r, plt.cm.GnBu_r]
    # cmaps = [plt.cm.YlOrRd_r, plt.cm.Oranges_r, plt.cm.YlGn_r, plt.cm.Blues_r, plt.cm.Purples_r]

    sns.set_style('darkgrid')
    sns.set_palette(sns.color_palette("pastel"))
    num_rows, num_cols = 5, 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5, 8))

    for j in range(num_cols):
        plot_true_posterior(axs[0, j], model, z1_mesh, z2_mesh, x_landmarks[j], p_x[j], cmaps[j])
        plot_approx_posterior(axs[1, j], z1_mesh, z2_mesh, np.array([x_landmarks[j]]), p_x[j], cmaps[j], free_energy_posterior, model)
        plot_approx_posterior(axs[2, j], z1_mesh, z2_mesh, np.array([x_landmarks[j]]), p_x[j], cmaps[j], vnce_posterior, model)
        plot_approx_posterior(axs[3, j], z1_mesh, z2_mesh, np.array([x_landmarks[j]]), p_x[j], cmaps[j], bad_vnce_posterior, model)
        plot_approx_posterior(axs[4, j], z1_mesh, z2_mesh, np.array([x_landmarks[j]]), p_x[j], cmaps[j], bad_vnce_posterior_nu50, model)

    # remove space between subplots and add row labels
    for ax in axs.ravel():
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.set_aspect('equal')

    # add label to each row
    rows = ['True', 'KL', 'Noise 1\n' + r'$\nu=1$', 'Noise 2\n' + r'$\nu=1$', 'Noise 2\n' + r'$\nu=100$']
    pad = 5  # in points
    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                    textcoords='offset points', size='large', ha='right', va='center')

    fig.subplots_adjust(left=0.15, wspace=0, hspace=0.1)
    save_fig(fig, save_dir + 'figs/', title)


def model_to_noise_frac(x, z, pos, model, noise, nu):
    joint_noise = nu * pos(z, x) * noise(x)
    model_val = model(x, z)
    return joint_noise / (joint_noise + model_val)


def print_landmark_prob_of_noise_class(model, noise, bad_noise, vnce_pos, bad_vnce_pos, bad_vnce_pos_nu50, x_landmarks):
    landmarks = np.array(x_landmarks)
    good_z_landmarks = vnce_pos.sample(1, landmarks)
    bad_z_landmarks = bad_vnce_pos.sample(1, landmarks)
    bad50_z_landmarks = bad_vnce_pos_nu50.sample(1, landmarks)
    print(model_to_noise_frac(landmarks, good_z_landmarks, vnce_pos, model, noise, 1))
    print(model_to_noise_frac(landmarks, bad_z_landmarks, bad_vnce_pos, model, bad_noise, 1))
    print(model_to_noise_frac(landmarks, bad50_z_landmarks, bad_vnce_pos_nu50, model, bad_noise, 50))


def main():
    load_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project-non-code/experimental-results/stars-and-moons/'
    save_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project-non-code/experiments/stars-and-moons/'

    model = pickle.load(open(os.path.join(load_dir, 'truncate=False_model.p'), 'rb'))
    # model_trunc = pickle.load(open(os.path.join(save_dir, 'truncate=True_model.p'), 'rb'))

    fe_pos = pickle.load(open(os.path.join(load_dir, 'FreeEnergyLoss_truncate_gaussian=False_good_noise_nu1_var_dist.p'), 'rb'))
    vnce_pos = pickle.load(open(os.path.join(load_dir, 'VnceLoss_truncate_gaussian=False_good_noise_nu1_var_dist.p'), 'rb'))
    bad_vnce_pos = pickle.load(open(os.path.join(load_dir, 'VnceLoss_truncate_gaussian=False_bad_noise_nu1_var_dist.p'), 'rb'))
    bad_vnce_pos_nu50 = pickle.load(open(os.path.join(load_dir, 'VnceLoss_truncate_gaussian=False_bad_noise_nu50_var_dist.p'), 'rb'))

    # fe_pos_trunc = pickle.load(open(os.path.join(save_dir, 'FreeEnergyLoss_truncate_gaussian=True_good_noise_nu1_var_dist.p'), 'rb'))
    # vnce_pos_trunc = pickle.load(open(os.path.join(save_dir, 'VnceLoss_truncate_gaussian=True_good_noise_nu1_var_dist.p'), 'rb'))
    # bad_vnce_pos_trunc = pickle.load(open(os.path.join(save_dir, 'VnceLoss_truncate_gaussian=True_bad_noise_nu1_var_dist.p'), 'rb'))
    # bad_vnce_pos_nu50_trunc = pickle.load(open(os.path.join(save_dir, 'VnceLoss_truncate_gaussian=True_bad_noise_nu50_var_dist.p'), 'rb'))

    noise = pickle.load(open(os.path.join(load_dir, 'good_noise.p'), 'rb'))
    bad_noise = pickle.load(open(os.path.join(load_dir, 'bad_noise.p'), 'rb'))

    sample_size = 500
    Z, X = model.sample(sample_size)
    # Z_trunc, X_trunc = model_trunc.sample(sample_size)

    x_landmarks = [np.array([-5, -5]), np.array([0, 0]), np.array([5, 5])]
    # x_landmarks_trunc = [np.array([-2, -2]), np.array([0, 0]), np.array([2, 2])]

    plot_marginals(X, Z, x_landmarks, noise, bad_noise, sample_size, title='marginals-for-gaussian-model', save_dir=save_dir)
    # plot_marginals(X_trunc, Z_trunc, x_landmarks_trunc, noise, bad_noise, sample_size, title='marginals-for-truncated-gaussian-model', save_dir=save_dir)

    p_x = compute_true_landmark_marginals(model, x_landmarks)
    # p_x_trunc = compute_true_landmark_marginals(model_trunc, x_landmarks_trunc)

    plot_landmark_posteriors(x_landmarks,
                             p_x,
                             model,
                             fe_pos,
                             vnce_pos,
                             bad_vnce_pos,
                             bad_vnce_pos_nu50,
                             title='landmark-posteriors-gaussian-model',
                             save_dir=save_dir)
    # plot_landmark_posteriors(x_landmarks_trunc,
    #                          p_x_trunc,
    #                          model_trunc,
    #                          fe_pos_trunc,
    #                          vnce_pos_trunc,
    #                          bad_vnce_pos_trunc,
    #                          bad_vnce_pos_nu50_trunc,
    #                          title='landmark-posteriors-truncated-gaussian-model',
    #                          save_dir=save_dir)

    print_landmark_prob_of_noise_class(model, noise, bad_noise, vnce_pos, bad_vnce_pos, bad_vnce_pos_nu50, x_landmarks)
    # print_landmark_prob_of_noise_class(model_trunc, noise, bad_noise, vnce_pos_trunc, bad_vnce_pos_trunc, bad_vnce_pos_nu50_trunc, x_landmarks_trunc)


if __name__ == "__main__":
    main()
