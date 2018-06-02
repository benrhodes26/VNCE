
import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)


import numpy as np
import pickle

from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from utils import plot_log_likelihood_training_curves, takeClosest, create_J_diff_plot, get_Js_for_vnce_thetas
from vnce_optimisers import VNCEOptimiser

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rc
from numpy import random as rnd

rc('lines', linewidth=1.5)
rc('font', size=18)
rc('legend', fontsize=16)
rc('text', usetex=True)
rc('xtick', labelsize=18)
rc('ytick', labelsize=18)

# For reproducibility
rng = rnd.RandomState(1083463236)

exp_name = 'usps-sgd'  # CHANGE ME
model_name = 'rbm'  # CHANGE ME
skip_files = []

data_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data'
exp_res_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/' + model_name
save_res_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/experiments/' + model_name
exp_dir = os.path.join(exp_res_dir, exp_name)
save_dir = os.path.join(save_res_dir, exp_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def annotate(axis, annotation, xy, a_length=100):
    axis.annotate(annotation, xy=xy, xycoords='data', xytext=(a_length, a_length), textcoords="offset points",
                  arrowprops=dict(facecolor='black', shrink=0.03, width=2), horizontalalignment='right',
                  verticalalignment='bottom', fontsize=16)

def plot_and_annotate(axis, times, vals, annotation, annotate_val, a_length=100):
    axis.semilogx(times, vals)
    ind = takeClosest(vals, annotate_val)
    annotate(axis, annotation, (times[ind], vals[ind]), a_length)
    axis.annotate(r"{}".format(round(vals[-1], 2)), xy=(times[-1], vals[-1]), fontsize=12)


"""
==========================================================================================================
                                            Log likelihood plot
==========================================================================================================
"""

# CHANGE ME
arrow_y_val = 0
arrow_y_val_2 = 0
arrow_y_val_3 = -2.22  # log-like plot vnce
arrow_y_val_4 = 0  # log-like plot cd

fig, axs = plt.subplots(2, 2, figsize=(20, 25))
axs = axs.ravel()
# sorted_files = [str(file) for file in sorted([int(file) for file in os.listdir(exp_dir)])]
for i, file in enumerate(os.listdir(exp_dir)):
    exp = os.path.join(exp_dir, file)

    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
    
    loaded_data = np.load(os.path.join(exp, 'data.npz'))
    loaded_init = np.load(os.path.join(exp, 'init_theta_and_likelihood.npz'))
    loaded_vnce = np.load(os.path.join(exp, 'vnce_results.npz'))
    loaded_cd = np.load(os.path.join(exp, 'cd_results.npz'))
    loaded_nce = np.load(os.path.join(exp, 'nce_results.npz'))
    
    X = loaded_data['X']
    Y = loaded_data['Y']
    loaded_test_data = np.load(os.path.join(data_dir, 'usps_3by3patches.npz'))
    X_test = loaded_test_data['test']

    theta0 = loaded_init['theta0']
    init_log_like = loaded_init['ll']
    
    J1s = loaded_vnce['J1s']
    vnce_thetas = loaded_vnce['params']
    vnce_times = loaded_vnce['times']
    vnce_reduced_times = loaded_vnce['reduced_times']
    av_log_like_vnce = loaded_vnce['ll']
    
    cd_times = loaded_cd['times']
    cd_reduced_times = loaded_cd['reduced_times']
    av_log_like_cd = loaded_cd['ll']
    
    Js = loaded_nce['Js']
    Js_for_vnce_thetas = loaded_nce['Js_for_vnce_thetas']
    nce_times = loaded_nce['times']
    nce_thetas = loaded_nce['params']
    nce_reduced_times = loaded_nce['reduced_times']
    av_log_like_nce = loaded_nce['ll']

    # todo: del me
    nce_reduced_times = nce_reduced_times[:len(av_log_like_nce)]
    cd_reduced_times[0] = 1e-6
    nce_reduced_times[0] = 1e-6

    rng = np.random.RandomState(config['random_seed'])
    
    # reconstruct models & optimisers
    model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
    nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)

    emp_dist = pickle.load(open(os.path.join(data_dir, 'usps_3by3_emp_dist.p'), 'rb'))
    emp_log_like = np.mean(np.log(emp_dist(X_test)))

    var_dist = RBMLatentPosterior(theta0, rng=rng)
    
    noise = MultivariateBernoulliNoise(np.mean(X, axis=0))
    noise_log_like = np.mean(np.log(noise(X)))
    
    optimiser = VNCEOptimiser(model=model, noise=noise, variational_dist=var_dist, noise_samples=Y, 
                              nu=config['nu'], latent_samples_per_datapoint=config['nz'], rng=rng)
    nce_optimiser = NCEOptimiser(model=nce_model, noise=noise, noise_samples=Y, nu=config['nu'])
    
    # specify arrow position and lengths for annotations
    arrow_y_val_3 = noise_log_like + ((emp_log_like - noise_log_like)/2) + 0.6  # log-like plot vnce
    arrow_y_val_4 = noise_log_like + ((emp_log_like - noise_log_like)/2) + 0.6  # log-like plot cd
    arrow_length = 30
    
    # LOG-LIKELIHOOD PLOT
    ax = axs[i]
    ax.set_title(file)
    plot_and_annotate(ax, cd_reduced_times, av_log_like_cd, "CD {}".format(config['cd_num_steps']),
                      arrow_y_val_4, a_length=arrow_length)
    plot_and_annotate(ax, vnce_reduced_times, av_log_like_vnce, "VNCE", arrow_y_val_3, a_length=arrow_length)
    
    ax.set_xlabel('time (seconds)', fontsize=18)
    ax.set_ylabel('log likelihood', fontsize=18)

    # PLOT NOISE AND OPTIMAL LOG-LIKELIHOODS AS HORIZONTAL LINES
    plot_and_annotate(ax, nce_reduced_times, av_log_like_nce, "NCE", arrow_y_val_3-0.1, a_length=arrow_length+20)
    ax.plot((0, plt.get(ax, 'xlim')[1]), (noise_log_like, noise_log_like), 'g--', label='Noise distribution')
    # ax.plot((0, plt.get(ax, 'xlim')[1]), (init_log_like, init_log_like), 'r--', label='initial model')
    ax.plot((0, plt.get(ax, 'xlim')[1]), (emp_log_like, emp_log_like), 'b--', label='Empirical distribution')

    axs[i].set_xlim(1e-2, axs[i].get_xlim()[1])
    axs[i].set_ylim((noise_log_like-0.05, emp_log_like + 0.05))
    axs[i].legend(loc='lower right')

    with open(os.path.join(save_dir, "config-{}.txt".format(file)), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))

fig.savefig(os.path.join(save_dir, 'usps-different_num_hiddens.pdf'))

"""
==========================================================================================================
                                            J/J1 diff plot
==========================================================================================================
"""

for i, file in enumerate(os.listdir(exp_dir)):
    exp = os.path.join(exp_dir, file)

    # LOAD EVERYTHING FROM FILE
    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
    
    loaded_data = np.load(os.path.join(exp, 'data.npz'))
    loaded_init = np.load(os.path.join(exp, 'init_theta_and_likelihood.npz'))
    loaded_vnce = np.load(os.path.join(exp, 'vnce_results.npz'))
    loaded_nce = np.load(os.path.join(exp, 'nce_results.npz'))
    
    X = loaded_data['X']
    Y = loaded_data['Y']
    
    theta0 = loaded_init['theta0']
    init_log_like = loaded_init['ll']
    
    E_step_ids = loaded_vnce['E_step_ids']
    J1s = loaded_vnce['J1s']
    J1s_1 = J1s[:, 0]
    J1s_2 = J1s[:, 1]
    vnce_thetas = loaded_vnce['params']
    vnce_times = loaded_vnce['times']
    vnce_reduced_times = loaded_vnce['reduced_times']
    av_log_like_vnce = loaded_vnce['ll']
    
    Js = loaded_nce['Js']
    Js_for_vnce_thetas = loaded_nce['Js_for_vnce_thetas']
    nce_times = loaded_nce['times']
    nce_reduced_times = loaded_nce['reduced_times']
    av_log_like_nce = loaded_nce['ll']
    
    # reconstruct models & optimisers
    model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
    nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
    
    var_dist = RBMLatentPosterior(theta0, rng=rng)
    
    noise = MultivariateBernoulliNoise(np.mean(X, axis=0))
    noise_log_like = np.mean(np.log(noise(X)))
    
    # ACTUAL PLOTTING
    if config['optimiser'] == 'VNCEOptimiserWithoutImportanceSampling':
        J_plot, _ = create_J_diff_plot(J1s, vnce_times, E_step_ids, Js_for_vnce_thetas, 
                                       posterior_ratio_vars=None, plot_posterior_ratio=False)
        axs = J_plot.axes
        axs[0].set_xlim(-5, 100)
        axs[0].set_ylim((-0.2, 0.2))
    else:
        J_plot, _ = create_J_diff_plot(J1s, vnce_times, E_step_ids, Js_for_vnce_thetas, 
                                       posterior_ratio_vars=None, plot_posterior_ratio=False)
        axs = J_plot.axes
        axs[0].set_xlim(-5, 100)
        axs[0].set_ylim((-0.2, 0.2))

    J_plot.suptitle('{}'.format(file))
    J_plot.set_size_inches(15, 15)

    J_plot.savefig(os.path.join(save_dir, 'two-terms-of-J1-{}.pdf'.format(file)))
