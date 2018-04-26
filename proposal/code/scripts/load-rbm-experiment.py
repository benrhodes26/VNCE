import os
import sys

# todo: work out how to do imports correctly, so I can remove this hack
code_dir = os.path.split(os.getcwd())[0]
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
from numpy import random as rnd

# For reproducibility
rng = rnd.RandomState(1083463236)

exp_name = 'large_models_var_threshold'  # CHANGE ME
model_name = 'rbm'  # CHANGE ME

# CHANGE ME
arrow_y_val = -1.11
arrow_y_val_2 = -1.11
arrow_y_val_3 = -6.9 # log-like plot vnce
arrow_y_val_4 = -7.0 # log-like plot cd

exp_res_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/' + model_name
save_res_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/experiments/' + model_name
exp_dir = os.path.join(exp_res_dir, exp_name)
save_dir = os.path.join(save_res_dir, exp_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def sanity_check_stats(J1s, Js_for_vnce_thetas, file):
    stats ='file: {0}. Percentage of iters for which J1 increases: {1:.4g}%\n          ' \
           'Percentage of iters for which J increases: {2:.4g}%\n          ' \
           'Percentage of iters for which J1 is less than J: {3:.4g}%\n'.format(file,
    100* np.sum(np.array([J1a < J1b for J1a, J1b in zip(J1s[:-1], J1s[1:])])) / (len(J1s)-1),
    100*np.sum(np.array([Ja < Jb for Ja, Jb in zip(Js_for_vnce_thetas[:-1], Js_for_vnce_thetas[1:])])) / (len(Js_for_vnce_thetas)-1),
    100*np.mean(J1s < Js_for_vnce_thetas))
    
    return stats

def annotate(ax, annotation, xy):
    ax.annotate(annotation, xy=xy, xycoords='data', xytext=(100, 100), textcoords="offset points", 
        arrowprops=dict(facecolor='black', shrink=0.03), horizontalalignment='right', 
        verticalalignment='bottom', fontsize=16)

def plot_and_annotate(ax, times, vals, annotation, annotate_val):
    ax.semilogx(times, vals)
    ind = takeClosest(vals, annotate_val)
    annotate(ax, annotation, (times[ind], vals[ind]))
    ax.annotate(r"{}".format(round(vals[-1], 4)), xy=(times[-1], vals[-1]), fontsize=12)

fig, axs = plt.subplots(2, 1, figsize=(20, 25))
axs = axs.ravel()
#sorted_files = [str(file) for file in sorted([int(file) for file in os.listdir(exp_dir)])]
for i, file in enumerate(os.listdir(exp_dir)):
    exp = os.path.join(exp_dir, file)

    # todo: delete this
    if file[:2] == 'm2':
        continue

    ######## LOAD EVERYTHING FROM FILE #########
    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
    
    loaded_data = np.load(os.path.join(exp, 'data.npz'))
    loaded_truth = np.load(os.path.join(exp, 'true_weights_and_likelihood.npz'))
    loaded_init = np.load(os.path.join(exp, 'init_theta_and_likelihood.npz'))
    loaded_vnce = np.load(os.path.join(exp, 'vnce_results.npz'))
    loaded_cd = np.load(os.path.join(exp, 'cd_results.npz'))
    loaded_nce = np.load(os.path.join(exp, 'nce_results.npz'))
    loaded_true = np.load(os.path.join(exp, 'true_weights_and_likelihood.npz'))
    
    X = loaded_data['X']
    Y = loaded_data['Y']
    
    true_theta = loaded_truth['true_theta']
    true_log_like = loaded_truth['ll']
    
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
    nce_reduced_times = loaded_nce['reduced_times']
    av_log_like_nce = loaded_nce['ll']
    
    rng = np.random.RandomState(config['random_seed'])
    
    # reconstruct models & optimisers
    model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
    nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
    
    var_dist = RBMLatentPosterior(theta0, rng=rng)
    
    noise = MultivariateBernoulliNoise(np.mean(X, axis=0))
    noise_log_like = np.mean(np.log(noise(X)))
    
    optimiser = VNCEOptimiser(model=model, noise=noise, variational_dist=var_dist, noise_samples=Y, 
                              sample_size=config['n'], nu=config['nu'], latent_samples_per_datapoint=config['nz'],
                              rng=rng)
    nce_optimiser = NCEOptimiser(model=nce_model, noise=noise, noise_samples=Y, nu=config['nu'])
    
    optimal_J1 = optimiser.evaluate_J1_at_param(theta=true_theta.reshape(-1), X=X)
    optimal_J = nce_optimiser.evaluate_J_at_param(true_theta.reshape(-1), X)
    
    if J1s.ndim == 2:
        J1s = np.sum(J1s, axis=1)
        Js_for_vnce_thetas = np.sum(Js_for_vnce_thetas, axis=1)
    
    ####### ACTUAL PLOTTING ###########
    
    # NCE OBJECTIVE PLOT
    ax = axs[0]
    #plot_and_annotate(ax, vnce_times, J1s, r"J1 {}".format(file), arrow_y_val)
    #arrow_y_val -= 0.01
    plot_and_annotate(ax, vnce_times, Js_for_vnce_thetas, r"J {}".format(file), arrow_y_val_2)
    #arrow_y_val_2 -= 0.01
    ax.set_xlabel('time (seconds)', fontsize=16)
    ax.set_ylabel('J1', fontsize=16)
    
    # LOG-LIKELIHOOD PLOT
    ax = axs[1]
    plot_and_annotate(ax, cd_reduced_times, av_log_like_cd, "CD {}".format(config['cd_num_steps']), arrow_y_val_4)
    arrow_y_val_4 -= 0.01
    plot_and_annotate(ax, vnce_reduced_times, av_log_like_vnce, "VNCE {}".format(file), arrow_y_val_3)
    arrow_y_val_3 -= 0.01
    
    ax.set_xlabel('time (seconds)', fontsize=16)
    ax.set_ylabel('log likelihood', fontsize=16)
    
    
    if i == 0:
        ax = axs[0]
        plot_and_annotate(ax, nce_times, Js, "J", arrow_y_val)
        ax.plot((vnce_times[0], vnce_times[-1]), (optimal_J1, optimal_J1), label=r'J1($\theta_{true}$)')
        ax.plot((vnce_times[0], vnce_times[-1]), (optimal_J, optimal_J), label=r'J($\theta_{true}$)')
        
        ax = axs[1]  
        plot_and_annotate(ax, nce_reduced_times, av_log_like_nce, "NCE", arrow_y_val_3)  
        ax.plot((0, plt.get(ax, 'xlim')[1]), (noise_log_like, noise_log_like),'g--', label='Noise distribution')
        ax.plot((0, plt.get(ax, 'xlim')[1]), (init_log_like, init_log_like),'r--', label='initial model')
        ax.plot((0, plt.get(ax, 'xlim')[1]), (true_log_like, true_log_like),'b--', label='True distribution')
    
    stats = sanity_check_stats(J1s, Js_for_vnce_thetas, file)     
    if i == 0:
        with open(os.path.join(save_dir, 'vnce_sanity_check.txt'), 'w') as f:
            f.write(stats)
    else:
        with open(os.path.join(save_dir, 'vnce_sanity_check.txt'), 'a') as f:
            f.write(stats)
            
    with open(os.path.join(save_dir, "config-{}.txt".format(file)), 'w') as f:
        for key, value in config.items():
            f.write("{}: {}\n".format(key, value))
            

axs[0].set_xlim(1e1, 1e4)
axs[0].set_ylim((optimal_J1 - 0.01, optimal_J1 + 0.03))

#axs[1].set_xlim((1e-1, 2e3))
axs[1].set_ylim((-7.2, -6.8))

axs[0].legend(loc='lower right')
axs[1].legend(loc='lower right')

fig.savefig(os.path.join(save_dir, 'partial-mstep-comparison.pdf'))


# ## plot two terms of J1/J separately

# In[ ]:


def create_J_diff_plot(J1s, times, Js_for_lnce_thetas, plot_posterior_ratio=False):
    """plot J (NCE objective function) minus J1 (lower bound to NCE objective)"""
    if plot_posterior_ratio:
        fig, axs = plt.subplots(2, 1, figsize=(15, 20))
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 20))
        axs = [axs]

    ax = axs[0]
    diff = np.sum(Js_for_lnce_thetas, axis=1) - np.sum(J1s, axis=1)
    ax.plot(optimiser.times, diff, c='k', label='J - J1')
    diff1 = Js_for_lnce_thetas[:, 0] - optimiser.J1s[:, 0]
    ax.plot(optimiser.times, diff1, c='r', label='term1: J - J1')
    diff2 = Js_for_lnce_thetas[:, 1] - optimiser.J1s[:, 1]
    ax.plot(optimiser.times, diff2, c='b', label='term2: J - J1')

    if plot_posterior_ratio:
        ax = axs[1]
        ax.plot(optimiser.times, optimiser.posterior_ratio_vars, c='k', label='V(p(z|y)/q(z|y))')

    for ax in axs:
        for time_id in optimiser.E_step_ids:
            time = optimiser.times[time_id]
            ax.plot((time, time), ax.get_ylim(), c='0.5')
        ax.set_xlabel('time (seconds)', fontsize=16)
        ax.legend()

    return fig, Js_for_lnce_thetas

#sorted_files = [str(file) for file in sorted([int(file) for file in os.listdir(exp_dir)])]
for i, file in enumerate(os.listdir(exp_dir)):
    exp = os.path.join(exp_dir, file)
    
    ######## LOAD EVERYTHING FROM FILE #########
    config = pickle.load(open(os.path.join(exp, 'config.p'), 'rb'))
    
    loaded_data = np.load(os.path.join(exp, 'data.npz'))
    loaded_truth = np.load(os.path.join(exp, 'true_weights_and_likelihood.npz'))
    loaded_init = np.load(os.path.join(exp, 'init_theta_and_likelihood.npz'))
    loaded_vnce = np.load(os.path.join(exp, 'vnce_results.npz'))
    loaded_nce = np.load(os.path.join(exp, 'nce_results.npz'))
    loaded_true = np.load(os.path.join(exp, 'true_weights_and_likelihood.npz'))
    
    X = loaded_data['X']
    Y = loaded_data['Y']
    
    true_theta = loaded_truth['true_theta']
    true_log_like = loaded_truth['ll']
    
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
    
    optimiser = VNCEOptimiser(model=model, noise=noise, variational_dist=var_dist, noise_samples=Y, 
                              sample_size=config['n'], nu=config['nu'], latent_samples_per_datapoint=config['nz'],
                              rng=rng)
    nce_optimiser = NCEOptimiser(model=nce_model, noise=noise, noise_samples=Y, nu=config['nu'])

    # todo: add everything from loaded files to optimiser, then put all this loading stuff into separate method
    # and clean up code generally
    
    optimal_J1 = optimiser.evaluate_J1_at_param(theta=true_theta.reshape(-1), X=X)
    optimal_J = nce_optimiser.evaluate_J_at_param(true_theta.reshape(-1), X)
    
    ####### ACTUAL PLOTTING ###########
    if config['optimiser'] == 'VNCEOptimiserWithoutImportanceSampling':
        J_plot, _ = create_J_diff_plot(optimiser, Js_for_vnce_thetas, plot_posterior_ratio=False)
        axs = J_plot.axes
        axs[0].set_xlim(-5, 50)
        axs[0].set_ylim((-0.05, 0.05))
    else:
        J_plot, _ = create_J_diff_plot(optimiser, Js_for_vnce_thetas, plot_posterior_ratio=True)
        axs = J_plot.axes
        axs[0].set_xlim(-5, 50)
        axs[0].set_ylim((-0.05, 0.05))
        axs[1].set_xlim(-5, 100)
        axs[1].set_ylim((0, 0.1))
        
    J_plot.suptitle('{}'.format(file))
    
#     ind1 = int(len(J1s_1)/2)
#     ind2 = int(len(J1s_2)/3)
#     annotate(ax, r"term1 J1 {}".format(file), (vnce_times[ind1], J1s_1[ind1]))
#     annotate(ax, r"term2 J1 {}".format(file), (vnce_times[ind2], J1s_2[ind2]))
        
    #axs[0].set_xlim((0, 1e1))
    
    J_plot.set_size_inches(15, 15)

    J_plot.savefig(os.path.join(save_dir, 'two-terms-of-J1-{}.pdf'.format(file)))

