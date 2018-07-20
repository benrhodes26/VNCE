"""Training script for experimental comparison of learning an RBM using
two methods: latent nce and contrastive divergence.
"""
import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

import numpy as np
import pickle

# my code
from contrastive_divergence_optimiser import CDOptimiser
from distribution import RBMLatentPosterior, MultivariateBernoulliNoise, ChowLiuTree
from fully_observed_models import VisibleRestrictedBoltzmannMachine
from latent_variable_model import RestrictedBoltzmannMachine
from nce_optimiser import NCEOptimiser
from plot import *
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, ScipyMinimiseEmStep, ExactEStep, MonteCarloVnceLoss, AdaptiveMonteCarloVnceLoss

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from numpy import random as rnd
from scipy.optimize import check_grad
from time import gmtime, strftime

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--data_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data/',
                    help='Path to directory where data is loaded and saved')
parser.add_argument('--save_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/rbm', help='Path to directory where model will be saved')

parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')

# Data arguments
parser.add_argument('--which_dataset', default='usps_4by4patches', help='options: usps and synthetic')
parser.add_argument('--n', type=int, default=10000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=int, default=1, help='ratio of noise to data samples in NCE')
parser.add_argument('--num_gibbs_steps', type=int, default=1000,
                    help='number of gibbs steps used to generate as synthetic dataset (if using synthetic)')

# Model arguments
parser.add_argument('--theta0_path', type=str, default=None, help='path to pre-trained weights')
parser.add_argument('--theta0scale', type=float, default=1.0, help='multiplier on initial weights of RBM.')
parser.add_argument('--true_sd', type=float, default=1.0, help='standard deviation of gaussian rv for synthetic ground truth weights')
parser.add_argument('--d', type=int, default=9, help='dimension of visibles (for synthetic dataset)')
parser.add_argument('--m', type=int, default=8, help='dimension of hiddens')

# Latent NCE optimisation arguments
parser.add_argument('--loss', type=str, default='AdaptiveMonteCarloVnceLoss', help='loss function class to use. See vnce_optimisers.py for options')
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
parser.add_argument('--use_importance_sampling', dest='use_importance_sampling', action='store_true', help='use importance sampling to estimate '
                    'the second term of the MonteCarloVnceLoss. If false, it must be possible to analytically marginalise out the latent variables in the model.')
parser.add_argument('--no-use_importance_sampling', dest='use_importance_sampling', action='store_false')
parser.set_defaults(use_importance_sampling=True)
parser.add_argument('--track_loss', dest='track_loss', action='store_true', help='track VNCE loss in E & M steps')
parser.add_argument('--no-track_loss', dest='track_loss', action='store_false')
parser.set_defaults(track_loss=False)

# Contrastive divergence optimisation arguments
parser.add_argument('--cd_num_steps', type=int, default=1, help='number of gibbs steps used to sample from model during learning with CD')
parser.add_argument('--cd_learn_rate', type=float, default=0.05, help='learning rate for contrastive divergence')
parser.add_argument('--cd_batch_size', type=int, default=100, help='number of datapoints used per gradient update')
parser.add_argument('--cd_num_epochs', type=int, default=1, help='number of passes through data set')

# nce optimisation arguments
parser.add_argument('--nce_opt_method', type=str, default='SGD', help='nce optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter_nce', type=int, default=500, help='number of iterations inside scipy.minimize')
parser.add_argument('--nce_num_epochs', type=int, default=1, help='if nce_opt_method=SGD, this is the number of passes through data set')
parser.add_argument('--nce_learn_rate', type=float, default=0.05, help='if nce_opt_method=SGD, this is the learning rate used')
parser.add_argument('--nce_batch_size', type=int, default=100, help='if nce_opt_method=SGD, this is the size of a minibatch')

# Other arguments
parser.add_argument('--num_log_like_steps', type=int, default=10, help='Number of time-steps for which we calculate log-likelihoods')
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=True)
parser.add_argument('--random_seed', type=int, default=1083463236, help='seed for np.random.RandomState')

args = parser.parse_args()
save_dir = os.path.join(args.save_dir, args.exp_name, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

"""
==========================================================================================================
                                            DATA
==========================================================================================================
"""
# For reproducibility
rng = rnd.RandomState(args.random_seed)

true_data_dist = None
# get training and test sets -either from synthetic or real data
if args.which_dataset == 'synthetic':
    n, d = args.n, args.d

    # generate weights of RBM that we want to learn
    true_theta = rng.randn(d + 1, args.m + 1) * args.true_sd
    true_theta[0, 0] = 0

    # generate synthetic training and test sets
    true_data_dist = RestrictedBoltzmannMachine(true_theta, rng=rng)
    X_train, Z = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
    X_test, _ = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
    X_mean = np.mean(X_train, axis=0)
else:
    loaded_data = np.load(os.path.join(args.data_dir, args.which_dataset + '.npz'))
    X_train = loaded_data['train']
    X_test = loaded_data['test']
    X_mean = np.mean(X_train, axis=0)
    n, d = X_train.shape

    if args.which_dataset[:4] == 'usps' and d <= 12:
        # Load empirical dist, which which treat as ground truth (this is reasonable in low dimensions)
        true_data_dist = pickle.load(open(os.path.join(args.data_dir, 'usps_3by3_emp_dist.p'), 'rb'))

"""
==========================================================================================================
                                       MODEL & NOISE
==========================================================================================================
"""
# initialise weights, either pre-trained or randomly initialised
if args.theta0_path:
    print('loading pre-trained theta0s')
    # vnce_theta0 = np.load(os.path.join(args.theta0_path, 'vnce_results.npz'))['vnce_thetas'][-1].reshape(d+1, args.m + 1)
    # nce_theta0 = np.load(os.path.join(args.theta0_path, 'nce_results.npz'))['nce_thetas'][1].reshape(d+1, args.m + 1)
    vnce_theta0 = np.load(os.path.join(args.theta0_path, 'cd_results.npz'))['cd_thetas'][-1].reshape(d+1, args.m + 1)
    cd_theta0 = np.load(os.path.join(args.theta0_path, 'cd_results.npz'))['cd_thetas'][-1].reshape(d+1, args.m + 1)
    nce_theta0 = np.load(os.path.join(args.theta0_path, 'cd_results.npz'))['cd_thetas'][-1].reshape(d+1, args.m + 1)
else:
    theta0 = np.asarray(
        rng.uniform(
            low=-args.theta0scale * np.sqrt(6. / (d + args.m)),
            high=args.theta0scale * np.sqrt(6. / (d + args.m)),
            size=(d + 1, args.m + 1)
        ))
    theta0[1:, 0] = np.log(X_mean + 1e-6 / (1 - X_mean + + 1e-6))  # visible biases
    theta0[0, 0] = -args.m * np.log(2) + np.sum(np.log(1 - X_mean + + 1e-6))  # scaling parameter
    # theta0[0, 1:] = 0  # hidden biases

    vnce_theta0 = theta0
    cd_theta0 = theta0
    nce_theta0 = theta0

# initialise models
model = RestrictedBoltzmannMachine(deepcopy(vnce_theta0), rng=rng)
cd_model = RestrictedBoltzmannMachine(deepcopy(cd_theta0), rng=rng)
nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(nce_theta0), rng=rng)
init_model = RestrictedBoltzmannMachine(deepcopy(vnce_theta0), rng=rng)  # rbm with initial parameters, for comparison

# noise distributions for V/NCE & sample from it
if args.noise == 'marginal':
    noise_dist = MultivariateBernoulliNoise(X_mean, rng=rng)
elif args.noise == 'chow_liu':
    noise_dist = ChowLiuTree(X_train, rng=rng)

# generate noise
Y = noise_dist.sample(int(n * args.nu))

"""
==========================================================================================================
                                LOSS FUNCTIONS AND OPTIMISERS
==========================================================================================================
"""

# initialise loss function
use_sgd = (args.opt_method == 'SGD')
if args.loss == 'MonteCarloVnceLoss':

    # variational distribution, which is the exact posterior in this case
    var_dist = RBMLatentPosterior(vnce_theta0, rng=rng)

    vnce_loss_function = MonteCarloVnceLoss(model=model,
                                            data=X_train,
                                            noise=noise_dist,
                                            noise_samples=Y,
                                            variational_noise=var_dist,
                                            noise_to_data_ratio=args.nu,
                                            num_latent_per_data=args.nz,
                                            use_minibatches=use_sgd,
                                            batch_size=args.batch_size,
                                            separate_terms=args.separate_terms,
                                            use_importance_sampling=args.use_importance_sampling,
                                            rng=rng)
elif args.loss == 'AdaptiveMonteCarloVnceLoss':

    # joint noise distribution for Adaptive V/NCE, which is the same class as the model
    variational_noise = RestrictedBoltzmannMachine(deepcopy(vnce_theta0), rng=rng)

    vnce_loss_function = AdaptiveMonteCarloVnceLoss(model=model,
                                                    data=X_train,
                                                    variational_noise=variational_noise,
                                                    noise_to_data_ratio=args.nu,
                                                    num_latent_per_data=args.nz,
                                                    num_mcmc_steps=args.num_gibbs_steps_for_adaptive_vnce,
                                                    use_minibatches=use_sgd,
                                                    batch_size=args.batch_size,
                                                    use_importance_sampling=args.use_importance_sampling,
                                                    separate_terms=args.separate_terms,
                                                    rng=rng)

if args.opt_method == 'SGD':
    m_step = SgdEmStep(do_m_step=True,
                       learning_rate=args.learn_rate,
                       num_batches_per_em_step=args.num_batch_per_em_step,
                       noise_to_data_ratio=args.nu,
                       track_loss=args.track_loss,
                       rng=rng)
else:
    m_step = ScipyMinimiseEmStep(do_m_step=True,
                                 optimisation_method=args.opt_method,
                                 max_iter=args.maxiter)

exact_e_step = ExactEStep(track_loss=args.track_loss)

optimiser = VemOptimiser(m_step=m_step, e_step=exact_e_step, num_em_steps_per_save=args.num_em_steps_per_save)
cd_optimiser = CDOptimiser(cd_model, rng=rng)
nce_optimiser = NCEOptimiser(model=nce_model, noise=noise_dist, noise_samples=Y, nu=args.nu)

"""
==========================================================================================================
                                            OPTIMISATION
==========================================================================================================
"""

# perform latent nce optimisation
print('starting latent nce optimisation...')
optimiser.fit(loss_function=vnce_loss_function,
              theta0=vnce_theta0.reshape(-1),
              alpha0=vnce_theta0.reshape(-1),
              stop_threshold=args.stop_threshold,
              max_num_em_steps=args.max_num_em_steps)
print('finished!')
print('starting cd optimisation...')
# perform contrastive divergence optimisation
_ = cd_optimiser.fit(X=X_train,
                     theta0=cd_theta0.reshape(-1),
                     num_gibbs_steps=args.cd_num_steps,
                     learning_rate=args.cd_learn_rate,
                     batch_size=args.cd_batch_size,
                     num_epochs=args.cd_num_epochs)
print('finished!')
print('starting nce optimisation...')
_ = nce_optimiser.fit(X=X_train,
                      theta0=nce_theta0.reshape(-1),
                      opt_method=args.nce_opt_method,
                      maxiter=args.maxiter_nce,
                      learning_rate=args.nce_learn_rate,
                      batch_size=args.nce_batch_size,
                      num_epochs=args.nce_num_epochs)
print('finished nce optimisation!')

"""
==========================================================================================================
                                        METRICS & PLOTTING
==========================================================================================================
"""
vnce_thetas, vnce_alphas, vnce_losses, vnce_times = optimiser.get_flattened_result_arrays()
m_step_ids, e_step_ids, _, _ = optimiser.get_m_and_e_step_ids()
vnce_val_losses = np.array(optimiser.val_losses)
vnce_plot = plot_vnce_loss(vnce_times, vnce_losses, vnce_val_losses, e_step_ids)
save_fig(vnce_plot, save_dir, 'vnce_loss')

nce_losses_for_vnce_params = None
J_plot = None
if args.separate_terms:
    nce_losses_for_vnce_params = get_nce_loss_for_vnce_params(X_train,
                                                              nce_optimiser,
                                                              optimiser,
                                                              separate_terms=args.separate_terms)
    J_plot = plot_nce_minus_vnce_loss(nce_losses_for_vnce_params,
                                      vnce_losses,
                                      vnce_times,
                                      e_step_ids)
    J_plot.savefig('{}/J-optimisation-curve.pdf'.format(save_dir))

# reduce results, since calculating log-likelihood is expensive
reduced_vnce_thetas, reduced_vnce_times = get_reduced_thetas_and_times(vnce_thetas, vnce_times[m_step_ids], args.num_log_like_steps)
reduced_cd_thetas, reduced_cd_times = get_reduced_thetas_and_times(cd_optimiser.thetas, cd_optimiser.times, args.num_log_like_steps)
reduced_nce_thetas, reduced_nce_times = get_reduced_thetas_and_times(nce_optimiser.thetas, nce_optimiser.times, args.num_log_like_steps)

# calculate average log-likelihood at each iteration for both models on test set
print('calculating log-likelihoods...')
av_log_like_vnce_train = get_av_log_like(reduced_vnce_thetas, init_model, X_train)
av_log_like_vnce_test = get_av_log_like(reduced_vnce_thetas, init_model, X_test)
av_log_like_cd_train = get_av_log_like(reduced_cd_thetas, init_model, X_train)
av_log_like_cd_test = get_av_log_like(reduced_cd_thetas, init_model, X_test)
av_log_like_nce_train = get_av_log_like(reduced_nce_thetas, init_model, X_train)
av_log_like_nce_test = get_av_log_like(reduced_nce_thetas, init_model, X_test)
init_model.theta = vnce_theta0.reshape(-1)
print('finished!')

# calculate average log-likelihood of empirical, initial & noise distributions on test set, for comparison
init_log_like = average_log_likelihood(init_model, X_test)
noise_log_like = np.mean(np.log(noise_dist(X_test) + 1e-6))
if args.which_dataset[:4] == 'usps' and true_data_dist:
    true_log_like = np.mean(np.log(true_data_dist(X_test)))
elif args.which_dataset == 'synthetic':
    true_log_like = average_log_likelihood(true_data_dist, X_test)

train_curves = [[reduced_vnce_times, av_log_like_vnce_train, 'vnce', 'blue'],
                [reduced_cd_times, av_log_like_cd_train, 'cd', 'red'],
                [reduced_nce_times, av_log_like_nce_train, 'nce', 'green']]
test_curves = [[reduced_vnce_times, av_log_like_vnce_test, 'vnce', 'blue'],
               [reduced_cd_times, av_log_like_cd_test, 'cd', 'red'],
               [reduced_nce_times, av_log_like_nce_test, 'nce', 'green']]
static_lines = [[init_log_like, 'initial'], [noise_log_like, 'noise']]
if true_data_dist:
    static_lines.append([true_log_like, 'empirical'])

# plot log-likelihood during training
plot_log_likelihood_learning_curves(train_curves, static_lines, save_dir, file_name='train')
plot_log_likelihood_learning_curves(test_curves, static_lines, save_dir, file_name='test')

# plot rbm weights for each model (including ground truth and random initialisation)
params = [model.theta, cd_model.theta, init_model.theta]
titles = ['Latent NCE parameters', 'Contrastive divergence parameters', 'Randomly initialised parameters']
rbm_weights_plot = plot_rbm_parameters(params, titles, d, args.m)
rbm_weights_plot.savefig('{}/rbm-weight-visualisation.pdf'.format(save_dir))

# check that latent nce has produced an approximately normalised model
model.reset_norm_const()
print('We hope that the normalisation constant of the learnt model is 1. In reality it is: {} '.format(
    model.norm_const))

END_TIME = strftime('%Y%m%d-%H%M', gmtime())

"""
==========================================================================================================
                                            SAVE TO FILE
==========================================================================================================
"""

# save everything
config = vars(args)
config['d'] = d
config.update({'start_time': START_TIME, 'end_time': END_TIME, 'random_seed': 1083463236})
with open(os.path.join(save_dir, "config.txt"), 'w') as f:
    for key, value in config.items():
        f.write("{}: {}\n".format(key, value))

pickle.dump(config, open(os.path.join(save_dir, "config.p"), "wb"))
pickle.dump(J_plot, open(os.path.join(save_dir, "J_plot.p"), "wb"))
pickle.dump(rbm_weights_plot, open(os.path.join(save_dir, "rbm_weights_plot.p"), "wb"))

pickle.dump(vnce_loss_function, open(os.path.join(save_dir, "vnce_loss_function.p"), "wb"))
pickle.dump(optimiser, open(os.path.join(save_dir, "vnce_optimiser.p"), "wb"))
pickle.dump(cd_optimiser, open(os.path.join(save_dir, "cd_optimiser.p"), "wb"))
pickle.dump(nce_optimiser, open(os.path.join(save_dir, "nce_optimiser.p"), "wb"))

np.savez(os.path.join(save_dir, "data"), X_train=X_train, X_test=X_test, Y=Y)
np.savez(os.path.join(save_dir, "init_theta_and_likelihood"), theta0=vnce_theta0, init_log_like=init_log_like)

np.savez(os.path.join(save_dir, "vnce_results"),
         vnce_thetas=vnce_thetas,
         vnce_alphas=vnce_alphas,
         vnce_times=vnce_times,
         vnce_losses=vnce_losses,
         reduced_vnce_thetas=reduced_vnce_thetas,
         reduced_vnce_times=reduced_vnce_times,
         av_log_like_vnce_train=av_log_like_vnce_train,
         av_log_like_vnce_test=av_log_like_vnce_test,
         m_step_ids=m_step_ids,
         e_step_ids=e_step_ids)

np.savez(os.path.join(save_dir, "cd_results"),
         cd_thetas=cd_optimiser.thetas,
         cd_times=cd_optimiser.times,
         reduced_cd_thetas=reduced_cd_thetas,
         reduced_cd_times=reduced_cd_times,
         av_log_like_cd_train=av_log_like_cd_train,
         av_log_like_cd_test=av_log_like_cd_test)

np.savez(os.path.join(save_dir, "nce_results"),
         nce_thetas=nce_optimiser.thetas,
         nce_times=nce_optimiser.times,
         nce_losses=nce_optimiser.Js,
         nce_losses_for_vnce_params=nce_losses_for_vnce_params,
         reduced_nce_thetas=reduced_nce_thetas,
         reduced_nce_times=reduced_nce_times,
         av_log_like_nce_train=av_log_like_nce_train,
         av_log_like_nce_test=av_log_like_nce_test)


