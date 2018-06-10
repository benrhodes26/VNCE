"""Training script for experimental comparison of learning an RBM using
two methods: latent nce and contrastive divergence.
"""
import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
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
from utils import *
from vnce_optimiser import VemOptimiser, SgdEmStep, ScipyMinimiseEmStep, ExactEStep, MonteCarloVnceLoss, AdaptiveMonteCarloVnceLoss

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from numpy import random as rnd
from time import gmtime, strftime

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--data_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data/usps',
                    help='Path to directory where data is loaded and saved')
# parser.add_argument('--save_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/rbm',
#                     help='Path to directory where model will be saved')
parser.add_argument('--save_dir', type=str, default='/home/ben/ben-rhodes-masters-project/proposal/experiments/rbm/synthetic/test',
                    help='Path to directory where model will be saved')

parser.add_argument('--exp_name', type=str, default='test', help='name of set of experiments this one belongs to')
parser.add_argument('--name', type=str, default=START_TIME, help='name of this exact experiment')

# Data arguments
parser.add_argument('--which_dataset', default='synthetic', help='options: usps and synthetic')
parser.add_argument('--n', type=int, default=10000, help='Number of datapoints')
parser.add_argument('--nz', type=int, default=1, help='Number of latent samples per datapoint')
parser.add_argument('--nu', type=float, default=1.0, help='ratio of noise to data samples in NCE')
parser.add_argument('--num_gibbs_steps', type=int, default=1000,
                    help='number of gibbs steps used to generate as synthetic dataset (if using synthetic)')

# Model arguments

parser.add_argument('--true_sd', type=float, default=1.0, help='standard deviation of gaussian rv for synthetic ground truth weights')
parser.add_argument('--d', type=int, default=9, help='dimension of visibles (for synthetic dataset)')
parser.add_argument('--m', type=int, default=4, help='dimension of hiddens')

# Latent NCE optimisation arguments
parser.add_argument('--loss', type=str, default='MonteCarloVnceLoss', help='loss function class to use. See vnce_optimisers.py for options')
parser.add_argument('--noise', type=str, default='marginal', help='type of noise distribution for latent NCE. Currently, this can be either marginals or chow-liu')
parser.add_argument('--opt_method', type=str, default='L-BFGS-B', help='optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter', type=int, default=10, help='number of iterations performed by L-BFGS-B optimiser inside each M step of EM')
parser.add_argument('--stop_threshold', type=float, default=0, help='Tolerance used as stopping criterion in EM loop')
parser.add_argument('--max_num_em_steps', type=int, default=10, help='Maximum number of EM steps to perform')
parser.add_argument('--learn_rate', type=float, default=0.1, help='if opt_method=SGD, this is the learning rate used')
parser.add_argument('--batch_size', type=int, default=100, help='if opt_method=SGD, this is the size of a minibatch')
parser.add_argument('--num_batch_per_em_step', type=int, default=1, help='if opt_method=SGD, this is the number of batches per EM step')
parser.add_argument('--num_em_steps_per_save', type=int, default=1, help='Every X EM steps save the current params, loss and time')
parser.add_argument('--num_gibbs_steps_for_adaptive_vnce', type=int, default=1, help='needed when sampling from joint noise distribution in adaptive vnce')
parser.add_argument('--use_importance_sampling', dest='use_importance_sampling', action='store_true', help='use importance sampling to estimate '
                    'the second term of the MonteCarloVnceLoss. If false, it must be possible to analytically marginalise out the latent variables in the model.')
parser.add_argument('--no-use_importance_sampling', dest='use_importance_sampling', action='store_false')
parser.set_defaults(use_importance_sampling=True)


# Contrastive divergence optimisation arguments
parser.add_argument('--cd_num_steps', type=int, default=1, help='number of gibbs steps used to sample from model during learning with CD')
parser.add_argument('--cd_learn_rate', type=float, default=0.1, help='learning rate for contrastive divergence')
parser.add_argument('--cd_batch_size', type=int, default=100, help='number of datapoints used per gradient update')
parser.add_argument('--cd_num_epochs', type=int, default=500, help='number of passes through data set')

# nce optimisation arguments
parser.add_argument('--nce_opt_method', type=str, default='SGD', help='nce optimisation method. L-BFGS-B and CG both seem to work')
parser.add_argument('--maxiter_nce', type=int, default=500, help='number of iterations inside scipy.minimize')
parser.add_argument('--nce_num_epochs', type=int, default=500, help='if nce_opt_method=SGD, this is the number of passes through data set')
parser.add_argument('--nce_learn_rate', type=float, default=0.1, help='if nce_opt_method=SGD, this is the learning rate used')
parser.add_argument('--nce_batch_size', type=int, default=100, help='if nce_opt_method=SGD, this is the size of a minibatch')

# Other arguments
parser.add_argument('--num_log_like_steps', type=int, default=50, help='Number of time-steps for which we calculate log-likelihoods')
parser.add_argument('--separate_terms', dest='separate_terms', action='store_true', help='separate the two terms that make up J1/J objective functions')
parser.add_argument('--no-separate_terms', dest='separate_terms', action='store_false')
parser.set_defaults(separate_terms=True)

args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir, args.exp_name, args.name)
os.makedirs(SAVE_DIR)

"""
==========================================================================================================
                                            SETUP
==========================================================================================================
"""

# For reproducibility
rng = rnd.RandomState(1083463236)

# get training and test sets
if args.which_dataset == 'usps':
    loaded_data = np.load(os.path.join(args.data_dir, 'usps_3by3patches.npz'))
    X = loaded_data['train']
    X_test = loaded_data['test']
    X_mean = np.mean(X, axis=0)
    n, d = X.shape

    # Load empirical dist, which which treat as ground truth (this is reasonable in low dimensions)
    true_data_dist = pickle.load(open(os.path.join(args.data_dir, 'usps_3by3_emp_dist.p'), 'rb'))

elif args.which_dataset == 'synthetic':
    n, d = args.n, args.d

    # generate weights of RBM that we want to learn
    true_theta = rng.randn(d + 1, args.m + 1) * args.true_sd
    true_theta[0, 0] = 0

    # generate synthetic training and test sets
    true_data_dist = RestrictedBoltzmannMachine(true_theta, rng=rng)
    X, Z = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
    X_test, _ = true_data_dist.sample(n, num_iter=args.num_gibbs_steps)
    X_mean = np.mean(X, axis=0)

else:
    print("Do not recognise which_dataset={}".format(args.which_dataset))
    raise TypeError

# initialise random weights, that depend on the data
theta0 = np.asarray(
    rng.uniform(
        low=-4 * np.sqrt(6. / (d + args.m)),
        high=4 * np.sqrt(6. / (d + args.m)),
        size=(d + 1, args.m + 1)
    ))
theta0[1:, 0] = np.log(X_mean / (1 - X_mean))  # visible biases
theta0[0, 0] = -args.m * np.log(2) + np.sum(np.log(1 - X_mean))  # scaling parameter
theta0[0, 1:] = 0  # hidden biases

# initialise models
model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
cd_model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
nce_model = VisibleRestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)
init_model = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)  # rbm with initial parameters, for comparison

# noise distributions for V/NCE & sample from it
if args.noise == 'marginal':
    noise_dist = MultivariateBernoulliNoise(X_mean, rng=rng)
elif args.noise == 'chow_liu':
    noise_dist = ChowLiuTree(X, rng=rng)

# generate noise
Y = noise_dist.sample(int(n * args.nu))

# initialise loss function
use_sgd = (args.opt_method == 'SGD')
if args.loss == 'MonteCarloVnceLoss':

    # variational distribution, which is the exact posterior in this case
    var_dist = RBMLatentPosterior(theta0, rng=rng)

    vnce_loss_function = MonteCarloVnceLoss(model=model,
                                            data=X,
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
    variational_noise = RestrictedBoltzmannMachine(deepcopy(theta0), rng=rng)

    vnce_loss_function = AdaptiveMonteCarloVnceLoss(model=model,
                                                    data=X,
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
                       rng=rng)
else:
    m_step = ScipyMinimiseEmStep(do_m_step=True,
                                 optimisation_method=args.opt_method,
                                 max_iter=args.maxiter)

exact_e_step = ExactEStep()

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
              theta0=theta0.reshape(-1),
              alpha0=theta0.reshape(-1),
              stop_threshold=args.stop_threshold,
              max_num_em_steps=args.max_num_em_steps)
print('finished!')
print('starting cd optimisation...')
# perform contrastive divergence optimisation
_ = cd_optimiser.fit(X=X,
                     theta0=theta0.reshape(-1),
                     num_gibbs_steps=args.cd_num_steps,
                     learning_rate=args.cd_learn_rate,
                     batch_size=args.cd_batch_size,
                     num_epochs=args.cd_num_epochs)
print('finished!')
print('starting nce optimisation...')
_ = nce_optimiser.fit(X=X,
                      theta0=theta0.reshape(-1),
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

nce_losses_for_vnce_params = None
J_plot = None
if args.separate_terms:
    nce_losses_for_vnce_params = get_nce_loss_for_vnce_params(X,
                                                              nce_optimiser,
                                                              optimiser,
                                                              separate_terms=args.separate_terms)
    J_plot = make_nce_minus_vnce_loss_plot(nce_losses_for_vnce_params,
                                           vnce_losses,
                                           vnce_times,
                                           e_step_ids)
    J_plot.savefig('{}/J-optimisation-curve.pdf'.format(SAVE_DIR))

# reduce results, since calculating log-likelihood is expensive
reduced_vnce_thetas, reduced_vnce_times = get_reduced_thetas_and_times(vnce_thetas, vnce_times[m_step_ids], args.num_log_like_steps)
reduced_cd_thetas, reduced_cd_times = get_reduced_thetas_and_times(cd_optimiser.thetas, cd_optimiser.times, args.num_log_like_steps)
reduced_nce_thetas, reduced_nce_times = get_reduced_thetas_and_times(nce_optimiser.thetas, nce_optimiser.times, args.num_log_like_steps)

# calculate average log-likelihood at each iteration for both models on test set
print('calculating log-likelihoods...')
av_log_like_vnce = optimiser.av_log_like_for_each_iter(X_test, loss_function=vnce_loss_function, thetas=reduced_vnce_thetas)
av_log_like_cd = cd_optimiser.av_log_like_for_each_iter(X_test, thetas=reduced_cd_thetas)
av_log_like_nce = nce_optimiser.av_log_like_for_each_iter(X_test, thetas=reduced_nce_thetas)
print('finished!')

# calculate average log-likelihood of empirical, initial & noise distributions on test set, for comparison
if args.which_dataset == 'usps':
    true_log_like = np.mean(np.log(true_data_dist(X_test)))
elif args.which_dataset == 'synthetic':
    true_log_like = average_log_likelihood(true_data_dist, X_test)
init_log_like = average_log_likelihood(init_model, X_test)
noise_log_like = np.mean(np.log(noise_dist(X_test)))

training_curves = [[reduced_vnce_times, av_log_like_vnce, 'vnce'],
                   [reduced_cd_times, av_log_like_cd, 'cd'],
                   [reduced_nce_times, av_log_like_nce, 'nce']]
static_lines = [[true_log_like, 'empirical'],
                [init_log_like, 'initial'],
                [noise_log_like, 'noise']]

# plot log-likelihood during training
like_training_plot = plot_log_likelihood_training_curves(training_curves, static_lines)

like_training_plot.gca().set_ylim((noise_log_like - 0.1, true_log_like + 0.05))
like_training_plot.savefig('{}/likelihood-optimisation-curve.pdf'.format(SAVE_DIR))

# plot rbm weights for each model (including ground truth and random initialisation)
params = [model.theta, cd_model.theta, init_model.theta]
titles = ['Latent NCE parameters', 'Contrastive divergence parameters', 'Randomly initialised parameters']
rbm_weights_plot = plot_rbm_parameters(params, titles, d, args.m)
rbm_weights_plot.savefig('{}/rbm-weight-visualisation.pdf'.format(SAVE_DIR))

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
config.update({'start_time': START_TIME, 'end_time': END_TIME, 'random_seed': 1083463236})
with open(os.path.join(SAVE_DIR, "config.txt"), 'w') as f:
    for key, value in config.items():
        f.write("{}: {}\n".format(key, value))

pickle.dump(config, open(os.path.join(SAVE_DIR, "config.p"), "wb"))
pickle.dump(like_training_plot, open(os.path.join(SAVE_DIR, "likelihood_training_plot.p"), "wb"))
pickle.dump(J_plot, open(os.path.join(SAVE_DIR, "J_plot.p"), "wb"))
pickle.dump(rbm_weights_plot, open(os.path.join(SAVE_DIR, "rbm_weights_plot.p"), "wb"))
pickle.dump(optimiser, open(os.path.join(SAVE_DIR, "vnce_optimiser.p"), "wb"))
pickle.dump(cd_optimiser, open(os.path.join(SAVE_DIR, "cd_optimiser.p"), "wb"))
pickle.dump(nce_optimiser, open(os.path.join(SAVE_DIR, "nce_optimiser.p"), "wb"))

np.savez(os.path.join(SAVE_DIR, "data"), X=X, X_test=X_test, Y=Y)
np.savez(os.path.join(SAVE_DIR, "init_theta_and_likelihood"), theta0=theta0, init_log_like=init_log_like)


np.savez(os.path.join(SAVE_DIR, "vnce_results"),
         vnce_thetas=vnce_thetas,
         vnce_alphas=vnce_alphas,
         vnce_times=vnce_times,
         vnce_losses=vnce_losses,
         reduced_vnce_thetas=reduced_vnce_thetas,
         reduced_vnce_times=reduced_vnce_times,
         av_log_like_vnce=av_log_like_vnce,
         m_step_ids=m_step_ids,
         e_step_ids=e_step_ids)

np.savez(os.path.join(SAVE_DIR, "cd_results"),
         cd_thetas=cd_optimiser.thetas,
         cd_times=cd_optimiser.times,
         reduced_cd_thetas=reduced_cd_thetas,
         reduced_cd_times=reduced_cd_times,
         av_log_like_cd=av_log_like_cd)

np.savez(os.path.join(SAVE_DIR, "nce_results"),
         nce_thetas=nce_optimiser.thetas,
         nce_times=nce_optimiser.times,
         nce_losses=nce_optimiser.Js,
         nce_losses_for_vnce_params=nce_losses_for_vnce_params,
         reduced_nce_thetas=reduced_nce_thetas,
         reduced_nce_times=reduced_nce_times,
         av_log_like_nce=av_log_like_nce)


