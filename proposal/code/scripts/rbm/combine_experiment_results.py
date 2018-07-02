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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from numpy import random as rnd
from latent_variable_model import RestrictedBoltzmannMachine
from utils import *

parser = ArgumentParser(description='Experimental comparison of training an RBM using latent nce and contrastive divergence',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/rbm/adaptive/adaptive_finetuned_combined/')
parser.add_argument('--exp_name', type=str, default=None, help='name of set of experiments this one belongs to')
parser.add_argument('--load_dir', type=str, default='/disk/scratch/ben-rhodes-masters-project/experimental-results/rbm/adaptive/adaptive_finetuned/')
parser.add_argument('--results_1', type=str, default=None)
parser.add_argument('--results_2', type=str, default=None)
parser.add_argument('--visible_dim', type=int, default=25)
parser.add_argument('--num_log_like_steps', type=int, default=50, help='Number of time-steps for which we calculate log-likelihoods')

args = parser.parse_args()
save_dir = os.path.join(args.save_dir, args.exp_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def combine_times(times_1, times_2):
    times_2 += times_1[-1]
    combined_times = np.concatenate((times_1, times_2))
    return combined_times

loaded_data = np.load(os.path.join(args.load_dir, args.results_1, 'data.npz'))
X_test = loaded_data['X_test']

loaded_1 = np.load(os.path.join(args.load_dir, args.results_1, 'cd_results.npz'))
cd_times_1 = loaded_1['cd_times']
cd_thetas_1 = loaded_1['cd_thetas']

loaded_2 = np.load(os.path.join(args.load_dir, args.results_2, 'cd_results.npz'))
cd_times_2 = loaded_2['cd_times']
cd_thetas_2 = loaded_2['cd_thetas']

loaded_3 = np.load(os.path.join(args.load_dir, args.results_2, 'vnce_results.npz'))
vnce_m_steps_ids = loaded_3['m_step_ids']
vnce_times = loaded_3['vnce_times'][vnce_m_steps_ids]
vnce_thetas = loaded_3['vnce_thetas']

cd_times = combine_times(cd_times_1, cd_times_2)
cd_thetas = np.concatenate((cd_thetas_1, cd_thetas_2), axis=0)

vnce_fine_tuned_times = combine_times(cd_times_1, vnce_times)
vnce_fine_tuned_thetas = np.concatenate((cd_thetas_1, vnce_thetas), axis=0)

# get cd results over the same time interval as the vnce results and then reduce them both, since calculating log-likelihood is expensive
if cd_times[-1] < vnce_fine_tuned_times[-1]:
    print('******************************************************************************************************************************\n'
          'WARNING: Expected CD to have run for at least as long as the fine-tuned vnce (so we can plot them over the same time interval)\n'
          '*******************************************************************************************************************************')
max_time = vnce_fine_tuned_times[-1]
cd_max_ind = take_closest(cd_times, max_time)
cd_times, cd_thetas = cd_times[:cd_max_ind], cd_thetas[:cd_max_ind]

reduced_vnce_thetas, reduced_vnce_times = get_reduced_thetas_and_times(vnce_fine_tuned_thetas, vnce_fine_tuned_times, args.num_log_like_steps, max_time)
reduced_cd_thetas, reduced_cd_times = get_reduced_thetas_and_times(cd_thetas, cd_times, args.num_log_like_steps, max_time)

# create an RBM which we can use to calculate the log-likelihoods
config = pickle.load(open(os.path.join(args.load_dir, args.results_1, 'config.p'), 'rb'))
model = RestrictedBoltzmannMachine(np.zeros((config['d'] + 1, config['m'] + 1)))

# calculate average log-likelihood at each iteration for both models on test set
print('calculating log-likelihoods...')
av_log_like_cd = np.zeros(len(reduced_cd_thetas))
for i in np.arange(0, len(reduced_cd_thetas)):
    model.theta = deepcopy(reduced_cd_thetas[i])
    av_log_like_cd[i] = average_log_likelihood(model, X_test)

av_log_like_vnce = np.zeros(len(reduced_vnce_thetas))
for i in np.arange(0, len(reduced_vnce_thetas)):
    model.theta = deepcopy(reduced_vnce_thetas[i])
    av_log_like_vnce[i] = average_log_likelihood(model, X_test)
print('finished!')

training_curves = [[reduced_vnce_times, av_log_like_vnce, 'vnce'],
                   [reduced_cd_times, av_log_like_cd, 'cd']]
# plot log-likelihood during training
like_training_plot = plot_log_likelihood_training_curves(training_curves, [])
like_training_plot.gca().set_ylim((av_log_like_cd.max() - 1.7, av_log_like_cd.max() + 0.3))
like_training_plot.savefig('{}/likelihood-optimisation-curve.pdf'.format(save_dir))
pickle.dump(like_training_plot, open(os.path.join(save_dir, "likelihood_training_plot.p"), "wb"))

pickle.dump(config, open(os.path.join(save_dir, "config.p"), "wb"))
np.savez(os.path.join(save_dir, "data"), **loaded_data)
np.savez(os.path.join(save_dir, "vnce_results"),
         vnce_thetas=vnce_fine_tuned_thetas,
         vnce_times=vnce_fine_tuned_times,
         reduced_vnce_thetas=reduced_vnce_thetas,
         reduced_vnce_times=reduced_vnce_times,
         av_log_like_vnce=av_log_like_vnce)

np.savez(os.path.join(save_dir, "cd_results"),
         cd_thetas=cd_thetas,
         cd_times=cd_times,
         reduced_cd_thetas=reduced_cd_thetas,
         reduced_cd_times=reduced_cd_times,
         av_log_like_cd=av_log_like_cd)

# these are not actually used - we just save them to match the output structure of the original experiment, so we can run make-combined-experiment-plot.py
loaded_nce = np.load(os.path.join(args.load_dir, args.results_1, 'nce_results.npz'))
np.savez(os.path.join(save_dir, "nce_results"), **loaded_nce)

loaded_init = np.load(os.path.join(args.load_dir, args.results_1, 'init_theta_and_likelihood.npz'))
np.savez(os.path.join(save_dir, "init_theta_and_likelihood"), **loaded_init)

