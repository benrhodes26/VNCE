import os
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description='make training scripts for adaptive VNCE applied to an RBM', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='~/masters-project/ben-rhodes-masters-project/proposal/code/scripts/rbm/bash_scripts/',
                    help='Path to directory where scripts will be saved')

args = parser.parse_args()
save_dir = os.path.expanduser(args.save_dir) + "adaptive_finetuned_test/"
# save_dir = os.path.expanduser(args.save_dir) + "adaptive_finetuned_lr=0.001/"
# save_dir = os.path.expanduser(args.save_dir) + "adaptive_finetuned/"
os.makedirs(save_dir)

data_sets = ['usps', 'adult', 'connect4', 'dna', 'mushrooms', 'nips', 'ocr_letters', 'rcv1', 'web']
# learn_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
learn_rates = [0.001]

for data_set in data_sets:
    for learn_rate in learn_rates:
        command_0 = "#!/bin/bash \n"
        command_1 = \
        "python $HOME/masters-project/ben-rhodes-masters-project/proposal/code/scripts/rbm/rbm_experiment.py \
        --exp_name=adaptive/adaptive_finetuned/{0}/ \
        --name={1}/ \
        --which_dataset={0} \
        --m=18 \
        --cd_num_epochs=5 \
        --cd_learn_rate={1} \
        --max_num_em_steps=1 \
        --nce_num_epochs=1 \
        --loss=AdaptiveMonteCarloVnceLoss \
        --no-separate_terms \
        --opt_method=SGD \
        --num_batch_per_em_step=1 \
        --num_gibbs_steps_for_adaptive_vnce=1 \
        --num_log_like_steps=1 \n".format(data_set, learn_rate)

        command_2 = \
        "python $HOME/masters-project/ben-rhodes-masters-project/proposal/code/scripts/rbm/rbm_experiment.py \
        --exp_name=adaptive/adaptive_finetuned/{0}/ \
        --name=fine_tuned_{1}/ \
        --which_dataset={0} \
        --m=18 \
        --loss=AdaptiveMonteCarloVnceLoss \
        --num_em_steps_per_save=1 \
        --nce_num_epochs=1 \
        --no-separate_terms \
        --theta0_path='/disk/scratch/ben-rhodes-masters-project/experimental-results/rbm/adaptive/adaptive_finetuned/{0}/{1}' \
        --opt_method=L-BFGS-B \
        --max_num_em_steps=1 \
        --maxiter=1 \
        --num_gibbs_steps_for_adaptive_vnce=50 \
        --cd_learn_rate={1} \
        --cd_num_epochs=1 \
        --cd_num_steps=50 \
        --num_log_like_steps=1 \n".format(data_set, learn_rate)

        command_3 = \
        "python $HOME/masters-project/ben-rhodes-masters-project/proposal/code/scripts/rbm/combine_experiment_results.py \
        --exp_name={0}/{1}/ \
        --results_1={0}/{1}/ \
        --results_2={0}/fine_tuned_{1} \n".format(data_set, learn_rate)

        command_4 = \
        "python $HOME/masters-project/ben-rhodes-masters-project/proposal/code/scripts/rbm/make_combined_adaptive_experiment_plot.py \
        --exp_name={0}/ \n".format(data_set)

        script = command_0 + command_1 + command_2 + command_3 + command_4
        with open(save_dir + data_set + '.sh', 'a') as f:
            f.write(script)
