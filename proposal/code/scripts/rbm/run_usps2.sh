#!bin/bash

# SGD
#python usps_experiment.py \
# --save_dir=/home/ben/ben-rhodes-masters-project/proposal/experiments/rbm/synthetic/test \
#--which_dataset=synthetic \
#--m=2 \
#--loss=MonteCarloVnceLoss \
#--opt_method=SGD \
#--max_num_em_steps=50000 \
#--learn_rate=0.1 \
#--num_em_steps_per_save=1000 \
#--cd_num_epochs=1 \
#--nce_opt_method=SGD \
#--nce_num_epochs=1 \
#--no-separate_terms

# No importance sampling SGD
#python usps_experiment.py \
# --save_dir=/home/ben/ben-rhodes-masters-project/proposal/experiments/rbm/synthetic/test \
#--which_dataset=synthetic \
#--m=2 \
#--loss=MonteCarloVnceLoss \
#--opt_method=SGD \
#--max_num_em_steps=50000 \
#--num_em_steps_per_save=1000 \
#--no-use_importance_sampling \
#--cd_num_epochs=1 \
#--nce_opt_method=SGD \
#--nce_num_epochs=1 \
#--no-separate_terms

# L-BFGS-B
python usps_experiment.py \
--save_dir=/home/ben/ben-rhodes-masters-project/proposal/experiments/rbm/synthetic/test \
--which_dataset=synthetic \
--m=2 \
--loss=MonteCarloVnceLoss \
--opt_method=L-BFGS-B \
--max_num_em_steps=1 \
--num_em_steps_per_save=1 \
--cd_num_epochs=1 \
--nce_opt_method=L-BFGS-B \
--maxiter_nce=100