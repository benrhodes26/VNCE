#!/bin/bash
# remember: specify exp_name (and maybe name) and how accurately to compute log-like
python run_experiment.py --maxiter=1 --max_num_em_steps=1000 --name=1 --nz=1 --exp_name=maxiter_comparison-5by5 --d=5 --m=5 --cd_num_steps=1
python run_experiment.py --maxiter=2 --max_num_em_steps=1000 --name=2 --nz=1 --exp_name=maxiter_comparison-5by5 --d=5 --m=5 --cd_num_steps=2
python run_experiment.py --maxiter=5 --max_num_em_steps=1000 --name=5 --nz=1 --exp_name=maxiter_comparison-5by5 --d=5 --m=5 --cd_num_steps=3
python run_experiment.py --maxiter=10 --max_num_em_steps=500 --name=10 --nz=1 --exp_name=maxiter_comparison-5by5 --d=5 --m=5 --cd_num_steps=4
python run_experiment.py --maxiter=20 --max_num_em_steps=500 --name=20 --nz=1 --exp_name=maxiter_comparison-5by5 --d=5 --m=5 --cd_num_steps=5
python run_experiment.py --maxiter=50 --max_num_em_steps=500 --name=50 --nz=1 --exp_name=maxiter_comparison-5by5 --d=5 --m=5 --cd_num_steps=10

