#!/bin/bash
# remember: specify exp_name (and maybe name) and how accurately to compute log like
python run_experiment.py --maxiter=1 --max_num_em_steps=1000 --cd_num_epochs=0 --name=1 --nz=10 --opt_method=CG --save_dir=/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/partial-m-step-with-cg-nz=10
python run_experiment.py --maxiter=2 --max_num_em_steps=750 --cd_num_epochs=0 --name=2 --nz=10 --opt_method=CG --save_dir=/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/partial-m-step-with-cg-nz=10
python run_experiment.py --maxiter=5 --max_num_em_steps=550 --cd_num_epochs=0 --name=5 --nz=10 --opt_method=CG --save_dir=/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/partial-m-step-with-cg-nz=10
python run_experiment.py --maxiter=10 --max_num_em_steps=400 --cd_num_epochs=0 --name=10 --nz=10 --opt_method=CG --save_dir=/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/partial-m-step-with-cg-nz=10
python run_experiment.py --maxiter=20 --max_num_em_steps=200 --cd_num_epochs=0 --name=20 --nz=10 --opt_method=CG --save_dir=/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/partial-m-step-with-cg-nz=10
python run_experiment.py --maxiter=50 --max_num_em_steps=100 --cd_num_epochs=0 --name=50 --nz=10 --opt_method=CG --save_dir=/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/partial-m-step-with-cg-nz=10

