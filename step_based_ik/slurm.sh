#!/bin/bash

#SBATCH -J gather
#SBATCH -o /pfs/data5/home/kit/stud/upgmi/airhockey/step_based_ik/slurm-%j.out
#SBATCH -e /pfs/data5/home/kit/stud/upgmi/airhockey/step_based_ik/slurm-%j.err

# Cluster Settings
#SBATCH -p single
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -t 600

source /home/kit/stud/upgmi/airhockey/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/kit/stud/upgmi/airhockey"

python /home/kit/stud/upgmi/airhockey/step_based_ik/gather_data.py
