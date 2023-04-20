#!/bin/bash
#SBATCH -p single
#SBATCH -J training_test

# Cluster Settings
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -t 600

source /home/kit/stud/upgmi/airhockey/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/kit/stud/upgmi/airhockey"

python /home/kit/stud/upgmi/airhockey/main.py