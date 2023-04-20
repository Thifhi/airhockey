#!/bin/bash
#SBATCH -p single
#SBATCH -J training_test

# Cluster Settings
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 300

source /home/kit/stud/upgmi/airhockey/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/kit/stud/upgmi/airhockey"

python /home/kit/stud/upgmi/airhockey/lunar.py