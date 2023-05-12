#!/bin/bash
sbatch <<EOT

#SBATCH -J %%name%%

# Cluster Settings
#SBATCH -p %%partition%%
#SBATCH -n %%n%%
#SBATCH -c %%c%%
#SBATCH -t %%t%%

source %%home%%/airhockey/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:%%home%%/airhockey"

python %%home%%/airhockey/run.py _from_slurm $@

EOT