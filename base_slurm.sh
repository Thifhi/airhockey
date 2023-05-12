#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH -J %%name%%
#SBATCH -e %%train_dir%%/slurm-%j.out
#SBATCH -e %%train_dir%%/slurm-%j.err

# Cluster Settings
#SBATCH -p %%partition%%
#SBATCH -n %%n%%
#SBATCH -c %%c%%
#SBATCH -t %%t%%

source %%home%%/airhockey/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:%%home%%/airhockey"

python %%home%%/airhockey/run.py --from_slurm $@

EOT