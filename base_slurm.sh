#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH -J %%name%%
#SBATCH -o %%train_dir%%/slurm-%j.out
#SBATCH -e %%train_dir%%/slurm-%j.err

# Cluster Settings
#SBATCH -p %%partition%%
#SBATCH -n %%n%%
#SBATCH -c %%c%%
#SBATCH -t %%t%%

source %%home%%/.conda/envs/airhockey2/bin/activate airhockey2
# export PYTHONPATH="${PYTHONPATH}:%%home%%/airhockey2"

python %%home%%/airhockey/run.py --from_slurm --train_dir %%train_dir%% $@

EOT