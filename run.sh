#!/bin/bash
#SBATCH -p single
#SBATCH -J training_test

# Cluster Settings
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 1200

NAME="interpolate1"
ENV="3dof-hit-interpolate"
REWARD_FUNC="custom_reward"
NUM_ENVS=120
HYPERPARAMETERS="n_steps:32000 batch_size:256 learning_rate:3e-4"

source /home/kit/stud/upgmi/airhockey/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/kit/stud/upgmi/airhockey"

python /home/kit/stud/upgmi/airhockey/run.py --name $NAME --env $ENV --reward_func $REWARD_FUNC --num_envs $NUM_ENVS --hyperparameters $HYPERPARAMETERS $@