NAME="16k"
ENV="3dof-hit-interpolate"
REWARD_FUNC="custom_reward"
NUM_ENVS=12
HYPERPARAMETERS="int:n_steps:50 float:learning_rate:0.0001"

python run.py --name $NAME --env $ENV --reward_func $REWARD_FUNC --num_envs $NUM_ENVS --hyperparameters $HYPERPARAMETERS $@