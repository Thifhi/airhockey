NAME="yarak"
ENV="3dof-hit"
REWARD_FUNC="custom_reward"
NUM_ENVS=12

python run.py --name $NAME --env $ENV --reward_func $REWARD_FUNC --num_envs $NUM_ENVS