import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from ppo_airhockey_benchmark.train import start_training

NAME="yarak"
ENV="3dof-hit"
REWARD_FUNC="custom_reward"
NUM_ENVS=12
debug = True
debug_args = f"--name {NAME} --env {ENV} --reward_func {REWARD_FUNC} --num_envs {NUM_ENVS}"

def pair(arg):
    return arg.split(':')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--reward_func", required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--hyperparameters", type=pair, nargs="*")
    if not debug:
        args = parser.parse_args()
    else:
        args = parser.parse_args(debug_args.split())
    start_training(args.name, args.env, args.reward_func, args.num_envs)
