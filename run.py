import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from ppo_airhockey_benchmark.train import start_training

NAME="yarak"
ENV="3dof-hit"
REWARD_FUNC="custom_reward"
NUM_ENVS=12
debug = False
debug_args = f"--name {NAME} --env {ENV} --reward_func {REWARD_FUNC} --num_envs {NUM_ENVS}"

def pair(arg):
    split = arg.split(':')
    if split[0] == "int":
        return (split[1], int(split[2]))
    elif split[0] == "float":
        return (split[1], float(split[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--reward_func", required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--hyperparameters", type=pair, nargs="*")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    if not debug:
        args = parser.parse_args()
    else:
        args = parser.parse_args(debug_args.split())

    hyperparameters = {x:y for x,y in args.hyperparameters}
    start_training(args.name, args.env, args.reward_func, args.num_envs, hyperparameters, args.load, args.checkpoint)
