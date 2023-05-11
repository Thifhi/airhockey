import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from ppo_airhockey_benchmark.train import start_training
from ppo_airhockey_benchmark.test import start_testing

debug = True
if debug:
    print("DEBUG"*10)

NAME="16k"
ENV="3dof-hit-interpolate"
REWARD_FUNC="custom_reward"
NUM_ENVS=12
debug_args = f"--name {NAME} --env {ENV} --reward_func {REWARD_FUNC} --num_envs {NUM_ENVS} --test"

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
    parser.add_argument("--hyperparameters", default=[], type=pair, nargs="*")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test", action="store_true")
    if not debug:
        args = parser.parse_args()
    else:
        args = parser.parse_args(debug_args.split())

    hyperparameters = {x:y for x,y in args.hyperparameters}
    if args.test:
        start_testing(args.name, args.env, args.reward_func)
    else:
        start_training(args.name, args.env, args.reward_func, args.num_envs, hyperparameters, args.load, args.checkpoint)
