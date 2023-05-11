import json
import pathlib
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import sync_envs_normalization

from .callbacks import CustomEvalCallback, SaveBestModel, CheckpointLog
from .constants import log_dir, custom_log_file_name, checkpoint_dir
from .util import make_environments


def start_training(name, env, reward_func, num_envs, hyperparameters={}, load=False, checkpoint=None):
    base_path = pathlib.Path(__file__).parent.parent.resolve()
    model_dir = base_path / log_dir / name
    if load:
        with open(model_dir / custom_log_file_name, "r") as f:
            custom_log = json.load(f)
        train_env, eval_env = make_environments(env, reward_func, num_envs, load=True, load_dir=model_dir / checkpoint_dir / (checkpoint + ".pkl"))
        model_load_dir = model_dir / checkpoint_dir / (checkpoint + ".zip")
        model = PPO.load(model_load_dir, train_env, verbose=1, tensorboard_log=model_dir, **hyperparameters)
    else:
        os.makedirs(model_dir)
        os.makedirs(model_dir / checkpoint_dir)
        custom_log = {
            "best_mean_reward": -1e10 # A very large value
        }
        train_env, eval_env = make_environments(env, reward_func, num_envs)
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=model_dir, **hyperparameters)
    sync_envs_normalization(train_env, eval_env)
    callbacks = CallbackList([CheckpointLog(model_dir, custom_log, int(10000/num_envs)), CustomEvalCallback(eval_env, SaveBestModel(model_dir, custom_log), 10, int(1000/num_envs), custom_log["best_mean_reward"])])
    model.learn(total_timesteps=2e8, progress_bar=True, reset_num_timesteps=False, tb_log_name="run", callback=callbacks)

