import json
import pathlib
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import sync_envs_normalization

from .callbacks import CustomEvalCallback, SaveBestModel, CheckpointLog
from .constants import log_dir, best_model_file_name, checkpoint_dir, vecnormalize_file_name
from .util import make_environments


def start_testing(name, env, reward_func):
    base_path = pathlib.Path(__file__).parent.parent.resolve()
    model_dir = base_path / log_dir / name
    eval_env = make_environments(env, reward_func, 0, only_eval=True, load=True, load_dir=model_dir / vecnormalize_file_name)
    model_load_dir = model_dir / best_model_file_name
    model = PPO.load(model_load_dir)
    for q in range(50):
        obs = eval_env.reset()
        cum_reward = 0
        while True:
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            cum_reward += reward
            eval_env.render()
            if done:
                print("Run {}".format(q))
                print("Reward until last:{}".format(cum_reward - reward))
                print("Full reward:{}".format(cum_reward))
                break
