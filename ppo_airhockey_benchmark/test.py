import pathlib
import numpy as np

import yaml
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from .constants import best_model_file_name, vecnormalize_file_name
from .util import make_environments


def start_testing():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = config["local_testing"]["env"]
    env_args = config["local_testing"]["env_args"]
    path = pathlib.Path(config["local_testing"]["path"])
    eval_env = make_environments(env, 0, env_args, only_eval=True, load=True, load_dir=path / vecnormalize_file_name)
    model_load_dir = path / best_model_file_name
    if config["model_type"] == 'recurrent':
        model = RecurrentPPO.load(model_load_dir)
    else:
        model = PPO.load(model_load_dir)
    for q in range(50):
        obs = eval_env.reset()
        cum_reward = 0
        if config["model_type"] == 'recurrent':
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,), dtype=bool)
        while True:
            if config["model_type"] == 'recurrent':
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            else:
                action, state = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            cum_reward += reward
            eval_env.render()
            if done:
                print("Run {}".format(q))
                print("Reward until last:{}".format(cum_reward - reward))
                print("Full reward:{}".format(cum_reward))
                break
