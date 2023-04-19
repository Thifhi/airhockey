from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize, sync_envs_normalization
import os
import time
import fancy_gym
import imageio
import torch

import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import json

def main():
    load = True
    test = True
    cpu_n = 3
    log_dir = "logs/"
    name = "enes_reward_4.1"
    log_dir = os.path.join(log_dir, name)
    best_model_dir = os.path.join(log_dir, "best_model")
    vec_normalize_data_dir = os.path.join(log_dir, "vec_normalize_data")
    custom_log_dir = os.path.join(log_dir, "custom_log.json")
    os.makedirs(log_dir, exist_ok=True)

    train_env = VecMonitor(SubprocVecEnv([lambda: fancy_gym.make("3dof-hit", seed=0) for _ in range(cpu_n)]))
    eval_env = VecNormalize(VecMonitor(DummyVecEnv([lambda: fancy_gym.make("3dof-hit", seed=1)])), training=False, norm_reward=False)

    if load:
        print("Loaded")
        with open(custom_log_dir, "r") as f:
            custom_log = json.load(f)
        train_env = VecNormalize.load(vec_normalize_data_dir, train_env)
        model = PPO.load(best_model_dir, env=train_env)
        model.num_timesteps = custom_log["num_timesteps"]
    else:
        print("Created new")
        custom_log = {"best_mean_reward": 0}
        train_env = VecNormalize(train_env)
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir)
    sync_envs_normalization(train_env, eval_env)

    if test:
        while True:
            obs = eval_env.reset()
            reward = 0
            for i in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                # action = np.array([eval_env.action_space.sample()])
                obs, _rewards, dones, _info = eval_env.step(action)
                reward += _rewards
                eval_env.render()
                time.sleep(0.01)
                if dones:
                    print("until:{}".format(reward - _rewards))
                    print("full:{}".format(reward))
                    break
    
    class SaveCustomLog(BaseCallback):
        def _on_step(self) -> bool:
            custom_log["num_timesteps"] = model.num_timesteps
            with open(custom_log_dir, "w") as f:
                json.dump(custom_log, f)
            return True
        
    eval_callback = EvalCallback(eval_env, n_eval_episodes=40, log_path=log_dir, eval_freq=10000, deterministic=True, render=False, callback_after_eval=SaveCustomLog())

    class SaveBestModel(BaseCallback):
        def _on_step(self) -> bool:
            model.save(best_model_dir)
            train_env.save(vec_normalize_data_dir)
            custom_log["best_mean_reward"] = float(eval_callback.best_mean_reward)
            return True

    eval_callback.callback_on_new_best = SaveBestModel()
    eval_callback.best_mean_reward = custom_log["best_mean_reward"]
    while True:
        model.learn(total_timesteps=int(1e7), progress_bar=True, reset_num_timesteps=False, tb_log_name="run", callback=eval_callback)


if __name__ == '__main__':
    main()