import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import fancy_gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor, VecNormalize)
from pathlib import Path
from stable_baselines3 import PPO
import random
import pickle

MODEL = "PPO/SecondOrderInterpolation/gamma0.997,lr5.e-5,toleration0.2"
LOAD_PATH = Path("logs") / MODEL
ENV = "3dof-hit-pos-gather-data"
ENV_ARGS = {
    "reward_coefficient": 0.001,
    "toleration": 0.2
}
NUM_ENV = 12
NUM_ITER = int(1e8)
FLUSH_INTERVAL = 1e2


def make_env(env_id: str, rank: int, seed: int = 0, **kwargs):
    def _init():
        env = fancy_gym.make(env_id, seed=seed + rank, **kwargs)
        return env

    set_random_seed(seed)
    return _init

def flush_all_episodes(all_episodes):
    model_data_path = Path("step_based_ik/data") / MODEL
    os.makedirs(model_data_path, exist_ok=True)
    c = 1
    while True:
        save_path = model_data_path / (str(c) + ".pkl")
        if not save_path.exists():
            with save_path.open("wb") as f:
                pickle.dump(all_episodes, f)
            all_episodes.clear()
            break
        else:
            c += 1
            continue

def start_data_gathering():
    eval_env = VecMonitor(SubprocVecEnv([make_env(ENV, rank=i, seed=random.randint(0, 1e9), **ENV_ARGS) for i in range(NUM_ENV)]))
    eval_env = VecNormalize.load(LOAD_PATH / "vecnormalize.pkl", eval_env)
    model = PPO.load(LOAD_PATH / "best_model.zip")

    all_episodes = []
    
    cur_episodes = [[] for _ in range(NUM_ENV)]

    obs = eval_env.reset()
    for step in range(NUM_ITER):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = eval_env.step(action)
        for i, (i_done, i_info) in enumerate(zip(done, info)):
            cur_episodes[i].append(i_info["ee_vel"].tolist())
            if i_done:
                all_episodes.append(cur_episodes[i])
                cur_episodes[i] = []
                if len(all_episodes) >= FLUSH_INTERVAL:
                    flush_all_episodes(all_episodes)
        if step % 1e2 == 0:
            print(f"Step: {step}")


if __name__ == "__main__":
    start_data_gathering()