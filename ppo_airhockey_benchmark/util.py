import fancy_gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecNormalize)

import wandb

def make_env(env_id: str, rank: int, seed: int = 0, **kwargs):
    def _init():
        env = fancy_gym.make(env_id, seed=seed + rank, **kwargs)
        return env

    set_random_seed(seed)
    return _init

def make_environments(env, num_envs, env_args, gamma=1, only_eval=False, load=False, load_dir=None):
    if only_eval:
        eval_env = VecNormalize.load(load_dir, DummyVecEnv([lambda: fancy_gym.make(env, seed=1, **env_args)]))
        eval_env.norm_reward = False
        eval_env.training = False
        return eval_env
    eval_env = VecNormalize(VecMonitor(DummyVecEnv([lambda: fancy_gym.make(env, seed=0, **env_args)])), training=False, norm_reward=False)
    train_env = VecMonitor(SubprocVecEnv([make_env(env_id=env, rank=i, **env_args) for i in range(num_envs)]))
    if not load:
        train_env = VecNormalize(train_env, gamma=gamma)
    else:
        train_env = VecNormalize.load(load_dir, train_env)
    return train_env, eval_env

def setup_wandb(train_dir, config):
    wandb_cfg = config["wandb"]
    wandb.init(dir=train_dir, **wandb_cfg, group=config["group"], job_type=config["job_type"], name=config["name"], sync_tensorboard=True, config=config)