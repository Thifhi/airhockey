import fancy_gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecNormalize)


def make_env(env_id: str, rank: int, seed: int = 0, **kwargs):
    def _init():
        env = fancy_gym.make(env_id, seed=seed + rank, **kwargs)
        return env

    set_random_seed(seed)
    return _init

def make_environments(env, reward_func, num_envs, only_eval=False, load=False, load_dir=None):
    if only_eval:
        eval_env = VecNormalize.load(load_dir, DummyVecEnv([lambda: fancy_gym.make(env, seed=0)]))
        eval_env.norm_reward = False
        eval_env.training = False
        return eval_env
    eval_env = VecNormalize(VecMonitor(DummyVecEnv([lambda: fancy_gym.make(env, seed=0)])), training=False, norm_reward=False)
    train_env = VecMonitor(SubprocVecEnv([make_env(env_id=env, rank=i) for i in range(num_envs)]))
    if not load:
        train_env = VecNormalize(train_env)
    else:
        train_env = VecNormalize.load(load_dir, train_env)
    return train_env, eval_env

def process_hyperparameters(hyperparameters, num_envs):
    res = {}
    for k,v in hyperparameters:
        if k == "n_steps":
            res[k] = int(int(v)/num_envs)
        elif k == "learning_rate":
            res[k] = float(v)
        elif k == "batch_size":
            res[k] = int(v)
        elif k == "gamma":
            res[k] = float(v)
    return res
