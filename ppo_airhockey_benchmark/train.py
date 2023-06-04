import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import sync_envs_normalization

from .callbacks import CustomEvalCallback, SaveBestModel, CheckpointLog
from .constants import checkpoint_dir
from .util import make_environments, setup_wandb
import yaml

def start_training(train_dir, load):
    with open(train_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    setup_wandb(train_dir, config)
    env = config["training"]["env"]
    num_envs = config["training"]["num_envs"]
    env_args = config["training"]["env_args"]
    if load:
        with open(train_dir / checkpoint_dir / (load + ".json"), "r") as f:
            custom_log = json.load(f)
        train_env, eval_env = make_environments(env, num_envs, env_args, load=True, load_dir=train_dir / checkpoint_dir / (load + ".pkl"))
        model_load_dir = train_dir / checkpoint_dir / (load + ".zip")
        model = PPO.load(model_load_dir, train_env)
    else:
        os.makedirs(train_dir / checkpoint_dir)
        custom_log = {
            "best_mean_reward": -1e10,
        }
        train_env, eval_env = make_environments(env, num_envs, env_args, gamma=config["hyperparameters"]["gamma"])
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=train_dir, **config["hyperparameters"])
    sync_envs_normalization(train_env, eval_env)
    checkpoint_callback = CheckpointLog(save_dir=train_dir, custom_log=custom_log, save_freq=int(1e6/num_envs))
    save_best_model_callback = SaveBestModel(train_dir, custom_log)
    custom_eval_callback = CustomEvalCallback(eval_env, save_best_model_callback, 30, int(1e5/num_envs), custom_log["best_mean_reward"])
    callbacks = CallbackList([checkpoint_callback, custom_eval_callback])
    model.learn(total_timesteps=2e8, progress_bar=True, reset_num_timesteps=False, tb_log_name="run", callback=callbacks)

