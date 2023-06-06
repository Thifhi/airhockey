import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from task_space_env import TaskSpaceEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import sync_envs_normalization, VecMonitor, SubprocVecEnv, VecNormalize, DummyVecEnv
from pathlib import Path
import datetime

here = Path(__file__).parent
LEARN_FROM = Path("PPO/SecondOrderInterpolation/gamma0.999,lr5.e-5,toleration0.2,change_rewards")
learn_from_dir = here / "data" / LEARN_FROM
date_time_now = datetime.datetime.now().strftime("%m-%d,%H:%M:%S")
MODEL_DIR = here / "models" / LEARN_FROM / date_time_now
NUM_CPU = 12

def start_training():
    train_env = VecNormalize(VecMonitor(SubprocVecEnv([lambda: TaskSpaceEnv(learn_from_dir.absolute(), toleration=0.2) for _ in range(NUM_CPU)])), gamma=1)
    eval_env = VecNormalize(VecMonitor(DummyVecEnv([lambda: TaskSpaceEnv(learn_from_dir.absolute(), toleration=0.2)])), training=False, norm_reward=False)
    eval_callback = EvalCallback(eval_env, eval_freq=1e4)
    checkpoint_callback = CheckpointCallback(1e5, MODEL_DIR, save_vecnormalize=True)
    cb_list = CallbackList([eval_callback, checkpoint_callback])
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=MODEL_DIR, gamma=1, batch_size=256, n_steps=100, learning_rate=5e-5)
    model.learn(total_timesteps=2e8, progress_bar=True, reset_num_timesteps=False, tb_log_name="run", callback=cb_list)

if __name__ == "__main__":
    start_training()
