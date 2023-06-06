from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import fancy_gym
from pathlib import Path
import numpy as np
from task_space_env import TaskSpaceEnv
import pickle

here = Path(__file__).parent
STEPS = 7200000
LOAD_PATH = here / "models/PPO/SecondOrderInterpolation/gamma0.999,lr5.e-5,toleration0.2,change_rewards/06-05,20:33:43"
VEC_NORMALIZE_LOAD_PATH = LOAD_PATH / f"rl_model_vecnormalize_{STEPS}_steps.pkl"
MODEL_LOAD_PATH = LOAD_PATH / f"rl_model_{STEPS}_steps.zip"

def test():
    test_env = DummyVecEnv([lambda: fancy_gym.make("3dof-hit-eval-ee-vel", seed=0, toleration=0.2, horizon=1000)])
    normalizer = VecNormalize.load(VEC_NORMALIZE_LOAD_PATH, DummyVecEnv([lambda: TaskSpaceEnv("", 0)])) # Just to load a workaround
    normalizer.training = False
    model = PPO.load(MODEL_LOAD_PATH)
    with open("data/PPO/SecondOrderInterpolation/gamma0.999,lr5.e-5,toleration0.2,change_rewards/AaCLvw.pkl", "rb") as f:
        episodes = pickle.load(f)
    for episode_i in range(1000):
        i = 0
        episode = episodes[episode_i]
        obs = test_env.reset()
        model_action = [0,0]
        while True:
            test_env.render()
            model_action = [0, 0.1]
            model_obs = np.hstack([model_action, obs[0][6:12]])
            norm_model_obs = normalizer.normalize_obs(model_obs)
            env_action = model.predict(norm_model_obs)
            obs, rew, done, info = test_env.step(env_action)
            print(np.abs(np.array(model_action) - test_env.envs[0].unwrapped.env.base_env.get_ee()[1][[3,4]]))
            i += 1
            if (False and len(episode) == i) or i==done[0]:
                break

if __name__ == "__main__":
    test()