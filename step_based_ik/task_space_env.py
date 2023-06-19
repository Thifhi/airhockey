import os
import copy
import random
import numpy as np
from typing import Union, Tuple, Optional

import fancy_gym
import gym
from gym import spaces, utils
from gym.core import ObsType, ActType
import pickle
from pathlib import Path

from air_hockey_challenge.utils import robot_to_world
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.environments.planar import AirHockeyHit, AirHockeyDefend

class TaskSpaceEnv(gym.Env):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = Path(data_path)
        self.episodes = None
        self.current_episode = None
        # base environment
        self.env = fancy_gym.make("7dof-joint-acc-ee-vel", seed=0) # Nothing random since we set the puck

        # observation space
        obs_low = np.hstack([-5., -5., self.env.observation_space.low[6:20], self.env.observation_space.low[-21:]])
        obs_high = np.hstack([5., 5., self.env.observation_space.high[6:20], self.env.observation_space.high[-21:]])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # action space
        self.action_space = spaces.Box(low=np.repeat(-100, 7), high=np.repeat(100, 7), dtype=np.float32)

    def get_next_episode(self):
        self.current_episode = []
        while not self.episodes:
            next_episodes_file = random.choice(os.listdir(self.data_path))
            with open(self.data_path / next_episodes_file, "rb") as f:
                self.episodes = pickle.load(f)
        while len(self.current_episode) < 2:
            self.current_episode = self.episodes.pop()

    def reset(self):
        env_obs = self.env.reset()
        self.get_next_episode()
        obs = np.hstack([self.current_episode[0], env_obs[6:20], env_obs[-21:]])
        return obs

    def step(self, action):
        env_obs, env_rew, env_done, env_info = self.env.step(action)

        supposed_obs = self.current_episode[0]
        rew, fatal = self.ik_reward(supposed_obs)

        self.current_episode.pop(0)

        obs = np.hstack([self.current_episode[0], env_obs[6:20], env_obs[-21:]])

        if env_done:
            rew = env_rew
        if env_done or fatal or len(self.current_episode) == 1:
            done = True
            self.env.reset()
        else:
            done = False
        info = {}

        return obs, rew, done, info
    
    def ik_reward(self, supposed_obs):
        ee_pos, ee_vel = self.env.unwrapped.env.base_env.get_ee()
        actual_ee_vel = ee_vel[[3,4]]
        max_abs_diff = np.max(np.abs(actual_ee_vel - np.array(supposed_obs)))
        rew = np.exp((0.1 - max_abs_diff) * 30)
        if max_abs_diff > 0.1:
            fatal = True
            rew = 0
        else:
            fatal = False
        return rew, fatal
    
    def render(self, mode):
        self.env.render()