from baseline.baseline_agent.baseline_agent import BaselineAgent
from stable_baselines3 import PPO
from air_hockey_challenge.utils import forward_kinematics, jacobian
import pickle
import numpy as np

IK_TARGET = [-0.86, 0]

class MyIK:
    def __init__(self, env_info):
        self.env_info = env_info
        with open("models/ik.zip", "rb") as f:
            self.model = PPO.load(f)
        with open("models/ik.pkl", "rb") as f:
            self.normalizer = pickle.load(f)
    
    def draw_action(self, obs):
        if self.first_stop:
            if np.max(np.abs(obs[self.env_info["joint_vel_ids"]])) < 0.03:
                self.first_stop = False
                self.interp_pos = obs[self.env_info["joint_pos_ids"]]
                self.interp_vel = obs[self.env_info["joint_vel_ids"]]
            else:
                self.interp_vel /= 2
                self.interp_pos += self.interp_vel * 0.02
                return np.vstack([self.interp_pos, self.interp_vel])

        target_q = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
        diff_q = obs[self.env_info["joint_pos_ids"]] - target_q
        if np.max(np.abs(obs[self.env_info["joint_vel_ids"]])) < 0.03 and np.max(np.abs(diff_q)) < 0.03:
            self.at_init = True
        if self.at_init or self.roll_to_init or (np.max(np.abs(diff_q)) < 0.2 and np.max(np.abs(obs[self.env_info["joint_vel_ids"]])) < 0.2):
            self.roll_to_init = True
            if np.max(np.abs(diff_q)) < 0.01:
                return np.vstack([target_q, [0 for i in range(7)]])
            else:
                use = (1. * target_q + 1. * obs[self.env_info["joint_pos_ids"]]) / 2
                return np.vstack([use, [0 for i in range(7)]])

        planned_world_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], self.interp_pos)[0]
        obs = np.hstack([obs[self.env_info["joint_pos_ids"]], obs[self.env_info["joint_vel_ids"]]])
        obs = np.hstack([obs, self.interp_pos, self.interp_vel, self.last_acceleration, planned_world_pos])

        norm_obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.hstack([action, 0])
        action /= 10
        new_vel = self.interp_vel + action
        jerk = 2 * (new_vel - self.interp_vel - self.last_acceleration * 0.02) / (0.02 ** 2)
        new_pos = self.interp_pos + self.interp_vel * 0.02 + (1/2) * self.last_acceleration * (0.02 ** 2) + (1/6) * jerk * (0.02 ** 3)

        self.interp_pos = new_pos
        self.interp_vel = new_vel
        self.last_acceleration += jerk * 0.02
        return np.vstack([new_pos, new_vel])

    def reset(self, obs):
        self.first_stop = True
        self.at_init = False
        self.roll_to_init = False
        self.last_acceleration = np.repeat(0., 7)
        self.interp_pos = obs[self.env_info["joint_pos_ids"]]
        self.interp_vel = obs[self.env_info["joint_vel_ids"]]


class MyHitAgent:
    def __init__(self, env_info):
        self.env_info = env_info
        with open("models/ppo_hit.zip", "rb") as f:
            self.model = PPO.load(f)
        with open("models/ppo_hit.pkl", "rb") as f:
            self.normalizer = pickle.load(f)
    
    def draw_action(self, obs):
        planned_world_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], self.interp_pos)[0]
        obs = np.hstack([obs[:-3], self.interp_pos, self.interp_vel, self.last_acceleration, planned_world_pos])

        norm_obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.hstack([action, 0])
        action /= 10
        new_vel = self.interp_vel + action
        jerk = 2 * (new_vel - self.interp_vel - self.last_acceleration * 0.02) / (0.02 ** 2)
        new_pos = self.interp_pos + self.interp_vel * 0.02 + (1/2) * self.last_acceleration * (0.02 ** 2) + (1/6) * jerk * (0.02 ** 3)

        self.interp_pos = new_pos
        self.interp_vel = new_vel
        self.last_acceleration += jerk * 0.02
        return np.vstack([new_pos, new_vel])

    def reset(self, obs):
        self.last_acceleration = np.repeat(0., 7)
        self.interp_pos = obs[self.env_info["joint_pos_ids"]]
        self.interp_vel = obs[self.env_info["joint_vel_ids"]]

class MyDefendAgent:
    def __init__(self, env_info):
        self.env_info = env_info
        with open("models/ppo_defend.zip", "rb") as f:
            self.model = PPO.load(f)
        with open("models/ppo_defend.pkl", "rb") as f:
            self.normalizer = pickle.load(f)
    
    def draw_action(self, obs):
        planned_world_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], self.interp_pos)[0]
        obs = np.hstack([obs[:-3], self.interp_pos, self.interp_vel, self.last_acceleration, planned_world_pos])

        norm_obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.hstack([action, 0])
        action /= 10
        new_vel = self.interp_vel + action
        jerk = 2 * (new_vel - self.interp_vel - self.last_acceleration * 0.02) / (0.02 ** 2)
        new_pos = self.interp_pos + self.interp_vel * 0.02 + (1/2) * self.last_acceleration * (0.02 ** 2) + (1/6) * jerk * (0.02 ** 3)

        self.interp_pos = new_pos
        self.interp_vel = new_vel
        self.last_acceleration += jerk * 0.02
        return np.vstack([new_pos, new_vel])

    def reset(self, obs):
        self.last_acceleration = np.repeat(0., 7)
        self.interp_pos = obs[self.env_info["joint_pos_ids"]]
        self.interp_vel = obs[self.env_info["joint_vel_ids"]]

class MyAgent:
    def __init__(self, env_info):
        self.env_info = env_info
        self.ik = MyIK(env_info)
        self.sm = StateMachine(env_info, self.ik)

        self.hit_agent = MyHitAgent(env_info=env_info)
        self.defend_agent = MyDefendAgent(env_info=env_info)
        self.prepare_agent = BaselineAgent(env_info=env_info, only_tactic="prepare", agent_id=2)
        self.reset()
    
    def reset(self):
        self.step = 0

    def draw_action(self, obs):
        if self.step == 0:
            self.sm.reset(obs)
            if self.sm.state == "hit":
                self.hit_agent.reset(obs)
            elif self.sm.state == "defend":
                self.defend_agent.reset(obs)
        
        if self.sm.update_state(obs):
            if self.sm.state == "hit":
                self.hit_agent.reset(obs)
            elif self.sm.state == "defend": 
                self.defend_agent.reset(obs)
            elif self.sm.state == "prepare": 
                self.prepare_agent.reset(obs)
            elif self.sm.state == "ik": 
                self.ik.reset(obs)
        
        if self.sm.state == "hit":
            action = self.hit_agent.draw_action(obs)
        elif self.sm.state == "defend": 
            action = self.defend_agent.draw_action(obs)
        elif self.sm.state == "prepare": 
            action = self.prepare_agent.draw_action(obs)
        elif self.sm.state == "ik": 
            action = self.ik.draw_action(obs)

        self.step += 1
        return action

class StateMachine:
    def __init__(self, env_info, ik: MyIK):
        self.ik = ik
        self.env_info = env_info
    
    def reset(self, obs):
        puck_pos = obs[self.env_info["puck_pos_ids"]]
        if puck_pos[0] - 1.51 < 0:
            self.state = "hit"
        else:
            self.state = "defend"
    
    def update_state(self, obs):
        # print(self.state)
        if self.state == "hit":
            puck_pos = obs[self.env_info["puck_pos_ids"]]
            if puck_pos[0] - 1.51 >= 0:
                self.state = "ik"
                return True
        elif self.state == "defend":
            puck_vel = obs[self.env_info["puck_vel_ids"]]
            if puck_vel[0] > -0.2:
                self.state = "ik"
                return True
        elif self.state == "prepare":
            puck_pos = obs[self.env_info["puck_pos_ids"]]
            puck_vel = obs[self.env_info["puck_vel_ids"]]
            if (puck_pos[1] > 0 and puck_vel[1] < 0) or (puck_pos[1] < 0 and puck_vel[1] > 0):
                self.state = "ik"
                return True
        elif self.state == "ik":
            if self.ik.at_init:
                puck_pos = obs[self.env_info["puck_pos_ids"]]
                puck_vel = obs[self.env_info["puck_vel_ids"]]
                if puck_pos[0] - 1.51 < 0.4 and puck_vel[0] < -0.2:
                    self.state = "defend"
                    return True 
                elif puck_pos[0] - 1.51 < -0.2 and puck_vel[0] < 0.25:
                    self.state = "hit"
                    return True 
        return False