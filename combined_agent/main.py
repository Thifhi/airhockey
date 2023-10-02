from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent
from my_agent import MyAgent
import numpy as np

class OpponentAgent:
    def __init__(self, env_info):
        self.baseline_agent = BaselineAgent(env_info=env_info, agent_id=2)

    def draw_action(self, obs):
        return self.baseline_agent.draw_action(obs) 

def main():
    env = AirHockeyChallengeWrapper(env="tournament")
    opponent_agent = OpponentAgent(env.env_info)
    # opponent_agent = MyAgent(env.env_info)
    my_agent = MyAgent(env.env_info)
    for i in range(100):
        obs = env.reset()
        while True:
            env.render()
            obs1, obs2 = np.split(obs, 2)
            act1 = my_agent.draw_action(obs1)
            act2 = opponent_agent.draw_action(obs2)
            action = np.array([act1, act2])
            obs, rew, done, info = env.step(action)
            if done:
                break 


if __name__ == "__main__":
    main()