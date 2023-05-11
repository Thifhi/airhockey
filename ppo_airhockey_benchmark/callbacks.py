import json

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

from .constants import (best_model_file_name, custom_log_file_name,
                        vecnormalize_file_name, checkpoint_dir)


class SaveBestModel(BaseCallback):
    def __init__(self, save_dir, custom_log):
        super().__init__(verbose=1)
        self.save_dir = save_dir
        self.custom_log = custom_log

    def _on_step(self) -> bool:
        self.model.save(self.save_dir / best_model_file_name)
        self.training_env.save(self.save_dir / vecnormalize_file_name)
        self.custom_log["best_mean_reward"] = float(self.parent.best_mean_reward)
        return True

class CheckpointLog(BaseCallback):
    def __init__(self, save_dir, custom_log, save_freq):
        super().__init__(verbose=1)
        self.save_dir = save_dir
        self.custom_log = custom_log
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls == 0 or self.n_calls % self.save_freq:
            return True
        self.custom_log["num_timesteps"] = self.model.num_timesteps
        with open(self.save_dir / custom_log_file_name, "w") as f:
            json.dump(self.custom_log, f)
        
        self.model.save(self.save_dir / checkpoint_dir / (str(self.model.num_timesteps) + ".zip"))
        self.training_env.save(self.save_dir / checkpoint_dir / (str(self.model.num_timesteps) + ".pkl"))
        with open(self.save_dir / checkpoint_dir / (str(self.model.num_timesteps) + ".json"), "w") as f:
            json.dump(self.custom_log, f)

        return True

class CustomEvalCallback(EventCallback):
    def __init__(self, eval_env, callback_on_new_best, n_eval_episodes, eval_freq, best_mean_reward = -np.inf):
        super().__init__(callback_on_new_best, verbose=1)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = best_mean_reward
    
    def _log_info_callback(self, locals, globals):
        self.all_infos.append(locals["info"])
        if locals["done"]:
            self.done_infos.append(locals["info"])

    def _on_step(self):
        if self.n_calls == 0 or self.n_calls % self.eval_freq:
            return True
        
        self.all_infos = []
        self.done_infos = []
        sync_envs_normalization(self.training_env, self.eval_env)
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True,
            warn=True,
            callback=self._log_info_callback,
        )
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        self.logger.record("eval/reward", float(mean_reward))
        self.logger.record("eval/episode_length", mean_ep_length)

        mean_success = np.mean([i["has_scored"] for i in self.done_infos])
        self.logger.record("eval/success", mean_success)

        mean_hit_rate = np.mean([i["has_hit"] for i in self.done_infos])
        self.logger.record("eval/hit_rate", mean_hit_rate)

        mean_min_dist_ee_puck = np.mean([i["min_dist_ee_puck"] for i in self.done_infos])
        self.logger.record("eval/min_dist_ee_puck", mean_min_dist_ee_puck)

        mean_min_dist_puck_goal = np.mean([i["min_dist_puck_goal"] for i in self.done_infos])
        self.logger.record("eval/min_dist_puck_goal", mean_min_dist_puck_goal)

        mean_puck_velocity = np.mean([i["mean_puck_vel_after_hit"] for i in self.done_infos])
        self.logger.record("eval/puck_velocity", mean_puck_velocity)

        lower_joint_pos = [i["constraints_value"]["joint_pos_constr"][:3] for i in self.all_infos]
        upper_joint_pos = [i["constraints_value"]["joint_pos_constr"][3:] for i in self.all_infos]
        max_joint_pos_constr = np.max(lower_joint_pos + upper_joint_pos, axis=1)
        self.logger.record("constraint/joint_pos", np.max(np.max(max_joint_pos_constr), 0))

        lower_joint_vel = [i["constraints_value"]["joint_vel_constr"][:3] for i in self.all_infos]
        upper_joint_vel = [i["constraints_value"]["joint_vel_constr"][3:] for i in self.all_infos]
        max_joint_vel_constr = np.max(lower_joint_vel + upper_joint_vel, axis=1)
        self.logger.record("constraint/joint_vel", np.max(np.max(max_joint_vel_constr), 0))

        max_joint_jerk_constr = np.max([i["jerk"] for i in self.all_infos], axis=1)
        self.logger.record("constraint/joint_jerk", np.max(np.max(max_joint_jerk_constr), 0))

        ee_constr_x = [i["constraints_value"]["ee_constr"][0] for i in self.all_infos]
        self.logger.record("constraint/ee_x", np.max(np.max(ee_constr_x), 0))

        lower_ee_x = [i["constraints_value"]["ee_constr"][1] for i in self.all_infos]
        upper_ee_x = [i["constraints_value"]["ee_constr"][2] for i in self.all_infos]
        self.logger.record("constraint/ee_y", np.max(np.max(lower_ee_x + upper_ee_x), 0))

        compute_times = [i["compute_time_ms"] for i in self.all_infos]
        self.logger.record("constraint/compute_time", np.max(compute_times))

        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            print("New best mean reward!")
            self.best_mean_reward = mean_reward
            self.callback.on_step()

        return True