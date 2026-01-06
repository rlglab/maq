import os
# import gymnasium as gym
import gym
import d4rl
import csv
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import torch

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gym


class SaveOnStepsCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, env_id: str, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "checkpoints")
        self.eval_env = gym.make(env_id)
    
    def _eval(self, n_episodes=100):
        avg_reward = np.zeros(n_episodes)
        for i in range(n_episodes):
            obs = self.eval_env.reset()
            total_reward = 0
            while True:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                if done:
                    break

            # print(f"Episode {i+1} total reward: {total_reward}")
            avg_reward[i] = total_reward
        avg_reward = np.mean(avg_reward)
        return avg_reward

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                eval_score = self._eval(n_episodes=10)
                normalized_eval_score = self.eval_env.get_normalized_score(eval_score) * 100
                print(f"Num timesteps: {self.num_timesteps}; Eval score: {eval_score}; Normalized eval score: {normalized_eval_score}")
                with open(os.path.join(self.log_dir, "SAC.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.num_timesteps, eval_score, normalized_eval_score])

                print(f"Saving new model to {self.save_path}/SAC_model_{self.num_timesteps}_{int(eval_score)}.zip")
                self.model.save(os.path.join(self.save_path, f"SAC_model_{self.num_timesteps}_{int(eval_score)}"))

        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training SAC.')
    parser.add_argument('--env', type=str, help='')
    parser.add_argument('--seed', type=int, help='')
    args = parser.parse_args()
    env = args.env

    env_id = f"{env}"
    log_dir = f"log/{env_id}/SAC_seed{args.seed}"
    os.makedirs(log_dir, exist_ok=True)
    
    num_cpu = 8
    env = make_vec_env(env_id, n_envs=num_cpu)
    
    callback = SaveOnStepsCallback(env_id=env_id, check_freq=1000, log_dir=log_dir)
    model = SAC("MlpPolicy", env, tensorboard_log=log_dir, verbose=0)
    # model = SAC("CnnPolicy", env, tensorboard_log=log_dir, verbose=1)
    model.learn(total_timesteps=int(1e6), log_interval=10, callback=callback)

    