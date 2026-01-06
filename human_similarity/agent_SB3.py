import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent
import gym 
import sys
import time
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
from stable_baselines3 import SAC

class SB3Agent(AbstractAgent):
    save_time = False
    time_csv = "SAC_time.csv"
    
    def load_agent(self, model_path, env_id):
        env = gym.make(env_id)
        agent = SAC("MlpPolicy", env, verbose=0)
        agent = SAC.load(model_path)
        return agent

    def inference(self, state):
        policy_start = time.time()
        actions, _states = self.agent.predict(state, deterministic=True)
        policy_end = time.time()
        
        inference_time = policy_end - policy_start
        if self.save_time:
            print(f"SB3 inference time: {inference_time} seconds")
            self._save_timing_to_csv(inference_time)
        return actions
