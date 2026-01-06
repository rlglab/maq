import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
import sys
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
from stable_baselines3 import SAC

class RndAgent(AbstractAgent):
    def load_agent(self, model_path, env_id):
        env = gym.make(env_id)
        return env.action_space


    def inference(self, state):
        actions = self.agent.sample()
        if len(actions.shape) == 1:  # 如果是 (28,)
            actions = actions.reshape(1, -1)  # 重塑為 (1, 28)
        return actions
