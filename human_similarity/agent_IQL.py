import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent
import sys
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
from IQL.iql import load_IQL_agent, load_IQL_and_decide_actions
import time

class IQLAgent(AbstractAgent):
    save_time = False
    time_csv = "IQL_time.csv"
    def load_agent(self, model_path, env_id):
        agent, _, self.state_mean, self.state_std = load_IQL_agent(model_path=model_path, env_id=env_id)
        return agent

    def inference(self, state):
        policy_start = time.time()
        actions = load_IQL_and_decide_actions(self.agent, state, self.state_mean, self.state_std)
        policy_end = time.time()
        
        inference_time = policy_end - policy_start
        if self.save_time:
            print(f"IQL inference time: {inference_time} seconds")
            self._save_timing_to_csv(inference_time)
        if len(actions.shape) == 1:  # 如果是 (28,)
            actions = actions.reshape(1, -1)  # 重塑為 (1, 28)
        return actions
