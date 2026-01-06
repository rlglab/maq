import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent
import sys
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
from IQL.iql_MAQ import load_IQL_agent, load_IQL_and_decide_actions, load_IQL_and_decide_actions_time
import time

class MAQIQLAgent(AbstractAgent):
    save_time = False
    time_csv = "MAQIQL_time.csv"
    def load_agent(self, model_path, env_id):
        seed = self.get_seed(model_path)
        print(model_path)
        default_k = 16
        default_seqlen = 3
        if "_k" in model_path:
            k = model_path.split("_k")[1].split("_")[0]
        else:
            k = default_k
        if "_sq" in model_path:
            seqlen = model_path.split("_sq")[1].split("_")[0]
        else:
            seqlen = default_seqlen
        print(k,seqlen)
        if k == "":
            k = default_k
        if seqlen == "":
            seqlen = default_seqlen
        agent, _, self.state_mean, self.state_std, self.vqvae_model, self.prior_model = load_IQL_agent(model_path=model_path, env_id=env_id, seed=seed, k=k, seqlen=seqlen)
        return agent
        # return None, None, None, None, None, None
    def inference(self, state):
        policy_start = time.time()
        
        if self.save_time:
            actions, policy_time, vqvae_time = load_IQL_and_decide_actions_time(self.agent, state, self.state_mean, self.state_std, self.vqvae_model, self.prior_model)
            inference_time = policy_time + vqvae_time
            print(f"MAQIQL decide_agent_actions time: {policy_time} seconds")
            print(f"MAQIQL decoded_primitive_actions time: {vqvae_time} seconds")
            self._save_detailed_timing_to_csv(policy_time, vqvae_time, inference_time)
        else:
            actions = load_IQL_and_decide_actions(self.agent, state, self.state_mean, self.state_std, self.vqvae_model, self.prior_model)

        # print(actions.shape) 
        return actions[0]

    def get_seed(self,model_path):
        if "seed100" in model_path:
            return 100
        if "seed10" in model_path:
            return 10
        if "seed1" in model_path:
            return 1
        if "VQVAE" in model_path:
            if "seed100" in model_path['VQVAE']:
                return 100
            if "seed10" in model_path['VQVAE']:
                return 10
            if "seed1" in model_path['VQVAE']:
                return 1