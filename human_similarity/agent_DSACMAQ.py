import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent
import sys
sys.path.append("RLPD_MAQ")
sys.path.append("RLPD_MAQ/models")
sys.path.append("VQVAE")
from VQVAE.modules import VectorQuantizedVAE
from VQVAE.prior_train import PriorNet
from VQVAE.vqvae_train import save, load
from VQVAE.dataset import load_data, split_data, D4RLSequenceDataset
from RLPD_MAQ.DSACMAQ import *
import time

# MAQDSAC
class DSACMAQAgent(AbstractAgent):
    save_time = False
    time_csv = "DSACMAQ_time.csv"
    

    def load_agent(self, model_path, env_id):
        model_path = model_path.rsplit('/', 1)
        config = json.load(open(os.path.join(model_path[0], 'config.json')))
        config["env_id"] = env_id
        config["logdir"] = 'log/dummy'

        config['vqvae_model_path'] = config['vqvae_model_path'].replace('../', '')  # adjust the path
        agent = MAQDSACAgent(config)
        path = os.path.join(model_path[0], model_path[1])
        agent.load(path)
        return agent


    def inference(self, state):
        state = np.expand_dims(state, axis=0)  # add batch dimension
        policy_start = time.time()
        actions = self.agent.decide_agent_actions(state, eval=True)
        policy_end = time.time()
        vqvae_start = time.time()
        decoded_actions = self.agent.decoded_primitive_actions(state, actions)
        vqvae_end = time.time()
        
        policy_time = policy_end - policy_start
        vqvae_time = vqvae_end - vqvae_start
        total_time = policy_time + vqvae_time
        
        if self.save_time:
            print(f"DSACMAQ decide_agent_actions time: {policy_time} seconds")
            print(f"DSACMAQ decoded_primitive_actions time: {vqvae_time} seconds")
            self._save_detailed_timing_to_csv(policy_time, vqvae_time, total_time)
        return decoded_actions[0]