import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent

import sys
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
sys.path.append("rlpd")
from rlpd.agents import SACLearner
from absl import app, flags
from flax.training import checkpoints
import d4rl
import gym 
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
from rlpd.wrappers import wrap_gym
import time

kwargs = {'actor_lr': 0.0003, 'backup_entropy': True, 'critic_layer_norm': True, 'critic_lr': 0.0003, 'critic_weight_decay': None, 'discount': 0.99, 'hidden_dims': (256, 256), 'init_temperature': 1.0, 'model_cls': 'SACLearner', 'num_min_qs': 2, 'num_qs': 10, 'target_entropy': None, 'tau': 0.005, 'temp_lr': 0.0003}
model_cls = kwargs.pop("model_cls")

class RLPDAgent(AbstractAgent):
    save_time = False
    time_csv = "RLPD_time.csv"
    def load_agent(self, model_path, env_id):
        env = gym.make(env_id)
        env = wrap_gym(env, rescale_actions=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        
        seed = 0
        seed = 1 if "seed1" in model_path else seed
        seed = 10 if "seed10" in model_path else seed
        seed = 100 if "seed100" in model_path else seed 
        agent = globals()[model_cls].create(seed, env.observation_space, env.action_space, **kwargs)
        agent = checkpoints.restore_checkpoint(ckpt_dir=model_path, target=agent)
        print(model_path,seed)
        # agent = checkpoints.restore_checkpoint(ckpt_dir='/workspace/rlpd/log_test_seed1_2/s1_0pretrain_LN/checkpoints/checkpoint_0', target=agent)
        return agent

    def inference(self, state):
        policy_start = time.time()
        actions = self.agent.eval_actions(state)
        policy_end = time.time()
        
        inference_time = policy_end - policy_start
        if self.save_time:
            print(f"RLPD inference time: {inference_time} seconds")
            self._save_timing_to_csv(inference_time)
        
        if len(actions.shape) == 1:  # 如果是 (28,)
            actions = actions.reshape(1, -1)  # 重塑為 (1, 28)
        return actions
