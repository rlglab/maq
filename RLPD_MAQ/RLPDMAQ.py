import torch
import torch.nn as nn
import numpy as np
import os
import random
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("..")
sys.path.append("../VQVAE")
from RLPD_MAQ.replay_buffer.replay_buffer import ReplayMemory, D4RLGameDataset
from RLPD_MAQ.base_agent import RLPDDiscreteSACAgent, Actor, Critic
from RLPD_MAQ.models.networks import PPONetSimple
from VQVAE.modules import VectorQuantizedVAE
# from VQVAE.prior_train import PriorNet
from VQVAE.vqvae_train import save, load
from VQVAE.dataset import load_data
import pickle

import json
import csv
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import d4rl
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  



from env_wrapper.macro_action_wrapper import MacroActionEnvWrapper

class MAQDRLPDSACAgent(RLPDDiscreteSACAgent):
	def __init__(self, config):
		super(MAQDRLPDSACAgent, self).__init__(config)
		self.config = config
		# load the VQVAE model
		observations, actions = load_data(self.config["env_id"],self.config["seed"])
		vqvae_config = json.load(open(os.path.join(config['vqvae_model_path'], 'config.json')))
		self.vqvae_config = vqvae_config
		self.vqvae_model = VectorQuantizedVAE(state_dim=observations.shape[1], seq_len=vqvae_config['sequence_length'], K=vqvae_config['k'], dim=vqvae_config['hidden_size'], output_dim=actions.shape[1]).to(self.device)
		
		
		# loading the last 
		ls = [filename for filename in os.listdir(config['vqvae_model_path']) if filename.endswith(".pth") ]
		


		# Sort checkpoints by epoch number
		sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[1]))

		# Get the last epoch filename
		last_vqvae_cp = sorted_checkpoints[-1]

		print("loaded vqvae model",last_vqvae_cp)
		
		load(self.vqvae_model, os.path.join(config['vqvae_model_path'], last_vqvae_cp ))
		self.vqvae_model.eval()
		print("VQVAE model loaded.")
		# load the prior model

		# loading the last 
		# ls = [filename for filename in os.listdir(config['prior_model_path']) if filename.endswith(".pth") ]
		


		# # Sort checkpoints by epoch number
		# sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[2]))

		# # Get the last epoch filename
		# last_prior_cp = sorted_checkpoints[-1]

		# print("loaded prior model",last_prior_cp)




		# self.prior_model = PriorNet(state_dim=observations.shape[1], hidden_dim=vqvae_config['hidden_size'], output_K=vqvae_config['k']).to(self.device)
		# load(self.prior_model, os.path.join(config['prior_model_path'],last_prior_cp))
		# self.prior_model.eval()
		# print("Prior model loaded.")

		self.seed = config["seed"] if "seed" in config else int(time.time())
		self.N_action = vqvae_config['k']
		print("num macro action: ", self.N_action)
		# create the environment
		self.env = gym.vector.AsyncVectorEnv(
			[
				lambda: MacroActionEnvWrapper(self.config["env_id"], self.N_action, seed=self.config['seed']+i) for i in range(self.num_envs) 
			] 
		)
		self.test_env = gym.vector.AsyncVectorEnv(
			[
				lambda: MacroActionEnvWrapper(self.config["env_id"], self.N_action,seed=self.config['seed']+i) for i in range(self.eval_episode)
			]
		)
		self.dummy_env = gym.make(self.config["env_id"])
		self.dummy_env.seed(self.config["seed"])
		self.dummy_env.action_space.seed(self.config["seed"])
		self.dummy_env.observation_space.seed(self.config["seed"])
		self.training_dataset = config["training_dataset"]
		self.testing_dataset = config["testing_dataset"]


		# Initialize SAC networks using the helper from the base class
		state_dim = observations.shape[1]
		# Use learning rates from config or default
		actor_lr = config.get("actor_lr", 3e-4)
		critic_lr = config.get("critic_lr", 1e-3)
		self._initialize_networks(Actor, Critic, state_dim, self.N_action,
								  hidden_dim=config.get("hidden_dim", 256),
								  actor_lr=actor_lr,
								  critic_lr=critic_lr)

		# Removed prior weights loading for actor (not needed for RLPD)
		# self.symmetric_sampling = True
		if self.symmetric_sampling:
			print("safe????????????????????")
		self.set_seed(self.seed)
		if "store_init" in config and config["store_init"]:
			self.save(os.path.join(self.writer.log_dir, f"model_0_0.pth"))
		
		try:
			self.train_trajectories = self.load_trajectories(f"../offline_data/{self.training_dataset}")
			print(f"loaded {len(self.train_trajectories)} trajectories")
			
			self.demonstrations = ReplayMemory(100000)  # size for the trajecotries
			train_dataset = D4RLGameDataset(self.train_trajectories, self.vqvae_config["sequence_length"])
			self.demonstrations.load_from_dataset(train_dataset)
			
		except:
			print("disable symmetric sampling")
			print(f"Error: Trajectory file not found at ../offline_data/{self.training_dataset}")
			# self.symmetric_sampling = False


	def load_trajectories(self, file):
		"""從檔案加載 trajectory 數據"""
		try:
			with open(file, 'rb') as f:
				return pickle.load(f)
		except FileNotFoundError:
			print(f"Error: Trajectory file not found at {file}")
			# Depending on requirements, you might want to exit or return empty list
			# sys.exit(1)
			return [] # Return empty list if file not found


	def set_seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed)
			random.seed(seed)
			torch.manual_seed(seed)

	# Implement abstract methods from base class
	def _preprocess_state(self, observation):
		# Convert numpy observation to torch tensor on the correct device
		return torch.from_numpy(observation).to(self.device, dtype=torch.float32)

	def _postprocess_action(self, action_tensor):
		# Convert torch tensor action back to numpy array for the environment
		return action_tensor.cpu().detach().numpy()

	def decoded_primitive_actions(self, states, K_index):
		# Convert K_index to tensor if it's numpy
		if isinstance(K_index, np.ndarray):
			K_index = torch.from_numpy(K_index).to(self.device, dtype=torch.long) # K_index should be long
		# Convert states to tensor if it's numpy
		if isinstance(states, np.ndarray):
			states = torch.from_numpy(states).to(self.device, dtype=torch.float32)
		
		with torch.no_grad():
			decoded_actions = self.vqvae_model.forward_decoder(states, K_index)
			if isinstance(decoded_actions, torch.Tensor):
				# Only convert to numpy if it's a tensor
				decoded_actions = decoded_actions.cpu().detach().numpy()
			return decoded_actions
	
	# def prior_referenced_logits(self, states):
	# 	# Ensure input is a tensor
	# 	if isinstance(states, np.ndarray):
	# 		states = torch.from_numpy(states).to(self.device, dtype=torch.float32)
	# 	pred = self.prior_model(states)
	# 	return pred
		
	# Let's refine action selection to optionally return logits for evaluation
	def decide_agent_actions_with_logits(self, observation, eval=True):
		"""Get actions and logits, primarily for evaluation analysis."""
		self.actor.eval()
		with torch.no_grad():
			state_tensor = self._preprocess_state(observation)
			logits, _ = self.actor(state_tensor)
			dist = Categorical(logits=logits)
			if eval and self._deterministic_eval:
				action_tensor = logits.argmax(dim=-1)
			else:
				action_tensor = dist.sample()
		action = self._postprocess_action(action_tensor)
		self.actor.train()
		return action, logits.cpu() # Return action (numpy) and logits (tensor)
	def symmetric_sampling(self):
		"""Symmetric sampling from the online replay buffer and the offline demonstration buffer.
		Converts offline primitive actions to macro-action indices using VQ-VAE.
		Returns standard (s, a, r, s', d) tuples where 'a' is the macro-action index.
		"""
		# if self.symmetric_sampling:
		online_batch_size = self.batch_size // 2
		offline_batch_size = self.batch_size - online_batch_size
  
			
		# Sample from online buffer (contains macro actions)
		states_online, actions_online, rewards_online, next_states_online, dones_online = self.replay_buffer.sample(online_batch_size, self.device)
		
		# Sample from offline buffer (contains primitive actions)
		# Sample returns 5 elements: (state, primitive_action_seq, reward, next_state, done)
		states_offline, actions_offline_raw, rewards_offline, next_states_offline, dones_offline = self.demonstrations.sample(offline_batch_size, self.device)
		# Check dimensions and fix shape issues for VQVAE processing
		# VQVAE expects states and actions to have the same number of dimensions
		actions_offline_raw = actions_offline_raw.view(actions_offline_raw.size(0), -1)
		
		# convert raw action to macro actions using VQ-VAE
		with torch.no_grad(): # Ensure no gradients are computed for VQVAE forward pass
			# Process actions through the VQVAE to get the macro action indices
			_, _, _, x_tilde = self.vqvae_model(states_offline, actions_offline_raw)
			 
		# x_tilde is the macro action index tensor (likely LongTensor)
		# Convert x_tilde to float to match the dtype from buffer.sample()
		actions_offline = x_tilde.detach().float()
		
		# Ensure actions_offline has the same number of dimensions as actions_online
		# Typically, .sample() returns shape [batch_size, 1] for actions
		if actions_offline.dim() == 1:
			actions_offline = actions_offline.unsqueeze(1)
		if actions_online.dim() != actions_offline.dim():
			# Handle potential mismatch if actions_online is not [batch_size, 1]
			print(f"Warning: Dimension mismatch between online ({actions_online.shape}) and offline ({actions_offline.shape}) actions.")
			# Attempt to reshape based on expected action dim (usually 1 for discrete index)
			if actions_online.shape[1] == 1 and actions_offline.dim() > 1:
				actions_offline = actions_offline.view(-1, 1)

		states = torch.cat((states_online, states_offline), dim=0)
		# Concatenate tensors: actions_online (float) and actions_offline (float)
		actions = torch.cat((actions_online, actions_offline), dim=0) 
		rewards = torch.cat((rewards_online, rewards_offline), dim=0)
		next_states = torch.cat((next_states_online, next_states_offline), dim=0)
		dones = torch.cat((dones_online, dones_offline), dim=0)
		
		return states, actions, rewards, next_states, dones
	def update(self):
		"""Update actor and critic networks using SAC logic."""
		if len(self.replay_buffer) < self.batch_size:
			return {} # Not enough samples yet

		self.actor.train()
		self.critic1.train()
		self.critic2.train()

		losses = {}

		for _ in range(self.update_freq): # Perform multiple updates per call if update_freq > 1
			
			state, action, reward, next_state, done = self.symmetric_sampling()

			action = action.type(torch.long) # Discrete actions are indices

			# --- Critic Update ---
			with torch.no_grad():
				# Get next action distribution from actor
				next_logits, _ = self.actor(next_state)
				next_dist = Categorical(logits=next_logits)
				next_action_probs = next_dist.probs

				# Get target Q values from target critics
				q_next1 = self.critic1_old(next_state)
				q_next2 = self.critic2_old(next_state)
				min_q_next = torch.min(q_next1, q_next2)

				# Calculate target Q: E_{a'~pi}[Q_target(s', a')] - alpha * H(pi(.|s'))
				# Original paper uses V_target(s') = E_{a'~pi}[Q_target(s', a') - alpha * log pi(a'|s')]
				# For discrete actions, this is sum_a'[pi(a'|s') * (Q_target(s', a') - alpha * log pi(a'|s'))]
				target_q_values = (next_action_probs * (min_q_next - self._alpha * torch.log(next_action_probs + 1e-8))).sum(dim=1, keepdim=True)

				# Bellman backup: r + gamma * V_target(s') * (1 - done)
				q_target = reward + self.gamma * target_q_values * (1 - done)

			# Get current Q estimates
			current_q1 = self.critic1(state).gather(1, action)
			current_q2 = self.critic2(state).gather(1, action)

			# Calculate critic losses
			critic1_loss = F.mse_loss(current_q1, q_target)
			critic2_loss = F.mse_loss(current_q2, q_target)

			# Optimize critics
			self.critic1_optim.zero_grad()
			critic1_loss.backward()
			self.critic1_optim.step()

			self.critic2_optim.zero_grad()
			critic2_loss.backward()
			self.critic2_optim.step()

			losses.update({
				'loss/critic1': critic1_loss.item(),
				'loss/critic2': critic2_loss.item(),
			})

			# --- Actor and Alpha Update ---
			# Freeze critic gradients
			for p in self.critic1.parameters(): p.requires_grad = False
			for p in self.critic2.parameters(): p.requires_grad = False

			# Calculate actor loss: E_{s~D, a~pi}[alpha * log pi(a|s) - Q(s,a)]
			logits, _ = self.actor(state)
			dist = Categorical(logits=logits)
			action_probs = dist.probs
			log_probs = torch.log(action_probs + 1e-8)
			entropy = - (action_probs * log_probs).sum(dim=1, keepdim=True)

			with torch.no_grad():
				q1_actor = self.critic1(state)
				q2_actor = self.critic2(state)
				min_q_actor = torch.min(q1_actor, q2_actor)

			# Actor objective: maximize E[Q - alpha * log pi] = minimize E[alpha * log pi - Q]
			# For discrete: minimize sum_a[pi(a|s) * (alpha * log pi(a|s) - Q(s,a))]
			actor_loss = (action_probs * (self._alpha * log_probs - min_q_actor)).sum(dim=1).mean()

			# Optimize actor
			self.actor_optim.zero_grad()
			actor_loss.backward()
			self.actor_optim.step()

			losses['loss/actor'] = actor_loss.item()
			losses['state/entropy'] = entropy.mean().item()

			# Unfreeze critic gradients
			for p in self.critic1.parameters(): p.requires_grad = True
			for p in self.critic2.parameters(): p.requires_grad = True

			# --- Alpha Update ---
			if self._is_auto_alpha:
				alpha_loss = -(self._log_alpha * (entropy.detach() + self.target_entropy)).mean()
				self._alpha_optim.zero_grad()
				alpha_loss.backward()
				self._alpha_optim.step()
				self._alpha = self._log_alpha.detach().exp()
				losses['loss/alpha'] = alpha_loss.item()
				losses['alpha'] = self._alpha.item()
			else:
				losses['alpha'] = self._alpha # Log fixed alpha

			self._behavior_updates += 1

			# --- Target Network Update ---
			if self._behavior_updates % self.target_update_freq == 0:
				self._sync_weight()

		return losses

	def train(self):
		if "store_init" in self.config and self.config["store_init"]:
			return
		
		self.actor.train()
		self.critic1.train()
		self.critic2.train()
		
		observations = self.env.reset()
		episode_rewards = np.zeros(self.num_envs)
		episode_lens = np.zeros(self.num_envs) # Length in primitive steps
		micro_episode_lens = np.zeros(self.num_envs) # Length in macro actions
		
		# Track steps per environment to ensure even sampling
		env_steps = np.zeros(self.num_envs)
		update_steps = 0
		
		while self.total_time_step <= self.training_steps:
			# Decide macro-action
			if self.total_time_step < self.warmup_steps:
				# Sample random macro actions during warmup
				macro_actions = np.random.randint(0, self.N_action, size=self.num_envs)
			else:
				# Use the base class method for training action selection
				macro_actions = super().decide_agent_actions(observations, eval=False)
			
			# Decode macro-action to primitive actions
			decoded_actions = self.decoded_primitive_actions(observations, macro_actions)
			
			# Step environment with decoded primitive actions
			next_observations, rewards, terminates, infos = self.env.step(decoded_actions)
			
			primitive_steps_taken = decoded_actions.shape[1]
			env_steps += primitive_steps_taken

			# Store transition in replay buffer (state, macro_action, reward, next_state, done)
			for i in range(self.num_envs):
				self.replay_buffer.append(
					observations[i],
					[macro_actions[i]], # Store the macro action
					[rewards[i]], # Store accumulated reward
					next_observations[i],
					[int(terminates[i])]
				)

			# Update networks after warmup period and with frequency control
			if (self.total_time_step >= self.warmup_steps and 
				update_steps % self.config["update_freq"] == 0 and
				len(self.replay_buffer) >= self.batch_size):
				
				# Normalize advantage by number of environments
				update_losses = self.update()
				if update_losses:
					for key, value in update_losses.items():
						self.writer.add_scalar(f'SAC/{key}', value / self.num_envs, self.total_time_step)
			
			update_steps += 1
			episode_rewards += rewards
			episode_lens += primitive_steps_taken
			micro_episode_lens += 1

			for i in range(self.num_envs):
				if terminates[i]:
					if i == 0:  # Log only for the first environment
						steps_taken = env_steps[i]
						self.writer.add_scalar('Train/Steps_Per_Episode', steps_taken, self.total_time_step)
						self.writer.add_scalar('Train/Episode_Reward', episode_rewards[i], self.total_time_step)
						self.writer.add_scalar('Train/Episode_Length', episode_lens[i], self.total_time_step)
						self.writer.add_scalar('Train/Macro_Episode_Length', micro_episode_lens[i], self.total_time_step)
						if self._is_auto_alpha:
							self.writer.add_scalar('SAC/Alpha', self._alpha.item(), self.total_time_step)
						else:
							self.writer.add_scalar('SAC/Alpha', self._alpha, self.total_time_step)
					
					print(f"[{len(self.replay_buffer)}/{self.replay_buffer_capacity}][{self.total_time_step}/{self.training_steps}] "
						  f"Env {i} Term: Reward: {episode_rewards[i]:.2f} Steps: {env_steps[i]}")
					
					episode_rewards[i] = 0
					episode_lens[i] = 0
					micro_episode_lens[i] = 0
					env_steps[i] = 0
					
			observations = next_observations
			self.total_time_step += self.num_envs 

			# Evaluate model periodically
			if self.total_time_step % self.eval_interval == 0 and self.total_time_step > 0:
				avg_score, norm_avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode_Reward', avg_score, self.total_time_step)
				self.writer.add_scalar('Evaluate/Normalized_Episode_Reward', norm_avg_score, self.total_time_step)
				# No BC logging needed for SAC directly here
				
				# Write data to csv file (adjust fields)
				csv_path = os.path.join(self.writer.log_dir, 'experiment_data.csv')
				file_exists = os.path.isfile(csv_path)
				with open(csv_path, 'a', newline='') as f:
					fieldnames = ['training_step', 'episode_reward', 'norm_episode_reward'] # Removed BC fields
					writer = csv.DictWriter(f, fieldnames=fieldnames)
					if not file_exists:
						writer.writeheader()
					writer.writerow({'training_step': self.total_time_step, 'episode_reward': avg_score, 'norm_episode_reward': norm_avg_score})
				
				# Switch back to training mode after evaluation
				self.actor.train()
				self.critic1.train()
				self.critic2.train()

	
	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		episode_rewards = np.zeros(self.eval_episode)
		all_rewards = np.zeros(self.eval_episode)
		all_done = [False] * self.eval_episode
		observations = self.test_env.reset()
		total = 0.0001
		correct = 0
		total_bc_actions_log_prob = np.zeros(self.eval_episode)
		all_bc_actions_log_prob = np.zeros(self.eval_episode)
		while True:
			actions, logits = self.decide_agent_actions_with_logits(observations, eval=True)
			decoded_actions = self.decoded_primitive_actions(observations, actions)
			next_observations, rewards, terminates, infos = self.test_env.step(decoded_actions)

			episode_rewards += rewards
			for i in range(self.eval_episode):
				if terminates[i] and not all_done[i]:
					print(f"env {i} terminated, reward: {episode_rewards[i]}")
					all_rewards[i] = episode_rewards[i]
					all_done[i] = True

			observations = next_observations
			
			# all episodes done, terminate
			if all(all_done):
				break
			
		avg = np.mean(all_rewards)
		normalized_avg = self.dummy_env.get_normalized_score(avg) * 100
		print(f"average score: {avg}  average normalized score: {normalized_avg}")
		print("==============================================")
		return avg, normalized_avg



"""
SUFFIX_BASE="_bind_source"
if [ "$SEQLEN" -ne 3 ]; then
  SUFFIX_BASE+="_sq${SEQLEN}"
fi
if [ "$K" -ne 16 ]; then # Only add _k{K} if not the default
  SUFFIX_BASE+="_k${K}"
fi
SUFFIX="${SUFFIX_BASE}_seed${SEED}"

"""
if __name__ == '__main__':
    import argparse
    import moviepy.video.io.ImageSequenceClip

    parser = argparse.ArgumentParser(description='Train and test MAQ-SAC agent.')
    parser.add_argument('--env', type=str, default='door', help='Environment name')
    parser.add_argument('--seqlen', type=int, default=3, help='Sequence length')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--n_episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--render', action='store_true', help='Whether to render and save videos during testing')
    parser.add_argument('--K', type=int, default=16, help='Number of macro actions')
    args = parser.parse_args()

    suffix = f"_bind_source"
    if args.seqlen != 3:
        suffix += f"_sq{args.seqlen}"
    if args.K != 16:
        suffix += f"_k{args.K}"
    suffix += f"_seed{args.seed}"
    num_envs = 8
    # Training configuration
    config = {
        "gpu": True,
        "training_steps": 1e6,
        "warmup_steps": 1000 * num_envs,  # Scale with num_envs - minimum steps before starting updates
        "batch_size": 256,
        "logdir": f'log_test/{args.env}_human_seqlen{args.seqlen}/reg3{suffix}',
        "learning_rate": 3e-4,
        "eval_interval": 10000,
        "num_envs": num_envs,
        "eval_episode": 10,
        "env_id": f"{args.env}-human-v1",
        "seed": args.seed,
        "vqvae_model_path": f"../VQVAE/log/{args.env}-human_seqlen{args.seqlen}_extra_jt{suffix}/",
        "prior_model_path": f"../VQVAE/log/prior/{args.env}-human_seqlen{args.seqlen}_extra_jt{suffix}/",
        # SAC specific parameters
        "update_freq": num_envs ,  # Reduce update frequency when using multiple envs
        "hidden_dim": 256,
        "actor_lr": 3e-4 ,  # Scale down learning rates with num_envs
        "critic_lr": 1e-3,
        "alpha_lr": 3e-4,
        "target_entropy": None,  # Will be set automatically based on action space
        "tau": 0.005,
        "gamma": 0.99,
        "replay_buffer_capacity": 10000,  # Scale buffer with num_envs
        "deterministic_eval": True,
        "auto_alpha": True,
        "initial_alpha": 0.2,
    }
    
    # Create directories
    os.makedirs(config["logdir"], exist_ok=True)
    
    # Save config
    with open(os.path.join(config["logdir"], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Starting training...")
    # Initialize and train agent
    agent = MAQDRLPDSACAgent(config)
    agent.train()
    
    print("\nTraining completed. Starting evaluation...")
    
    # Run evaluation
    episode_rewards = np.zeros(args.n_episodes)
    norm_episode_rewards = np.zeros(args.n_episodes)
    
    for i in range(args.n_episodes):
        env = gym.make(agent.config['env_id'])
        if args.render:
            viewer = env.mj_viewer_setup()  # For Adroit environments
            
        obs_list = []
        observation = env.reset()
        total_reward = 0
        total_length = 0
        
        while True:
            if args.render:
                obs_list.append(env.viewer._read_pixels_as_in_window())
                
            observation = np.expand_dims(observation, axis=0)
            action, _ = agent.decide_agent_actions_with_logits(observation, eval=True)
            decoded_actions = agent.decoded_primitive_actions(observation, action)
            
            for a in decoded_actions[0]:
                next_observation, reward, done, _ = env.step(a)
                if args.render:
                    obs_list.append(env.viewer._read_pixels_as_in_window())
                total_reward += reward
                total_length += 1
                if done:
                    break
                    
            if done:
                break
            observation = next_observation
            
        episode_rewards[i] = total_reward
        norm_episode_rewards[i] = env.get_normalized_score(total_reward) * 100
        print(f"Test Episode {i+1}, Total reward: {total_reward:.2f}, Total length: {total_length}, Normalized reward: {norm_episode_rewards[i]:.2f}")
        
        if args.render:
            os.makedirs(os.path.join(config["logdir"], 'videos'), exist_ok=True)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(obs_list, fps=30)
            clip.write_videofile(os.path.join(config["logdir"], 'videos', f'test_ep{i+1}_reward_{int(total_reward)}.mp4'))
    
    print("\nFinal Test Results:")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average normalized reward: {np.mean(norm_episode_rewards):.2f} ± {np.std(norm_episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}, Max normalized reward: {np.max(norm_episode_rewards):.2f}")

    # Save test results
    results = {
        'episode_rewards': episode_rewards.tolist(),
        'norm_episode_rewards': norm_episode_rewards.tolist(),
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_norm_reward': float(np.mean(norm_episode_rewards)),
        'std_norm_reward': float(np.std(norm_episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'max_norm_reward': float(np.max(norm_episode_rewards))
    }
    
    with open(os.path.join(config["logdir"], 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {config['logdir']}")
    
    # Example commands:
    # 1. Train and test:
    # python DSACMAQ.py --env door --seed 1000
    # 2. Train and test with rendering:
    # env CUDA_VISIBLE_DEVICES=2 xvfb-run -s "-screen 0 1024x768x24" python DSACMAQ.py --env door --render
	



