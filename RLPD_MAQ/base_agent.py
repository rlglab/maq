import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
from copy import deepcopy

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = int(2**16) if "eval_interval" not in config else int(config["eval_interval"])
		self.eval_episode = 16 if "eval_episode" not in config else int(config["eval_episode"])
		self.num_envs = config["num_envs"] if "num_envs" in config else 1

		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : 512 if "horizon" not in config else int(config["horizon"]), 
			"use_return_as_advantage": False,
			"episode_sequence_size": 8,
			"agent_count": self.num_envs,
			})

		self.writer = SummaryWriter(config["logdir"])

	def train(self):
		# self.evaluate()
		self.net.train()
		observations, infos = self.env.reset()
		episode_rewards = [0] * self.num_envs
		episode_lens = [0] * self.num_envs
		
		while self.total_time_step <= self.training_steps:
			actions, values, logp_pis = self.decide_agent_actions(observations)
			actions = actions.cpu().detach().numpy()
			values = values.cpu().detach().numpy()
			logp_pis = logp_pis.cpu().detach().numpy()
			
			next_observations, rewards, terminates, truncates, infos = self.env.step(actions)

			for i in range(self.num_envs):
				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				obs = {}
				obs["observation_2d"] = np.asarray(observations[i], dtype=np.float32)
				self.gae_replay_buffer.append(i, {
						"observation": obs,
						"action": np.array([actions[i]]),
						"reward": rewards[i],
						"value": values[i],
						"logp_pi": logp_pis[i],
						"done": terminates[i],
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()
					break

			episode_rewards = [episode_rewards[i] + rewards[i] for i in range(self.num_envs)]
			episode_lens = [episode_lens[i] + 1 for i in range(self.num_envs)]

			for i in range(self.num_envs):
				if terminates[i] or truncates[i]:
					if i == 0:
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[0], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[0], self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]\
						\tenv {i} \
	   					\tepisode reward: {episode_rewards[i]}\
						\tepisode len: {episode_lens[i]}\
						")
					episode_rewards[i] = 0
					episode_lens[i] = 0

			observations = next_observations
			self.total_time_step += self.num_envs
			

			if self.total_time_step % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	
	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		self.net.eval()
		episode_rewards = [0] * self.eval_episode
		all_rewards = [0] * self.eval_episode
		all_done = [False] * self.eval_episode
		observations, infos = self.test_env.reset()
		while True:
			actions, values, logp_pis = self.decide_agent_actions(observations)
			actions = actions.cpu().detach().numpy()
			
			next_observations, rewards, terminates, truncates, infos = self.test_env.step(actions)
			for i in range(self.eval_episode):
				if (terminates[i] or truncates[i]) and not all_done[i]:
					print(f"env {i} terminated, reward: {episode_rewards[i]}")
					all_rewards[i] = episode_rewards[i]
					all_done[i] = True

			episode_rewards = [episode_rewards[i] + rewards[i] for i in range(self.eval_episode)]
			observations = next_observations
			
			# all episodes done, terminate
			if all(all_done):
				break
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		self.net.train()
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		p_loss_counter = 0.0001
		v_loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		# for _ in range(self.update_count):
		# 	for start in range(0, sample_count, self.batch_size):
		#		ob_train_batch = {}
		#		for key in observation_batch:
		#			ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
		# 		ac_train_batch = action_batch[start:start + self.batch_size]
		# 		return_train_batch = return_batch[start:start + self.batch_size]
		# 		adv_train_batch = adv_batch[start:start + self.batch_size]
		# 		v_train_batch = v_batch[start:start + self.batch_size]
		# 		logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

		# 		adv_train_batch = torch.from_numpy(adv_train_batch)
		# 		adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
		# 		logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
		# 		logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
		# 		return_train_batch = torch.from_numpy(return_train_batch)
		# 		return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

		# 		_, logp, value = self.net(ob_train_batch, ac_train_batch)

		# 		# policy block
		# 		total_ratio = torch.exp(logp - logp_pi_train_batch)
		# 		p_opt_a = total_ratio * adv_train_batch
		# 		p_opt_b = torch.clamp(total_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_train_batch
		# 		surrogate_loss = -torch.mean(torch.min(p_opt_a, p_opt_b))
		# 		# value block
		# 		value_criterion = nn.MSELoss()
		# 		v_loss = value_criterion(value, return_train_batch)

		# 		loss = surrogate_loss + v_loss
		# 		self.optim.zero_grad()
		# 		loss.backward()
		# 		nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
		# 		self.optim.step()

		# 		total_surrogate_loss += surrogate_loss.item()
		# 		total_v_loss += v_loss.item()
		# 		total_loss += loss.item()
		# 		p_loss_counter += 1

		# self.writer.add_scalar('PPO/Loss', total_loss / p_loss_counter, self.total_time_step)
		# self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / p_loss_counter, self.total_time_step)
		# self.writer.add_scalar('PPO/Value Loss', total_v_loss / p_loss_counter, self.total_time_step)
		# print(f"Loss: {total_loss / p_loss_counter}\
		# 	\tSurrogate Loss: {total_surrogate_loss / p_loss_counter}\
		# 	\tValue Loss: {total_v_loss / p_loss_counter}\
		# 	")


class DQNBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.epsilon = 1.0
		self.eps_min = config["eps_min"]
		self.eps_decay = config["eps_decay"]
		self.eval_epsilon = config["eval_epsilon"]
		self.warmup_steps = config["warmup_steps"]
		self.use_double = config["use_double"]
		self.eval_interval = int(2**16)
		self.eval_episode = 16
		self.num_envs = config["num_envs"]
		self.gamma = config["gamma"]
		self.update_freq = config["update_freq"]
		self.update_target_freq = config["update_target_freq"]
	
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		# get action from behavior net, with epsilon-greedy selection

		return NotImplementedError
	
	def update(self):
		if self.total_time_step % self.update_freq == 0:
			self.update_behavior_network()
		if self.total_time_step % self.update_target_freq == 0:
			self.update_target_network()

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		action = action.type(torch.long)
		q_value = self.behavior_net(state).gather(1, action)
		with torch.no_grad():
			if self.use_double:
				q_next = self.behavior_net(next_state)
				action_index = q_next.max(dim=1)[1].view(-1, 1)
				# choose related Q from target net
				q_next = self.target_net(next_state).gather(dim=1, index=action_index.long())
			else:
				q_next = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)

			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)
        
		criterion = nn.SmoothL1Loss()
		# criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		# nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 5)
		for param in self.behavior_net.parameters():
			param.grad.data.clamp_(-1, 1)
        
		self.optim.step()

	def update_target_network(self):
		self.target_net.load_state_dict(self.behavior_net.state_dict())
	
	def epsilon_decay(self):
		self.epsilon -= (1 - self.eps_min) / self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_min)

	def train(self):
		self.behavior_net.train()
		observations, infos = self.env.reset()
		episode_rewards = [0] * self.num_envs
		episode_lens = [0] * self.num_envs
		
		while self.total_time_step <= self.training_steps:
			if self.total_time_step < self.warmup_steps:
				actions = self.decide_agent_actions(observations, 1.0, self.env.action_space.n)
			else:
				actions = self.decide_agent_actions(observations, self.epsilon, self.env.action_space.n)
				self.epsilon_decay()

			next_observations, rewards, terminates, truncates, infos = self.env.step(actions)

			for i in range(self.num_envs):
				self.replay_buffer.append(
						observations[i],
						[actions[i]],
						[rewards[i]],
						next_observations[i],
						[int(terminates[i])]
					)

			if self.total_time_step >= self.warmup_steps:
				self.update()

			episode_rewards = [episode_rewards[i] + rewards[i] for i in range(self.num_envs)]
			episode_lens = [episode_lens[i] + 1 for i in range(self.num_envs)]

			for i in range(self.num_envs):
				if terminates[i] or truncates[i]:
					if i == 0:
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[0], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[0], self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps}]\
						\tenv {i} \
	   					\tepisode reward: {episode_rewards[i]}\
						\tepisode len: {episode_lens[i]}\
						\tepsilon: {self.epsilon}\
						")
					episode_rewards[i] = 0
					episode_lens[i] = 0

			observations = next_observations
			self.total_time_step += self.num_envs
			
			if self.total_time_step % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		self.behavior_net.eval()
		episode_rewards = [0] * self.eval_episode
		all_rewards = [0] * self.eval_episode
		all_done = [False] * self.eval_episode
		observations, infos = self.test_env.reset()
		while True:
			actions = self.decide_agent_actions(observations, self.eval_epsilon, self.test_env.action_space.n)
			next_observations, rewards, terminates, truncates, infos = self.test_env.step(actions)
			for i in range(self.eval_episode):
				if (terminates[i] or truncates[i]) and not all_done[i]:
					print(f"env {i} terminated, reward: {episode_rewards[i]}")
					all_rewards[i] = episode_rewards[i]
					all_done[i] = True

			episode_rewards = [episode_rewards[i] + rewards[i] for i in range(self.eval_episode)]
			observations = next_observations
			
			# all episodes done, terminate
			if all(all_done):
				break
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		self.behavior_net.train()
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.behavior_net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.behavior_net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

class GaussianNoise:
	def __init__(self, dim, mu=None, std=None):
		self.mu = mu if mu else np.zeros(dim)
		self.std = np.ones(dim) * std if std else np.ones(dim) * .1
	
	def reset(self):
		pass

	def generate(self):
		return np.random.normal(self.mu, self.std)

class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x

class TD3BaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.total_episode = config["total_episode"] if "total_episode" in config else 1
		self.eval_interval = int(10)
		self.eval_episode = 10
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config["update_freq"]
		# exploration noise
		# self.exploration_noise = GaussianNoise(dim=action_dim, std=args.exploration_noise)
		# policy noise
		# self.policy_noise = GaussianNoise(dim=action_dim, std=args.policy_noise)
	
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, state, sigma=0.0):
		'''based on the behavior (actor) network and exploration noise'''
		with torch.no_grad():
			# sample_noise = torch.from_numpy(self.exploration_noise.generate()).view(1,-1).to(self.device)
			state = torch.FloatTensor(state).unsqueeze(0)
			action = self.actor_net(state.to(self.device))
			# action = (action + sigma * sample_noise).clamp(-1*self.max_action, self.max_action)
	
		return action.cpu().numpy().squeeze()
		
	
	def update(self):
		# update the behavior networks
		self.update_behavior_network()
		# update the target networks
		if self.total_time_step % self.update_freq == 0:
			self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
			self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
			self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		## update critic ##
		# critic loss
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)
		with torch.no_grad():
			sample_noise = torch.from_numpy(self.policy_noise.generate()).float().view(1,-1).to(self.device).clamp(-1*self.noise_clip, self.noise_clip)
			a_next = (a_next + sample_noise).clamp(-1*self.max_action, self.max_action)

			q_next1 = self.target_critic_net1(next_state, a_next)
			q_next2 = self.target_critic_net2(next_state, a_next)
			q_target = reward + self.gamma * torch.min(q_next1, q_next2) * (1 - done)
		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_net2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		if self.total_time_step % self.update_freq == 0:
			## update actor ##
			# actor loss
			# select action a from behavior actor network (a is different from sample transition's action)
			# get Q from behavior critic network, mean Q value -> objective function
			# maximize (objective function) = minimize -1 * (objective function)
			action = self.actor_net(state)
			actor_loss = -1 * (self.critic_net1(state, action).mean())
			# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()

	@staticmethod
	def update_target_network(target_net, net, tau):
		'''update target network by _soft_ copying from behavior network'''
		for target, behavior in zip(target_net.parameters(), net.parameters()):
			target.data.copy_((1 - tau) * target.data + tau * behavior.data)
	
	def train(self):
		for episode in range(self.total_episode):
			total_reward = 0
			state, infos = self.env.reset()
			self.noise.reset()
			for t in range(10000):
				if self.total_time_step < self.warmup_steps:
					action = self.env.action_space.sample()
				else:
					# exploration degree
					sigma = max(0.1*(1-episode/self.total_episode), 0.01)
					action = self.decide_agent_actions(state, sigma=sigma)
				
				next_state, reward, terminates, truncates, _ = self.env.step(action)
				self.replay_buffer.append(state, action, [reward/10], next_state, [int(terminates)])
				if self.total_time_step >= self.warmup_steps:
					self.update()

				self.total_time_step += 1
				total_reward += reward
				state = next_state
				if terminates or truncates:
					self.writer.add_scalar('Train/Episode Reward', total_reward, self.total_time_step)
					print(
						'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(self.total_time_step, episode+1, t, total_reward))
				
					break
			
			if (episode+1) % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for episode in range(self.eval_episode):
			total_reward = 0
			state, infos = self.test_env.reset()
			for t in range(10000):
				action = self.decide_agent_actions(state)
				next_state, reward, terminates, truncates, _ = self.test_env.step(action)
				# print(f"action: {action}")
				total_reward += reward
				state = next_state
				if terminates or truncates:
					print(
						'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(episode+1, t, total_reward))
					all_rewards.append(total_reward)
					break

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
					'critic2': self.critic_net2.state_dict(),
				}, save_path)

	# load model
	def load(self, load_path):
		checkpoint = torch.load(load_path)
		self.actor_net.load_state_dict(checkpoint['actor'])
		self.critic_net1.load_state_dict(checkpoint['critic1'])
		self.critic_net2.load_state_dict(checkpoint['critic2'])

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

# --- Added SAC Actor Network ---
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super().__init__()
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)

	def forward(self, state, state_out=None, info=None): # Added dummy args for compatibility
		x = F.relu(self.l1(state))
		x = F.relu(self.l2(x))
		logits = self.l3(x)
		# Return logits directly for Categorical distribution
		# Also return None for hidden state to match expected output format
		return logits, None

# --- Added SAC Critic Network ---
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super().__init__()
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim) # Output Q-value for each discrete action

	def forward(self, state):
		x = F.relu(self.l1(state))
		x = F.relu(self.l2(x))
		q_values = self.l3(x)
		return q_values


# --- Added DiscreteSACAgent ---
class DiscreteSACAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.eval_interval = int(config.get("eval_interval", 2**16))
		self.eval_episode = int(config.get("eval_episode", 16))
		self.num_envs = int(config.get("num_envs", 1))
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config.get("update_freq", 1) # How often to update networks
		self.target_update_freq = config.get("target_update_freq", 2) # How often to update target networks (in terms of behavior updates)
		
		self.replay_buffer_capacity = int(config["replay_buffer_capacity"])
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

		# Networks (will be defined in subclasses)
		self.actor: Actor = None
		self.critic1: Critic = None
		self.critic2: Critic = None
		self.actor_optim: torch.optim.Optimizer = None
		self.critic1_optim: torch.optim.Optimizer = None
		self.critic2_optim: torch.optim.Optimizer = None

		# Target networks
		self.critic1_old: Critic = None
		self.critic2_old: Critic = None

		# Alpha (Entropy temperature)
		self.alpha_config = config.get("alpha", 0.2)
		self._is_auto_alpha = not isinstance(self.alpha_config, float)
		self._alpha: Union[float, torch.Tensor]
		if self._is_auto_alpha:
			try:
				self.target_entropy = float(self.alpha_config['target_entropy'])
				log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
				alpha_optim = torch.optim.Adam([log_alpha], lr=self.alpha_config['lr'])
				self._log_alpha = log_alpha
				self._alpha_optim = alpha_optim
				self._alpha = self._log_alpha.detach().exp()
			except KeyError:
				raise ValueError("alpha config dict must contain 'target_entropy' and 'lr' for auto alpha tuning.")
		else:
			self._alpha = self.alpha_config

		self._deterministic_eval = config.get("deterministic_eval", True)
		self._rew_norm = config.get("reward_normalization", False) # Not implemented in this basic version yet
		self._n_step = config.get("estimation_step", 1) # N-step return, not implemented in this basic version yet
		self._behavior_updates = 0 # Counter for target network updates


	def _initialize_networks(self, actor_cls, critic_cls, state_dim, action_dim, hidden_dim=256, actor_lr=1e-4, critic_lr=1e-3):
		"""Helper to initialize networks and optimizers. Call this in subclass __init__."""
		self.actor = actor_cls(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic1 = critic_cls(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic2 = critic_cls(state_dim, action_dim, hidden_dim).to(self.device)

		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
		self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

		self.critic1_old = deepcopy(self.critic1)
		self.critic1_old.eval()
		self.critic2_old = deepcopy(self.critic2)
		self.critic2_old.eval()

	def _sync_weight(self):
		"""Soft update target networks."""
		for target_param, param in zip(self.critic1_old.parameters(), self.critic1.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
		for target_param, param in zip(self.critic2_old.parameters(), self.critic2.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	@abstractmethod
	def _preprocess_state(self, state):
		"""Convert raw environment observation to network input tensor."""
		raise NotImplementedError

	@abstractmethod
	def _postprocess_action(self, action_tensor):
		"""Convert network output action tensor to environment action format."""
		raise NotImplementedError

	def decide_agent_actions(self, observation, eval=False):
		"""Get actions from actor network."""
		self.actor.eval() # Set actor to eval mode for action selection
		with torch.no_grad():
			state_tensor = self._preprocess_state(observation)
			logits, _ = self.actor(state_tensor)
			dist = Categorical(logits=logits)

			if eval and self._deterministic_eval:
				action_tensor = logits.argmax(dim=-1)
			else:
				action_tensor = dist.sample()

		self.actor.train() # Set actor back to train mode
		return self._postprocess_action(action_tensor)

	def update(self):
		"""Update actor and critic networks."""
		if len(self.replay_buffer) < self.batch_size:
			return {} # Not enough samples yet

		self.actor.train()
		self.critic1.train()
		self.critic2.train()

		losses = {}

		for _ in range(self.update_freq): # Perform multiple updates per call if update_freq > 1
			state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
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
		"""Main training loop structure (adapt in subclass)."""
		self.actor.train()
		self.critic1.train()
		self.critic2.train()
		observations, infos = self.env.reset() # Assumes self.env exists in subclass
		episode_rewards = np.zeros(self.num_envs)
		episode_lens = np.zeros(self.num_envs)

		while self.total_time_step <= self.training_steps:
			if self.total_time_step < self.warmup_steps:
				# Sample random actions during warmup
				actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)]) # Assumes env.action_space
			else:
				actions = self.decide_agent_actions(observations, eval=False)

			next_observations, rewards, terminates, truncates, infos = self.env.step(actions) # Assumes self.env

			# Store transition in replay buffer
			for i in range(self.num_envs):
				 # Assuming observation is numpy array, action is numpy scalar/array
				self.replay_buffer.append(
					observations[i],
					[actions[i]] if np.isscalar(actions[i]) else actions[i], # Ensure action is list/array
					[rewards[i]],
					next_observations[i],
					[int(terminates[i])]
				)


			# Update networks
			if self.total_time_step >= self.warmup_steps:
				update_losses = self.update()
				# Log losses
				if update_losses:
					for key, value in update_losses.items():
						self.writer.add_scalar(f'SAC/{key}', value, self.total_time_step)
			episode_rewards += rewards
			episode_lens += 1 # Assuming each step adds 1 to length

			# Handle episode termination and logging
			for i in range(self.num_envs):
				done = terminates[i] or truncates[i]
				if done:
					if i == 0: # Log only for the first environment for simplicity
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[i], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[i], self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps}] Env {i} Episode End: Reward={episode_rewards[i]:.2f} Len={episode_lens[i]}")
					episode_rewards[i] = 0
					episode_lens[i] = 0
					# Resetting is handled by VecEnv automatically if using AsyncVectorEnv

			observations = next_observations
			self.total_time_step += self.num_envs

			# Evaluate model periodically
			if self.total_time_step % self.eval_interval == 0:
				avg_score = self.evaluate() # Assumes self.evaluate exists
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth")) # Assumes self.save exists
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				# Switch back to training mode
				self.actor.train()
				self.critic1.train()
				self.critic2.train()


	def evaluate(self):
		"""Evaluation loop structure (adapt in subclass)."""
		print("==============================================")
		print("Evaluating...")
		self.actor.eval() # Use eval mode
		self.critic1.eval()
		self.critic2.eval()

		episode_rewards = np.zeros(self.eval_episode)
		all_rewards = np.zeros(self.eval_episode)
		episode_lens = np.zeros(self.eval_episode)
		all_done = [False] * self.eval_episode
		observations, infos = self.test_env.reset() # Assumes self.test_env

		while not all(all_done):
			actions = self.decide_agent_actions(observations, eval=True)
			next_observations, rewards, terminates, truncates, infos = self.test_env.step(actions) # Assumes self.test_env

			current_rewards = rewards.copy()
			current_lens = np.ones(self.eval_episode)

			for i in range(self.eval_episode):
				if not all_done[i]: # Only update if the episode is not finished
					episode_rewards[i] += current_rewards[i]
					episode_lens[i] += current_lens[i]
					done = terminates[i] or truncates[i]
					if done:
						print(f"Eval Env {i} terminated: Reward={episode_rewards[i]:.2f} Len={episode_lens[i]}")
						all_rewards[i] = episode_rewards[i]
						all_done[i] = True

			observations = next_observations

		avg_score = np.mean(all_rewards)
		print(f"Average Evaluation Score: {avg_score:.2f}")
		print("==============================================")
		# Return to training mode handled in train loop after evaluation call
		return avg_score

	def save(self, save_path):
		"""Save model weights."""
		print(f"Saving model to {save_path}")
		torch.save({
			'actor_state_dict': self.actor.state_dict(),
			'critic1_state_dict': self.critic1.state_dict(),
			'critic2_state_dict': self.critic2.state_dict(),
			'actor_optimizer_state_dict': self.actor_optim.state_dict(),
			'critic1_optimizer_state_dict': self.critic1_optim.state_dict(),
			'critic2_optimizer_state_dict': self.critic2_optim.state_dict(),
			'log_alpha': self._log_alpha if self._is_auto_alpha else None,
			'alpha_optimizer_state_dict': self._alpha_optim.state_dict() if self._is_auto_alpha else None,
		}, save_path)

	def load(self, load_path, evaluate=False):
		"""Load model weights."""
		print(f"Loading model from {load_path}")
		checkpoint = torch.load(load_path, map_location=self.device)
		self.actor.load_state_dict(checkpoint['actor_state_dict'])
		self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
		self.critic2.load_state_dict(checkpoint['critic2_state_dict'])

		self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
		self.critic1_optim.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
		self.critic2_optim.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

		# Reload target networks
		self.critic1_old = deepcopy(self.critic1)
		self.critic1_old.eval()
		self.critic2_old = deepcopy(self.critic2)
		self.critic2_old.eval()

		if self._is_auto_alpha and 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
			self._log_alpha.data.copy_(checkpoint['log_alpha'])
			self._alpha = self._log_alpha.detach().exp()
			if 'alpha_optimizer_state_dict' in checkpoint and checkpoint['alpha_optimizer_state_dict'] is not None:
				self._alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

		if evaluate:
			self.evaluate()



class RLPDDiscreteSACAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.eval_interval = int(config.get("eval_interval", 2**16))
		self.eval_episode = int(config.get("eval_episode", 16))
		self.num_envs = int(config.get("num_envs", 1))
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config.get("update_freq", 1) # How often to update networks
		self.target_update_freq = config.get("target_update_freq", 2) # How often to update target networks (in terms of behavior updates)
		
		self.replay_buffer_capacity = int(config["replay_buffer_capacity"])
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

		# Networks (will be defined in subclasses)
		self.actor: Actor = None
		self.critic1: Critic = None
		self.critic2: Critic = None
		self.actor_optim: torch.optim.Optimizer = None
		self.critic1_optim: torch.optim.Optimizer = None
		self.critic2_optim: torch.optim.Optimizer = None

		# Target networks
		self.critic1_old: Critic = None
		self.critic2_old: Critic = None

		# Alpha (Entropy temperature)
		self.alpha_config = config.get("alpha", 0.2)
		self._is_auto_alpha = not isinstance(self.alpha_config, float)
		self._alpha: Union[float, torch.Tensor]
		if self._is_auto_alpha:
			try:
				self.target_entropy = float(self.alpha_config['target_entropy'])
				log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
				alpha_optim = torch.optim.Adam([log_alpha], lr=self.alpha_config['lr'])
				self._log_alpha = log_alpha
				self._alpha_optim = alpha_optim
				self._alpha = self._log_alpha.detach().exp()
			except KeyError:
				raise ValueError("alpha config dict must contain 'target_entropy' and 'lr' for auto alpha tuning.")
		else:
			self._alpha = self.alpha_config

		self._deterministic_eval = config.get("deterministic_eval", True)
		self._rew_norm = config.get("reward_normalization", False) # Not implemented in this basic version yet
		self._n_step = config.get("estimation_step", 1) # N-step return, not implemented in this basic version yet
		self._behavior_updates = 0 # Counter for target network updates


	def _initialize_networks(self, actor_cls, critic_cls, state_dim, action_dim, hidden_dim=256, actor_lr=1e-4, critic_lr=1e-3):
		"""Helper to initialize networks and optimizers. Call this in subclass __init__."""
		self.actor = actor_cls(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic1 = critic_cls(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic2 = critic_cls(state_dim, action_dim, hidden_dim).to(self.device)

		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
		self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

		self.critic1_old = deepcopy(self.critic1)
		self.critic1_old.eval()
		self.critic2_old = deepcopy(self.critic2)
		self.critic2_old.eval()

	def _sync_weight(self):
		"""Soft update target networks."""
		for target_param, param in zip(self.critic1_old.parameters(), self.critic1.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
		for target_param, param in zip(self.critic2_old.parameters(), self.critic2.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	@abstractmethod
	def _preprocess_state(self, state):
		"""Convert raw environment observation to network input tensor."""
		raise NotImplementedError

	@abstractmethod
	def _postprocess_action(self, action_tensor):
		"""Convert network output action tensor to environment action format."""
		raise NotImplementedError

	def decide_agent_actions(self, observation, eval=False):
		"""Get actions from actor network."""
		self.actor.eval() # Set actor to eval mode for action selection
		with torch.no_grad():
			state_tensor = self._preprocess_state(observation)
			logits, _ = self.actor(state_tensor)
			dist = Categorical(logits=logits)

			if eval and self._deterministic_eval:
				action_tensor = logits.argmax(dim=-1)
			else:
				action_tensor = dist.sample()

		self.actor.train() # Set actor back to train mode
		return self._postprocess_action(action_tensor)

	def update(self):
		"""Update actor and critic networks."""
		raise NotImplementedError


	def train(self):
		"""Main training loop structure (adapt in subclass)."""
		self.actor.train()
		self.critic1.train()
		self.critic2.train()
		observations, infos = self.env.reset() # Assumes self.env exists in subclass
		episode_rewards = np.zeros(self.num_envs)
		episode_lens = np.zeros(self.num_envs)

		while self.total_time_step <= self.training_steps:
			if self.total_time_step < self.warmup_steps:
				# Sample random actions during warmup
				actions = np.array([self.env.action_space.sample() for _ in range(self.num_envs)]) # Assumes env.action_space
			else:
				actions = self.decide_agent_actions(observations, eval=False)

			next_observations, rewards, terminates, truncates, infos = self.env.step(actions) # Assumes self.env

			# Store transition in replay buffer
			for i in range(self.num_envs):
				 # Assuming observation is numpy array, action is numpy scalar/array
				self.replay_buffer.append(
					observations[i],
					[actions[i]] if np.isscalar(actions[i]) else actions[i], # Ensure action is list/array
					[rewards[i]],
					next_observations[i],
					[int(terminates[i])]
				)


			# Update networks
			if self.total_time_step >= self.warmup_steps:
				update_losses = self.update()
				# Log losses
				if update_losses:
					for key, value in update_losses.items():
						self.writer.add_scalar(f'SAC/{key}', value, self.total_time_step)
			episode_rewards += rewards
			episode_lens += 1 # Assuming each step adds 1 to length

			# Handle episode termination and logging
			for i in range(self.num_envs):
				done = terminates[i] or truncates[i]
				if done:
					if i == 0: # Log only for the first environment for simplicity
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[i], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[i], self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps}] Env {i} Episode End: Reward={episode_rewards[i]:.2f} Len={episode_lens[i]}")
					episode_rewards[i] = 0
					episode_lens[i] = 0
					# Resetting is handled by VecEnv automatically if using AsyncVectorEnv

			observations = next_observations
			self.total_time_step += self.num_envs

			# Evaluate model periodically
			if self.total_time_step % self.eval_interval == 0:
				avg_score = self.evaluate() # Assumes self.evaluate exists
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth")) # Assumes self.save exists
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				# Switch back to training mode
				self.actor.train()
				self.critic1.train()
				self.critic2.train()


	def evaluate(self):
		"""Evaluation loop structure (adapt in subclass)."""
		print("==============================================")
		print("Evaluating...")
		self.actor.eval() # Use eval mode
		self.critic1.eval()
		self.critic2.eval()

		episode_rewards = np.zeros(self.eval_episode)
		all_rewards = np.zeros(self.eval_episode)
		episode_lens = np.zeros(self.eval_episode)
		all_done = [False] * self.eval_episode
		observations, infos = self.test_env.reset() # Assumes self.test_env

		while not all(all_done):
			actions = self.decide_agent_actions(observations, eval=True)
			next_observations, rewards, terminates, truncates, infos = self.test_env.step(actions) # Assumes self.test_env

			current_rewards = rewards.copy()
			current_lens = np.ones(self.eval_episode)

			for i in range(self.eval_episode):
				if not all_done[i]: # Only update if the episode is not finished
					episode_rewards[i] += current_rewards[i]
					episode_lens[i] += current_lens[i]
					done = terminates[i] or truncates[i]
					if done:
						print(f"Eval Env {i} terminated: Reward={episode_rewards[i]:.2f} Len={episode_lens[i]}")
						all_rewards[i] = episode_rewards[i]
						all_done[i] = True

			observations = next_observations

		avg_score = np.mean(all_rewards)
		print(f"Average Evaluation Score: {avg_score:.2f}")
		print("==============================================")
		# Return to training mode handled in train loop after evaluation call
		return avg_score

	def save(self, save_path):
		"""Save model weights."""
		print(f"Saving model to {save_path}")
		torch.save({
			'actor_state_dict': self.actor.state_dict(),
			'critic1_state_dict': self.critic1.state_dict(),
			'critic2_state_dict': self.critic2.state_dict(),
			'actor_optimizer_state_dict': self.actor_optim.state_dict(),
			'critic1_optimizer_state_dict': self.critic1_optim.state_dict(),
			'critic2_optimizer_state_dict': self.critic2_optim.state_dict(),
			'log_alpha': self._log_alpha if self._is_auto_alpha else None,
			'alpha_optimizer_state_dict': self._alpha_optim.state_dict() if self._is_auto_alpha else None,
		}, save_path)

	def load(self, load_path, evaluate=False):
		"""Load model weights."""
		print(f"Loading model from {load_path}")
		checkpoint = torch.load(load_path, map_location=self.device)
		self.actor.load_state_dict(checkpoint['actor_state_dict'])
		self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
		self.critic2.load_state_dict(checkpoint['critic2_state_dict'])

		self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
		self.critic1_optim.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
		self.critic2_optim.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

		# Reload target networks
		self.critic1_old = deepcopy(self.critic1)
		self.critic1_old.eval()
		self.critic2_old = deepcopy(self.critic2)
		self.critic2_old.eval()

		if self._is_auto_alpha and 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
			self._log_alpha.data.copy_(checkpoint['log_alpha'])
			self._alpha = self._log_alpha.detach().exp()
			if 'alpha_optimizer_state_dict' in checkpoint and checkpoint['alpha_optimizer_state_dict'] is not None:
				self._alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

		if evaluate:
			self.evaluate()

