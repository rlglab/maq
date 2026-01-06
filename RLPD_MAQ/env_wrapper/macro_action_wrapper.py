import gym

class MacroActionEnvWrapper(gym.Env):
	metadata = {}
	def __init__(self, game_name, K,seed=None):
		self.game_name = game_name
		self.env = gym.make(game_name)
		if seed != None:
			# self.env.seed(seed)
			self.env.action_space.seed(seed)
			self.env.observation_space.seed(seed)
		self.action_space = gym.spaces.Discrete(K)
		self.observation_space = self.env.observation_space
		self.length = 0
	
	def step(self, macro_action):
		total_reward = 0
		for a in macro_action:
			observation, reward, done, info = self.env.step(a)
			self.length += 1
			total_reward += reward
			
			if done:
				break

		return observation, total_reward, done, info
	
	def reset(self):
		observation = self.env.reset()
		self.length = 0
		return observation