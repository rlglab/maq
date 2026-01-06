import numpy as np
import torch
from collections import deque
import random
from torch.utils.data import Dataset


class D4RLGameDataset(Dataset):
    def __init__(self, trajectories, sequence_length=4):
        """
        初始化數據集，將多場遊戲壓縮成一個連續的數據集。

        :param trajectories: List[Dict]，包含多場遊戲的數據 (如 observations, actions 等)
        :param sequence_length: 每個序列的長度
        """
        self.sequence_length = sequence_length

        # 壓縮多個遊戲為單一連續數據
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminals = []

        for traj in trajectories:
            # 确保轨迹长度足够
            if len(traj['observations']) < self.sequence_length:
                continue
                
            self.observations.append(traj['observations'])
            self.actions.append(traj['actions'])
            self.rewards.append(traj['rewards'])
            self.next_observations.append(traj['next_observations'])
            self.terminals.append(traj['terminals'])

        # 合併所有遊戲的數據
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.next_observations = np.concatenate(self.next_observations, axis=0)
        self.terminals = np.concatenate(self.terminals, axis=0)

        # 計算有效的數據範圍 (需要保证后续有足够的序列长度)
        self.valid_indices = len(self.observations) - self.sequence_length + 1
        
        print(f"D4RLGameDataset initialized with {self.valid_indices} valid indices")

    def __len__(self):
        """
        返回數據集的長度。
        """
        return self.valid_indices

    def __getitem__(self, idx):
        """
        返回第 idx 個數據點，包含觀測和动作序列。

        :param idx: 數據點的索引
        :return: Tuple(state, action_seq, reward, next_state, done)
        """
        # 当前状态是序列的起始状态
        state = self.observations[idx]
        # 获取从当前状态开始的动作序列
        action_seq = self.actions[idx:idx + self.sequence_length]
        # 当前的奖励和结束状态
        reward = self.rewards[idx]
        next_state = self.next_observations[idx]
        done = self.terminals[idx]
        
        return state, action_seq, reward, next_state, done
    

class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def load_from_dataset(self, dataset):
        """
        从数据集加载转换到缓冲区。
        
        :param dataset: D4RLGameDataset 实例，包含状态和动作序列
        """
        for i in range(len(dataset)):
            state, action_seq, reward, next_state, done = dataset[i]
            # The VQVAE model will only use the first action of the sequence 
            # when converting to macro actions in symmetric_sampling.
            # We still store the full sequence for potential future use.
            self.append(state, action_seq, [reward], next_state, [done])
            
        print(f"Loaded {len(dataset)} transitions from dataset, current buffer size: {len(self.buffer)}")

    def load_from_trajectories(self, trajectories):
        """Loads transitions from a list of trajectory dictionaries."""
        total_transitions = 0
        for trajectory in trajectories:
            # Extract data from the trajectory dictionary
            states = trajectory.get('observations')
            actions = trajectory.get('actions')
            rewards = trajectory.get('rewards')
            next_states = trajectory.get('next_observations')
            # Use 'terminals' key as per gen_offline_data.py
            dones = trajectory.get('terminals') 

            # Basic validation
            if any(x is None for x in [states, actions, rewards, next_states, dones]):
                print("Warning: Skipping trajectory due to missing keys.")
                continue
                
            num_steps = len(states)
            if not (len(actions) == num_steps and len(rewards) == num_steps and 
                      len(next_states) == num_steps and len(dones) == num_steps):
                print(f"Warning: Skipping trajectory due to inconsistent lengths (States: {num_steps}, Actions: {len(actions)}, etc.).")
                continue

            # Iterate through each transition (step) in the trajectory
            for i in range(num_steps):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = bool(dones[i]) # Ensure done is boolean
                
                # Append the transition components
                self.append(state, [action], [reward], next_state, [done]) # Wrap in lists for sample expectation
                total_transitions += 1
                
        print(f"Loaded {total_transitions} transitions from {len(trajectories)} trajectories, current buffer size: {len(self.buffer)}")

    def append(self, *transition):
        """Saves a transition"""
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(np.asarray(x), dtype=torch.float, device=device) for x in zip(*transitions))
    