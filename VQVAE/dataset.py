import gym
import d4rl
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

        for traj in trajectories:
            self.observations.append(traj['observations'])
            self.actions.append(traj['actions'])

        # 合併所有遊戲的數據
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        # 計算有效的數據範圍
        self.valid_indices = len(self.observations) - self.sequence_length + 1

    def __len__(self):
        """
        返回數據集的長度。
        """
        return self.valid_indices

    def __getitem__(self, idx):
        """
        返回第 idx 個數據點，包含觀測序列和動作序列。

        :param idx: 數據點的索引
        :return: Tuple(obs_seq, acts_seq)
        """
        obs_seq = self.observations[idx]
        acts_seq = self.actions[idx:idx + self.sequence_length]
        return obs_seq, acts_seq
    
class D4RLSequenceDataset(Dataset):
    def __init__(self, observations, actions, indices, sequence_length=4):
        self.observations = observations
        self.actions = actions
        self.indices = indices
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        obs_seq = self.observations[index]
        acts_seq = self.actions[index:index + self.sequence_length]
        return obs_seq, acts_seq

def load_data(env_name,seed):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    data = env.get_dataset()
    observations = data['observations']
    actions = data['actions']
    print(actions.shape)
    return observations, actions

def split_data(observations, sequence_length=4, test_size=0.1):
    max_index = observations.shape[0] - sequence_length + 1
    indices = np.arange(max_index)
    np.random.shuffle(indices)
    split_point = int(np.floor(test_size * len(indices)))
    test_indices = indices[:split_point]
    train_indices = indices[split_point:]
    
    print(len(train_indices))
    print(len(test_indices))
    
    return train_indices, test_indices

if __name__ == "__main__":
    # Parameters
    ENV_NAME = 'pen-human'
    SEQUENCE_LENGTH = 2
    TEST_SIZE = 0.1
    BATCH_SIZE = 32

    # Load data
    observations, actions = load_data(ENV_NAME)
    print(observations.shape)
    print(actions.shape)
    
    # Split indices into training and testing sets
    train_indices, test_indices = split_data(observations, SEQUENCE_LENGTH, TEST_SIZE)

    # Create training and testing datasets
    train_dataset = D4RLSequenceDataset(observations, actions, train_indices, SEQUENCE_LENGTH)
    test_dataset = D4RLSequenceDataset(observations, actions, test_indices, SEQUENCE_LENGTH)

    # Create DataLoaders for both training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Example of how to iterate over the DataLoader
    print("Training data size:", len(train_dataset))
    for obs, acts in train_loader:
        print("Training batch - Observations:", obs.shape, "Actions:", acts.shape)
        break

    print("Testing data size:", len(test_dataset))
    for obs, acts in test_loader:
        print("Testing batch - Observations:", obs.shape, "Actions:", acts.shape)
        break
