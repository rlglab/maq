import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import d4rl
import os
import sys
import json
import argparse
import random
import pickle
import gym
sys.path.append('..')

from VQVAE.dataset import load_data, split_data, D4RLSequenceDataset
g = torch.Generator()
def set_seed(seed=None):
    global g
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)



def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def load_offline_data_by_reward_rank(env, threshold=1e-6):
    dataset = d4rl.qlearning_dataset(gym.make(env))

    """生成 dones_float 來判斷終止"""
    dones_float = np.zeros_like(dataset['rewards'])
    for i in range(len(dones_float) - 1):
        if (np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > threshold or 
            dataset['terminals'][i] == 1.0):
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1  # 確保最後一個步驟為終止

    """分割數據為多個 trajectory"""
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']

    trajectories = []
    start_idx = 0

    for i in range(len(dones_float)):
        if dones_float[i] == 1:  # 如果該步為 trajectory 的終止
            trajectory = {
                'observations': observations[start_idx:i + 1],
                'actions': actions[start_idx:i + 1],
                'rewards': rewards[start_idx:i + 1],
                'next_observations': next_observations[start_idx:i + 1],
                'terminals': dones_float[start_idx:i + 1],
            }
            # Calculate cumulative reward for this trajectory
            trajectory['cumulative_reward'] = np.sum(trajectory['rewards'])
            trajectories.append(trajectory)
            start_idx = i + 1

    # Sort trajectories by cumulative reward in descending order (highest reward first)
    trajectories_sorted = sorted(trajectories, key=lambda x: x['cumulative_reward'])
    
    return trajectories_sorted

def load_offline_data(env, threshold=1e-6):
    dataset = d4rl.qlearning_dataset(gym.make(env))

    """生成 dones_float 來判斷終止"""
    dones_float = np.zeros_like(dataset['rewards'])
    for i in range(len(dones_float) - 1):
        if (np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > threshold or 
            dataset['terminals'][i] == 1.0):
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1  # 確保最後一個步驟為終止

    """分割數據為多個 trajectory"""
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']

    trajectories = []
    start_idx = 0

    for i in range(len(dones_float)):
        if dones_float[i] == 1:  # 如果該步為 trajectory 的終止
            trajectory = {
                'observations': observations[start_idx:i + 1],
                'actions': actions[start_idx:i + 1],
                'rewards': rewards[start_idx:i + 1],
                'next_observations': next_observations[start_idx:i + 1],
                'terminals': dones_float[start_idx:i + 1],
            }
            trajectories.append(trajectory)
            start_idx = i + 1

    return trajectories

def save_trajectories(train_trajectories, test_trajectories, train_file="train.pkl", test_file="test.pkl"):
    """保存訓練集與測試集至檔案"""
    with open(train_file, 'wb') as f:
        pickle.dump(train_trajectories, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test_trajectories, f)
    print(f"已保存訓練集至 {train_file}，測試集至 {test_file}.")
    
def split_first_percent_data(trajectories, percent=50):
    # for each trajectory, split the data into first percent and last percent
    train_trajectories = []
    test_trajectories = []
    for trajectory in trajectories:
        num_train = int(len(trajectory['observations']) * percent / 100)
        print(num_train,len(trajectory['observations']))
        train_trajectory = {}
        train_trajectory['observations'] = trajectory['observations'][:num_train]
        train_trajectory['actions'] = trajectory['actions'][:num_train]
        train_trajectory['rewards'] = trajectory['rewards'][:num_train]
        train_trajectory['next_observations'] = trajectory['next_observations'][:num_train]
        train_trajectory['terminals'] = trajectory['terminals'][:num_train]
        test_trajectory = {}
        test_trajectory['observations'] = trajectory['observations'][num_train:]
        test_trajectory['actions'] = trajectory['actions'][num_train:]
        test_trajectory['rewards'] = trajectory['rewards'][num_train:]
        test_trajectory['next_observations'] = trajectory['next_observations'][num_train:]
        test_trajectory['terminals'] = trajectory['terminals'][num_train:]
        train_trajectories.append(train_trajectory)
        test_trajectories.append(test_trajectory)
    return train_trajectories, test_trajectories

def split_train_test(trajectories, train_ratio=0.8, ranked=False):
    """隨機打亂並分割 trajectory 為訓練集和測試集"""
    if ranked:

        num_train = int(len(trajectories) * train_ratio)
        train_trajectories = trajectories[:num_train]
        test_trajectories = trajectories[num_train:]
    else:
        np.random.shuffle(trajectories)  # 隨機打亂
        num_train = int(len(trajectories) * train_ratio)
        train_trajectories = trajectories[:num_train]
        test_trajectories = trajectories[num_train:]
    # print(len(train_trajectories))
    # print(len(test_trajectories))
    return train_trajectories, test_trajectories

def load_trajectories(file):
    """從檔案加載 trajectory 數據"""
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def compare_trajectories(traj1, traj2):
    """比較兩個 trajectory 清單是否相同"""
    if len(traj1) != len(traj2):
        return False

    for t1, t2 in zip(traj1, traj2):
        for key in t1:
            if not np.array_equal(t1[key], t2[key]):  # 使用 np.array_equal 比較陣列是否相同
                return False
    return True

def try_offline_data_with_seed(config,seed):
    trajectories = load_offline_data(config["env_name"])
    total_length = 0
    total_reward = 0
    for i in range(len(trajectories)):
        
        r = 0
        for reward in trajectories[i]['rewards']:
            r += reward
        total_length += len(trajectories[i]['rewards'])
        total_reward += r
        print("total reward ",r,"length ",len(trajectories[i]['rewards']))
    print("total length ",total_length, "mean length ",total_length/len(trajectories), "demonstrations ",len(trajectories))
    print("total reward ",total_reward, "mean reward ",total_reward/len(trajectories), "demonstrations ",len(trajectories))

def load_offline_data_with_seed(config,seed,ranked=False):
    loaded_train_trajectories = load_trajectories(f"../offline_data/{env}_train_seed{seed}.pkl")
    loaded_test_trajectories = load_trajectories(f"../offline_data/{env}_test_seed{seed}.pkl")
    # print(len(loaded_train_trajectories))
    # print(len(loaded_test_trajectories))
    tot = 0
    for i in range(len(loaded_train_trajectories)):
        
        tot += len(loaded_train_trajectories[i]['rewards'])
        r = 0
        for reward in loaded_train_trajectories[i]['rewards']:
            r += reward
        print("total reward ",r,"length ",len(loaded_train_trajectories[i]['rewards']))
    for i in range(len(loaded_test_trajectories)):
        print(len(loaded_test_trajectories[i]['rewards']))
        tot += len(loaded_test_trajectories[i]['rewards'])
        r = 0
        for reward in loaded_test_trajectories[i]['rewards']:
            r += reward
    print("total length ",tot)
    
def gen_first_percent_data(config,seed,percent=50):
    env = config["env_name"]
    

    trajectories = load_offline_data(config["env_name"])
        
    print(">>>",len(trajectories))
    print(">>>",len(trajectories[0]['rewards']))
    
    extra_tag = "_percent" + str(percent)
    
    train_trajectories, test_trajectories = split_first_percent_data(trajectories, percent=percent)


    print(len(train_trajectories))
    print(len(test_trajectories))
    
    # return 
    # return
    
    save_trajectories(train_trajectories, test_trajectories,train_file=f"{env}_train_seed{seed}{extra_tag}.pkl",test_file=f"{env}_test_seed{seed}{extra_tag}.pkl")
    
    loaded_train_trajectories = load_trajectories(f"{env}_train_seed{seed}{extra_tag}.pkl")
    loaded_test_trajectories = load_trajectories(f"{env}_test_seed{seed}{extra_tag}.pkl")
    print(len(loaded_train_trajectories))
    print(len(loaded_test_trajectories))



def gen_offline_data_with_seed(config,seed,train_ratio=0.9, ranked=False):
    env = config["env_name"]
    
    training_dataset_name = config['training_dataset_name']
    testing_dataset_name = config['testing_dataset_name']
    set_seed(seed)
    if ranked:
        trajectories = load_offline_data_by_reward_rank(config["env_name"])
        
        # Extract cumulative rewards for statistical analysis
        rewards = [traj['cumulative_reward'] for traj in trajectories]
        
        # Calculate quartiles
        q25 = np.percentile(rewards, 25)
        q50 = np.percentile(rewards, 50)  # median
        q75 = np.percentile(rewards, 75)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        mean_reward = np.mean(rewards)
        
        # print(f"=== Reward Statistics (Sorted by Reward) ===")
        # print(f"Total trajectories: {len(trajectories)}")
        # print(f"Min reward: {min_reward:.2f}")
        # print(f"25th percentile (Q1): {q25:.2f}")
        # print(f"50th percentile (Median): {q50:.2f}")
        # print(f"75th percentile (Q3): {q75:.2f}")
        # print(f"Max reward: {max_reward:.2f}")
        # print(f"Mean reward: {mean_reward:.2f}")
        # print(f"Standard deviation: {np.std(rewards):.2f}")
        # print("=" * 45)
        
    else:
        trajectories = load_offline_data(config["env_name"])
    # print("saved to ",training_dataset_name,testing_dataset_name)
    # print(">>>",len(trajectories))
    # print(">>>",len(trajectories[0]['rewards']))
    
    
    train_trajectories, test_trajectories = split_train_test(trajectories, train_ratio=train_ratio, ranked=ranked)
    # max_reward = max([traj['cumulative_reward'] for traj in train_trajectories])
    # min_reward = min([traj['cumulative_reward'] for traj in train_trajectories])
    # mean_reward = np.mean([traj['cumulative_reward'] for traj in train_trajectories])
    # print("train ratio",train_ratio)
    # print("length",len(train_trajectories))
    # print(f"Max reward: {max_reward:.2f}")
    # print(f"Min reward: {min_reward:.2f}")
    # print(f"Mean reward: {mean_reward:.2f}")
    
    print("Training Dataset:",len(train_trajectories))
    print("Testing Dataset:",len(test_trajectories))
    
    # return 
    # return
    
    save_trajectories(train_trajectories, 
            test_trajectories,
            train_file=f"{training_dataset_name}",
            test_file=f"{testing_dataset_name}")
    
    # train_equal = compare_trajectories(loaded_train_trajectories_tmp, train_trajectories)
    # test_equal = compare_trajectories(loaded_test_trajectories_tmp, test_trajectories)
    # print("訓練資料相同:", train_equal)
    # print("測試資料相同:", test_equal)
    
    loaded_train_trajectories = load_trajectories(f"{training_dataset_name}")
    loaded_test_trajectories = load_trajectories(f"{testing_dataset_name}")
    # print(len(loaded_train_trajectories))
    # print(len(loaded_test_trajectories))
    legal, error_message = check_dataset_legal(config) 
    if legal:
        print("Dataset check passed after generation.")
    if not legal:
        print("Dataset check failed after generation:", error_message)


def check_dataset_legal(config):
    training_source = config["training_dataset_name"]
    testing_source = config["testing_dataset_name"]
    error_message = ""
    
    if training_source == "" or testing_source == "":
        error_message = "training_source or testing_source is empty"
        return False, error_message
    
    if not os.path.exists(f"{training_source}") or not os.path.exists(f"{testing_source}"):
        error_message = "training_source or testing_source is not exist"
        return False, error_message
    
    # dataset's format
    
    for file in [training_source, testing_source]:
        if not file.endswith(".pkl"):
            error_message = "training_source or testing_source is not a pkl file"
            return False, error_message
        
        loaded_trajectories = load_trajectories(file)
        if len(loaded_trajectories) == 0:
            error_message = "training_source or testing_source is empty"
            return False, error_message
        
        for traj in loaded_trajectories:
            
            if 'observations' not in traj or len(traj['observations']) == 0:
                error_message = f"{file} dont have 'observations' keys"
                return False, error_message
            if 'actions' not in traj or len(traj['actions']) == 0:
                error_message = f"{file} dont have 'actions' keys"
                return False, error_message
            if 'rewards' not in traj or len(traj['rewards']) == 0:
                error_message = f"{file} dont have 'rewards' keys"
                return False, error_message
            if 'next_observations' not in traj or len(traj['next_observations']) == 0:
                error_message = f"{file} dont have 'next_observations' keys"
                return False, error_message
            if 'terminals' not in traj or len(traj['terminals']) == 0:
                error_message = f"{file} dont have 'terminals' keys"
                return False, error_message
            
    
    return True, error_message
    
    
    
    
    
    return True
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vqvae train.')


    parser.add_argument('--env', type=str, 
                    help='')
    parser.add_argument('--seed', type=int,  default=1,
                    help='')
    parser.add_argument('--train_ratio', type=float,  default=0.9,
                    help='')
    parser.add_argument('--ranked', action='store_true',
                    help='')
    parser.add_argument('--training_dataset_name', type=str,  default="",
                    help='')
    parser.add_argument('--testing_dataset_name', type=str,  default="",
                    help='')
    parser.add_argument('--default_generate', action='store_true',
                    help='')
    
    
    
    
    
    args = parser.parse_args()
    seed = args.seed
    env = args.env
    train_ratio = args.train_ratio
    ranked = args.ranked
    training_dataset_name = args.training_dataset_name
    testing_dataset_name = args.testing_dataset_name
    config = {
        'training_dataset_name': training_dataset_name,
        'testing_dataset_name': testing_dataset_name,
        'env_name': f'{env}'
    }
    legal, error_message = check_dataset_legal(config) 
    if not args.default_generate:
        # check dataset
        
        if (not legal):
            # generate error.txt
            
            with open('error.txt','w') as f:
                f.write(error_message)
            exit(0)
        exit(0)
    if legal:
        print("Dataset check passed, passing generation.")
        exit(0)
    gen_offline_data_with_seed(config,seed,train_ratio,ranked=ranked) 
    
    # for seed in seeds:
    #     # try_offline_data_with_seed(config,seed)
    #     # load_offline_data_with_seed(config,seed,ranked=False)
    #     # for r in ratio:
    #     # gen_offline_data_with_seed(config,seed,0.05, ranked=True) # only 1 success
    #     gen_first_percent_data(config,seed) 
            





