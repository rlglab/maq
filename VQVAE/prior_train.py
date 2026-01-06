import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import gym
import os
import sys
import pickle
import random
sys.path.append('..')
import json
from collections import deque
import argparse
sys.path.append('../offline_data')

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
    
from vqvae_train import save, load
from modules import VectorQuantizedVAE, to_scalar
    

from dataset import load_data, split_data, D4RLGameDataset

class PriorNet(nn.Module):
    """
    Mapping states to VQVAE embedding index
    """
    def __init__(self, state_dim, hidden_dim=32, output_K=10):
        super().__init__()
        
        self.state_linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),  # size = (B, 32)
        )
        self.classifier = nn.Linear(hidden_dim, output_K)
    
    def forward(self, x):
        x = self.state_linear(x)
        x = self.classifier(x)
        return x


def load_trajectories(file):
    """從檔案加載 trajectory 數據"""
    with open(file, 'rb') as f:
        return pickle.load(f)


class PPONetSimple(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PPONetSimple, self).__init__()
        self.state_linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),  # size = (B, 32)
        )
        self.action_logits = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.state_linear(x)
        x = self.action_logits(x)
        return x

def train(config):
    env = config["env"]
    seed = config["seed"]
    training_dataset = config["training_dataset"]
    testing_dataset = config["testing_dataset"]
    # Load data
    train_trajectories = load_trajectories(f"../offline_data/{training_dataset}")
    test_trajectories = load_trajectories(f"../offline_data/{testing_dataset}")
    observations = train_trajectories[0]['observations']
    actions = train_trajectories[0]['actions']
    
    # Create training and testing datasets
    train_dataset = D4RLGameDataset(train_trajectories, config["sequence_length"])
    test_dataset = D4RLGameDataset(test_trajectories, config["sequence_length"])


    # Create DataLoaders for both training and testing datasets
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,num_workers=0,worker_init_fn=seed_worker,generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,num_workers=0,worker_init_fn=seed_worker,generator=g)
    
    print("---------- Prior (MAQ+IQL) ----------")
    print("Training data size:", len(train_dataset),len(train_trajectories))
    print("Testing data size:", len(test_dataset),len(test_trajectories))
    device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
    
    logdir = config['logdir']
    # Check if logdir exists and contains prior model files
    if os.path.exists(logdir):
        files = os.listdir(logdir)
        model_files = [f for f in files if f.startswith('prior_model_') and f.endswith('.pth')]
        if model_files:
            print(f"Log directory '{logdir}' already contains prior model files. No need to train.")
            return # Exit the train function

    # Ensure logdir exists for SummaryWriter and saving config/models
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # load the VQVAE model
    vqvae_config = json.load(open(os.path.join(config['vqvae_model_path'], 'config.json')))


    # loading the last 
    ls = [filename for filename in os.listdir(config['vqvae_model_path']) if filename.endswith(".pth") ]
    


    # Sort checkpoints by epoch number
    sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[1]))

    # Get the last epoch filename
    last_cp = sorted_checkpoints[-1]

    print("loaded vqvae model",last_cp)

    vqvae_model = VectorQuantizedVAE(state_dim=observations.shape[1], seq_len=config["sequence_length"], K=vqvae_config['k'], dim=vqvae_config['hidden_size'], output_dim=actions.shape[1]).to(device)
    vqvae_model.double()
    load(vqvae_model, os.path.join(config['vqvae_model_path'], last_cp))
    vqvae_model.eval()
    print("VQVAE model loaded.")


    prior_model = PriorNet(state_dim=observations.shape[1], hidden_dim=config['hidden_size'], output_K=vqvae_config['k']).to(device)    
    prior_model.double()
    optimizer = torch.optim.Adam(prior_model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1/(epoch+1))
    writer.add_scalar('lr', scheduler.get_last_lr()[0], 0)
    with open(os.path.join(config['logdir'], 'config.json'), 'w') as f:
        json.dump(config, f)
    
    # forward the VQVAE model to get the embedding index given states, actions
    training_steps = 0
    
    for epoch in range(config['epoch']):
        prior_model.train()
        for i, (state, action) in enumerate(train_dataloader):
            state = state.to(device, dtype=torch.float64)
            action = action.to(device, dtype=torch.float64)

            action = action.view(action.size(0), -1)
            _, _, _, indices = vqvae_model(state, action)
            
            pred = prior_model(state)
            loss = criterion(pred, indices.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_steps += 1
            if i % 100 == 0:
                print(f"epoch: {epoch+1}, iter: {i}, loss: {loss.item()}")
                writer.add_scalar("loss", loss.item(), training_steps)
        
        if (epoch+1) % 5 == 0:
            scheduler.step()
            writer.add_scalar("lr", scheduler.get_last_lr()[0], training_steps)

            prior_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for state, action in test_dataloader:
                    state = state.to(device, dtype=torch.float64)
                    action = action.to(device, dtype=torch.float64)

                    action = action.view(action.size(0), -1)
                    _, _, _, indices = vqvae_model(state, action)
                    pred = prior_model(state).argmax(dim=1)
                    # print(pred) # codebook index
                    correct += (pred == indices).sum().item()
                    total += indices.shape[0]
            acc = correct / total
            print("=====================================")
            print(f"test accuracy: {acc}")
            writer.add_scalar("test_accuracy", acc, training_steps)
            print("=====================================")
            save(prior_model, os.path.join(config['logdir'], f'prior_model_{epoch+1}_{int(acc*100)}.pth'))

def gameplay(config, prior_model_path, env_id, num_episode=1, render=True):
    import moviepy.video.io.ImageSequenceClip
    """
    Play the game with the trained prior model and decoded primitive actions
    """
    observations, actions = load_data(env_id)
    device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
    
    # load the VQVAE model
    vqvae_config = json.load(open(os.path.join(config['vqvae_model_path'], 'config.json')))
    vqvae_model = VectorQuantizedVAE(state_dim=observations.shape[1], seq_len=config["sequence_length"], K=vqvae_config['k'], dim=vqvae_config['hidden_size'], output_dim=actions.shape[1]).to(device)
    load(vqvae_model, os.path.join(config['vqvae_model_path'], config['vqvae_model_name']))
    vqvae_model.eval()
    print("VQVAE model loaded.")
    # load the prior model
    prior_model = PriorNet(state_dim=observations.shape[1], hidden_dim=config['hidden_size'], output_K=vqvae_config['k']).to(device)
    load(prior_model, prior_model_path)
    prior_model.eval()
    print("Prior model loaded.\n")
    # Environment
    env = gym.make(env_id)

    # viewer for Adroit
    viewer = env.mj_viewer_setup()
    obs_list = []

    for i in range(num_episode):
        obs_list.clear()
        state = env.reset()
        # obs_list.append(env.render(mode='rgb_array'))
        obs_list.append(env.viewer._read_pixels_as_in_window())  # for Adroit
        total_reward = 0
        while True:
            state = np.expand_dims(state, axis=0)  # add batch dimension
            pred_k = prior_model(torch.from_numpy(state).to(device, dtype=torch.float64)).argmax(dim=1)
            decoded_actions = vqvae_model.forward_decoder(torch.from_numpy(state).to(device, dtype=torch.float32), pred_k)
            # print(f'Choosing k = {pred_k[0].item()}')
            for primitive_action in decoded_actions[0]:
                state, reward, done, _ = env.step(primitive_action)
                total_reward += reward
                # obs_list.append(env.render(mode='rgb_array'))
                obs_list.append(env.viewer._read_pixels_as_in_window())  # for Adroit
                if done:
                    break
            if done:
                break
            

        print(f"Episode {i+1}, Total reward: {total_reward}")
        if render:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(obs_list, fps=30)
            clip.write_videofile(f'{env_id}-prior-{i+1}-reward{total_reward}.mp4')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='piror train.')


    parser.add_argument('--env', type=str, 
                    help='')
    parser.add_argument('--seqlen', type=int,  default=3,
                    help='')
    parser.add_argument('--suffix', type=str,  default="",
                    help='')
    parser.add_argument('--seed', type=int,  default=None,
                    help='')
    parser.add_argument('--k', type=int, default=16,
                        help='Codebook size (K) used by VQVAE')
    parser.add_argument('--training_dataset', type=str, default="",
                        help='Training dataset, must put your dataset in root/offline_data/')
    parser.add_argument('--testing_dataset', type=str, default="",
                        help='Testing dataset, must put your dataset in root/offline_data/')

    args = parser.parse_args()
    set_seed(args.seed) 
    env = args.env
    seqlen = str(args.seqlen)
    suffix = args.suffix


    config = {
        'gpu': True,
        'batch_size': 32,
        'lr': 1e-3,
        'epoch': 100,
        'sequence_length': args.seqlen,
        'hidden_size': 256,
        'env': args.env,
        'seed': args.seed,
        'training_dataset':args.training_dataset,
        'testing_dataset':args.testing_dataset,
        'vqvae_model_path': f"log/{env}_{suffix}/",
        'logdir': f'log/prior/{env}_{suffix}/',        
    }
    train(config)