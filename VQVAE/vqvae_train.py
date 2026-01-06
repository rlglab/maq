import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import json
import argparse
import pickle
import random
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
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
from dataset import load_data, split_data, D4RLGameDataset
from modules import VectorQuantizedVAE, to_scalar


def load_trajectories(file):
    with open(file, 'rb') as f:
        return pickle.load(f)



def count_labels(data):
    d = {}
    for i in data:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1

    d = dict(sorted(d.items(), key=lambda x:x[1], reverse=True))
    for key, value in d.items():
        print(key, value)
    print("Total: ", len(d))
    print()
    return d

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def train(config):


    env = config["env"]
    seed = config["seed"]
    training_dataset = config["training_dataset"]
    testing_dataset = config["testing_dataset"]
    # Load data
    
    train_trajectories = load_trajectories(f"../offline_data/{training_dataset}")
    test_trajectories = load_trajectories(f"../offline_data/{testing_dataset}")
    
    # Create training and testing datasets
    train_dataset = D4RLGameDataset(train_trajectories, config["sequence_length"])
    test_dataset = D4RLGameDataset(test_trajectories, config["sequence_length"])

    # Create DataLoaders for both training and testing datasets、
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,num_workers=0,worker_init_fn=seed_worker,generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,num_workers=0,worker_init_fn=seed_worker,generator=g)

    print("---------- VQVAE ----------")
    print("Training data size:", len(train_dataset),len(train_trajectories))
    print("Testing data size:", len(test_dataset),len(test_trajectories))

    device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
    
    logdir = config['logdir']
    # Check if logdir exists and contains model files
    if os.path.exists(logdir):
        files = os.listdir(logdir)
        model_files = [f for f in files if f.startswith('model_') and f.endswith('.pth')]
        if model_files:
            print(f"Log directory '{logdir}' already contains model files. No need to train.")
            return # Exit the train function

    # Ensure logdir exists for SummaryWriter and saving config/models
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    observations = train_trajectories[0]['observations']
    actions = train_trajectories[0]['actions']
    
    # print(observations.shape[1])
    # print(actions.shape[1])
    model = VectorQuantizedVAE(state_dim=observations.shape[1], seq_len=config["sequence_length"], K=config['k'], dim=config['hidden_size'], output_dim=actions.shape[1]).to(device)
    model.double()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1/(epoch+1))
    writer.add_scalar('lr', scheduler.get_last_lr()[0], 0)
    with open(os.path.join(config['logdir'], 'config.json'), 'w') as f:
        json.dump(config, f)

    steps = 0
    train_indices_counter = 0
    test_indices_counter = 0
    criterion = torch.nn.MSELoss()
    for epoch in range(config['epoch']):
        model.train()
        ls = []
        for i, (state, action) in enumerate(train_dataloader):
            # print(images[0])
            images = state.to(device, dtype=torch.float64)   # (B, state_dim)
            action = action.to(device, dtype=torch.float64)  # (B, sequence_len, action_dim)

            optimizer.zero_grad()
            
            action = action.view(action.size(0), -1)
            x_tilde, z_e_x, z_q_x, indices = model(images, action)

            # Reconstruction loss
            loss_recons = criterion(x_tilde, action)

            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            
            loss = loss_recons + loss_vq + config['beta'] * loss_commit
            # ls.append(str(loss.item()))
            loss.backward()

            # Logs
            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), steps)
            writer.add_scalar('loss/train/quantization', loss_vq.item(), steps)
            writer.add_scalar('index/train', indices.cpu().detach().numpy()[0], train_indices_counter)

            optimizer.step()
            steps += 1
            train_indices_counter += 1
        print(f'Epoch {epoch+1} finished.')
        # with open("./test_1.txt","a") as f:
        #     f.write("Train"+str(epoch)+"          |"+",".join(ls)+"\n")
        ls = []
        if (epoch+1) % 5 == 0:
            if (epoch+1) % 20 == 0:
                scheduler.step()
                writer.add_scalar('lr', scheduler.get_last_lr()[0], steps)

            model.eval()
            total_count = 0
            total_loss = 0
            index_list = []
            ls = []
            for i, (state, action) in enumerate(test_dataloader):
                images = state.to(device, dtype=torch.float64)
                action = action.to(device, dtype=torch.float64)
                action = action.view(action.size(0), -1)
                x_tilde, z_e_x, z_q_x, indices = model(images, action)
        
                total_count += action.shape[0]
                eval_mse_loss = criterion(x_tilde, action)
                # print("action",actions.shape[1],action.shape,action[0])
                # ls.append(str(eval_mse_loss.item()))

                # print("predict",actions.shape[1],x_tilde.shape,x_tilde[0])
                total_loss += eval_mse_loss.item()

                writer.add_scalar('index/test', indices.cpu().detach().numpy()[0], test_indices_counter)
                index_list.append(indices.cpu().detach().numpy()[0])
                test_indices_counter += 1
                
            print('=========================================')
            print(f'Eval MSE Loss: {total_loss / total_count}')
            save(model, os.path.join(writer.log_dir, f"model_{epoch+1}_{total_loss/total_count}.pth"))
            
            print('\nCodebook usage: ')
            d = count_labels(index_list)
            codebook_usage = np.zeros(config['k'])
            for k, v in d.items():
                codebook_usage[k] = v
            model.reinit_unused_codes(torch.from_numpy(codebook_usage).to(device))
            print('=========================================')
            writer.add_scalar('loss/eval', total_loss / total_count, steps)
            
            # with open("./test_1.txt","a") as f:
            #     f.write("Test"+str(epoch)+"           |"+",".join(ls)+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vqvae train.')


    parser.add_argument('--env', type=str, 
                    help='environment')
    parser.add_argument('--seqlen', type=int,  default=3,
                    help='macro action length')
    parser.add_argument('--suffix', type=str,  default="",
                    help='log\' tag')
    parser.add_argument('--seed', type=int,  default=None,
                    help='seed')
    parser.add_argument('--k', type=int, default=16,
                        help='Codebook size')
    parser.add_argument('--training_dataset', type=str, default="",
                        help='Training dataset, must put your dataset in root/offline_data/')
    parser.add_argument('--testing_dataset', type=str, default="",
                        help='Testing dataset, must put your dataset in root/offline_data/')
    # _rationXXX
    


    args = parser.parse_args()
    set_seed(args.seed) 
    env = args.env
    seqlen = str(args.seqlen)
    suffix = args.suffix
    
    config = {
        'gpu': True,
        'hidden_size': 256,
        'batch_size': 32,
        'k': args.k,
        'lr': 3e-4,
        'beta': 0.25,
        'epoch': 100,
        'env': args.env,
        'sequence_length': args.seqlen,
        'seed':args.seed,
        'training_dataset':args.training_dataset,
        'testing_dataset':args.testing_dataset,
        'logdir': f'log/{env}_{suffix}/',
    }
    train(config)





