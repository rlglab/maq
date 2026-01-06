import json
from DSACMAQ import MAQDSACAgent
from RLPDMAQ import MAQDRLPDSACAgent
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import os
import math
import random
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use only the first GPU
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
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MAQ agents.')
    parser.add_argument('--env', type=str, required=True,
                    help='Environment name')
    parser.add_argument('--seqlen', type=int, default=3,
                    help='Sequence length')
    parser.add_argument('--seed', type=int, required= True,
                    help='Random seed')
    parser.add_argument('--type', type=str, choices=['DSAC', 'RLPD'], required=True,
                    help='Type of agent to train (DSAC or RLPD)')
    parser.add_argument('--suffix', type=str, help='Additional suffix for experiment', default='')
    parser.add_argument('--k', type=int, default=16)
    
    parser.add_argument('--training_dataset', type=str, default="",
                        help='Training dataset, must put your dataset in root/offline_data/')
    parser.add_argument('--testing_dataset', type=str, default="",
                        help='Testing dataset, must put your dataset in root/offline_data/')
    
    args = parser.parse_args()
    env = args.env
    seqlen = str(args.seqlen)
    seed = str(args.seed)
    set_seed(args.seed)
    suffix = args.suffix
    K = args.k
    print(f"K: {K}")

    # Common configuration parameters
    num_envs = 8
    base_config = {
        "gpu": True,
        "training_steps": 1e6,
        "warmup_steps": 1000 * num_envs,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "eval_interval": 10000,
        "num_envs": num_envs,
        "eval_episode": 10,
        "env_id": env,
        "seed": args.seed,
        'training_dataset':args.training_dataset,
        'testing_dataset':args.testing_dataset,
        "vqvae_model_path": f"../VQVAE/log/{env}_{suffix}/",
        # SAC specific parameters
        "update_freq": num_envs ,  # Reduce update frequency when using multiple envs
        "hidden_dim": 256,
        "actor_lr": 3e-4 ,  # Scale down learning rates with num_envs
        "critic_lr": 1e-3,
        "alpha_lr": 3e-4,
        "target_entropy": -math.log(1/K),  # Manual setting: negative values (-0.5 to -2.0 typical range)
        "tau": 0.005,
        "gamma": 0.99,
        "replay_buffer_capacity": 10000,  # Scale buffer with num_envs
        "deterministic_eval": True,
        "auto_alpha": True,
        "initial_alpha": 0.2,
    }

    # Set log directory based on agent type
    if args.type == 'DSAC':
        base_config["logdir"] = f'log/exp_{suffix}/{env}/DSACMAQ/'
        agent_class = MAQDSACAgent
    else:  # RLPD
        base_config["logdir"] = f'log/exp_{suffix}/{env}/RLPDMAQ/'
        agent_class = MAQDRLPDSACAgent

    # Create directories
    os.makedirs(base_config["logdir"], exist_ok=True)
    
    # Save config
    with open(os.path.join(base_config["logdir"], 'config.json'), 'w') as f:
        json.dump(base_config, f, indent=4)
    
    print(f"Starting training for {args.type} agent...")
    agent = agent_class(base_config)
    agent.train()
    
    print("\nTraining completed.")

	