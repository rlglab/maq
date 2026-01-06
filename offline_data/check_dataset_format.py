import pickle
import os
import argparse
import numpy as np

def load_trajectories(file):
    """Load trajectory data from file."""
    with open(file, 'rb') as f:
        return pickle.load(f)

def check_dataset_legal(dataset_path):
    error_message = ""
    
    if dataset_path == "":
        error_message = "dataset_path is empty"
        return False, error_message
    
    if not os.path.exists(dataset_path):
        error_message = "dataset_path does not exist"
        return False, error_message
    
    # Check file extension
    if not dataset_path.endswith(".pkl"):
        error_message = "dataset_path is not a pkl file"
        return False, error_message
        
    try:
        loaded_trajectories = load_trajectories(dataset_path)
    except Exception as e:
        error_message = f"Failed to load dataset: {str(e)}"
        return False, error_message

    if len(loaded_trajectories) == 0:
        error_message = "dataset is empty"
        return False, error_message
    
    for i, traj in enumerate(loaded_trajectories):
        if 'observations' not in traj or len(traj['observations']) == 0:
            error_message = f"Trajectory {i} does not have 'observations' keys or is empty"
            return False, error_message
        if 'actions' not in traj or len(traj['actions']) == 0:
            error_message = f"Trajectory {i} does not have 'actions' keys or is empty"
            return False, error_message
        if 'rewards' not in traj or len(traj['rewards']) == 0:
            error_message = f"Trajectory {i} does not have 'rewards' keys or is empty"
            return False, error_message
        if 'next_observations' not in traj or len(traj['next_observations']) == 0:
            error_message = f"Trajectory {i} does not have 'next_observations' keys or is empty"
            return False, error_message
        if 'terminals' not in traj or len(traj['terminals']) == 0:
            error_message = f"Trajectory {i} does not have 'terminals' keys or is empty"
            return False, error_message
            
    return True, ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check dataset format.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset .pkl file')
    
    args = parser.parse_args()
    
    legal, error_message = check_dataset_legal(args.dataset_path)
    
    if legal:
        print("is legal")
    else:
        print(f"illegal: {error_message}")
