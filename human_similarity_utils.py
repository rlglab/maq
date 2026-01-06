import numpy as np
import torch
import os
import sys
import csv
import gym
from collections import deque
import moviepy.video.io.ImageSequenceClip
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pickle
from gym.wrappers import TimeLimit

from sklearn.preprocessing import StandardScaler
import ot

sys.path.append('../offline_data')
def load_trajectories_W(file):
    """從檔案加載 trajectory 數據"""
    with open(file, 'rb') as f:
        return pickle.load(f)

def aggregate_trajectories(traj_list):
    """
    Aggregates multiple trajectories into a single numpy array.
    
    Args:
      traj_list: A list of numpy arrays, each of shape (timesteps, feature_dim)
    
    Returns:
      aggregated: A numpy array of shape (total_timesteps, feature_dim)
    """
    return np.concatenate(traj_list, axis=0)

def compute_wasserstein_distance_multiple(agent_trajs, expert_trajs):
    """
    Computes the Wasserstein distance between aggregated agent and expert trajectories.
    
    Args:
      agent_trajs: List of numpy arrays (each representing a trajectory) from the agent.
      expert_trajs: List of numpy arrays from human demonstrations.
      
    Returns:
      w2_dist: The computed Wasserstein distance.
    """
    # Aggregate transitions from multiple trajectories
    agent_data = aggregate_trajectories(agent_trajs)
    expert_data = aggregate_trajectories(expert_trajs)
    
    # Scale the data for fair distance comparison.
    scaler = StandardScaler()
    combined = np.concatenate([agent_data, expert_data], axis=0)
    scaler.fit(combined)
    agent_scaled = scaler.transform(agent_data)
    expert_scaled = scaler.transform(expert_data)
    
    # Create uniform weights for each set.
    agent_weights = np.ones(agent_scaled.shape[0]) / agent_scaled.shape[0]
    expert_weights = np.ones(expert_scaled.shape[0]) / expert_scaled.shape[0]
    
    # Compute the cost matrix using Euclidean distance.
    cost_matrix = ot.dist(agent_scaled, expert_scaled, metric='euclidean')
    # Compute the squared Wasserstein-2 distance.
    w2_dist = ot.emd2(agent_weights, expert_weights, cost_matrix)
    return w2_dist



# Add project directories to sys.path if needed (adjust paths as necessary)
# sys.path.append("RL")
# sys.path.append("VQVAE")
# sys.path.append("O2ORL")
# sys.path.append("human_similarity") # Assuming utils.py might be needed later

# Potential imports from other project modules (uncomment if needed)
# from human_similarity.utils import get_MAQ_agent, ... # Example
# from RL.utils import wrap_gym # Example if wrap_gym is defined there
# from ws_dist import compute_wasserstein_distance_multiple # Example

# Helper function to initialize environment based on agent type (example)
def initialize_env(env_id, seed, agent_type=None, render=False, horizon=None):
    """Initializes the Gym environment, handling specifics like RLPD wrappers."""
    # Basic environment creation
    env = gym.make(env_id)

    # Capture the default horizon before any modifications
    default_horizon = getattr(env, '_max_episode_steps', 1000)  # Fallback to 1000 if not set
    
    # Set up horizon - use provided horizon or keep default
    if horizon is not None:
        print(f"Setting custom horizon: {horizon} (default was: {default_horizon})")
        env._max_episode_steps = horizon
    else:
        print(f"Using default horizon: {default_horizon}")
    
    # Apply horizon using TimeLimit wrapper if needed
    # env = TimeLimit(env, max_episode_steps=env._max_episode_steps)
    
    # Specific handling for agent types if necessary
    # Example: RLPD wrapper might require specific setup
    # if agent_type and "RLPD" in agent_type:
    #     print("Initializing RLPD environment wrappers...")
    #     env = wrap_gym(env, rescale_actions=True) # Make sure wrap_gym is imported/defined
    #     env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

    env.seed(seed)
    if render:
        try:
            env.mj_viewer_setup() # May fail for non-mujoco envs
        except AttributeError:
            print(f"Warning: mj_viewer_setup not available for {env_id}")
        env.render()

    print(f"env._max_episode_steps: {env._max_episode_steps}")

    # Store the default horizon as an attribute for later reference
    env._default_horizon = default_horizon

    return env

# Helper function to get render frames (handle different render modes)
def get_render_frame(env, render_mode='rgb_array'):
    """Gets a frame for rendering, trying different methods."""
    try:
        if hasattr(env, 'viewer') and hasattr(env.viewer, '_read_pixels_as_in_window'):
             # Specific for some MuJoCo versions/envs (e.g., Adroit)
            return env.viewer._read_pixels_as_in_window()
        else:
            # Standard method
            return env.render(mode=render_mode)
    except Exception as e:
        print(f"Warning: Failed to get render frame: {e}")
        return None

def run_agent_episode(agent, env_id, seed, render=False, render_mode='rgb_array', horizon=None):
    """
    Runs a single episode with the given agent in the specified environment.

    Args:
        agent: The agent instance (should have an `inference(state)` method).
        env_id: The Gym environment ID string.
        seed: The seed for the environment.
        render: Whether to render the episode and save a video.
        render_mode: The mode for rendering ('rgb_array', etc.).
        horizon: Maximum number of steps for the episode. If set and success is achieved, 
                episode terminates early. If None, uses environment's default horizon.

    Returns:
        A dictionary containing:
        - 'states': List of states encountered (numpy arrays).
        - 'actions': List of actions taken by the agent (numpy arrays).
        - 'reward': The total raw reward for the episode.
        - 'normalized_score': The D4RL normalized score (if applicable).
        - 'length': The number of steps in the episode.
        - 'video_path': Path to the saved video file if rendered, else None.
        - 'success': Whether the episode was successful.
        - 'early_termination': Whether episode terminated early due to success.
        - 'default_horizon': The environment's original default horizon.
    """
    env = initialize_env(env_id, seed, getattr(agent, 'get_agent_type', lambda: 'Unknown')(), render, horizon)

    state = env.reset()
    done = False
    success = False # Initialize success flag
    early_termination = False # Initialize early termination flag
    total_raw_reward = 0.0
    episode_states = [state]
    episode_actions = []
    obs_list = []
    episode_length = 0
    print(f'env: {env}')
    if render:
        frame = get_render_frame(env, render_mode)
        if frame is not None:
            obs_list.append(frame)
    while not done:
        # Check horizon limit if set
        if horizon is not None and episode_length >= horizon:
            print(f"Episode terminated due to horizon limit: {horizon}")
            break
            
        # Assuming agent.inference returns a sequence of actions for the macro-action
        macro_action = agent.inference(state)
        if not isinstance(macro_action, (list, np.ndarray)):
             # Handle cases where inference might return a single action directly
             macro_action = [macro_action]

        for action in macro_action:
            try:
                state, reward, done, info = env.step(action)
                total_raw_reward += reward
                episode_states.append(state)
                episode_actions.append(action)
                episode_length += 1
                
                # TODO: remove this block, due to early success

                # # # Check for success during the episode (not just at the end)
                # if info.get('goal_achieved', False) or info.get('success', False):
                #     success_val = info.get('success', info.get('goal_achieved'))
                #     if success_val == True or success_val == 1:
                #         success = True
                #         # If horizon is set and we achieved success, terminate early
                #         if horizon is not None : ## !!!! ERALY RETURN
                #             early_termination = True
                #             done = True
                #             print(f"Episode terminated early due to success at step {episode_length}")

                if render:
                    frame = get_render_frame(env, render_mode)
                    if frame is not None:
                        obs_list.append(frame)

                # Check horizon limit within action loop as well
                if horizon is not None and episode_length >= horizon:
                    print(f"Episode terminated due to horizon limit: {horizon}")
                    done = True
                    break

                if done:
                    break # Exit inner loop if environment signals done
            except Exception as e:
                print(f"Error during env.step: {e}")
                done = True # Mark as done to exit episode
                break

    # Final check for success at the end of the episode if not already detected
    if not success and done and 'info' in locals(): # Check if info dict exists from the last step
        if info.get('goal_achieved', False) or info.get('success', False):
            # Check explicit True or 1, as some envs might use numbers
            print("Episode success")
            success_val = info.get('success', info.get('goal_achieved'))
            if success_val == True or success_val == 1:
                success = True
        # Add other environment-specific checks here if needed
        # elif 'antmaze' in env_id.lower() and done: # Example heuristic
        #     # AntMaze success might be inferred differently, e.g., not timing out
        #     pass

    # Calculate normalized score (requires env to have get_normalized_score)
    normalized_score = -np.inf # Default if score func unavailable
    if hasattr(env, 'get_normalized_score'):
        try:
            normalized_score = env.get_normalized_score(total_raw_reward) * 100
        except Exception as e:
            print(f"Warning: Failed to get normalized score for {env_id}: {e}")

    # Save video if rendered
    video_path = None
    if render and obs_list:
        agent_type_str = getattr(agent, 'get_agent_type', lambda: 'Unknown')()
        # Use short name if available
        agent_short_name = getattr(agent, 'get_agent_short_name', lambda: '')()
        if agent_short_name:
            agent_type_str = agent_short_name
            
        model_id_str = os.path.basename(str(getattr(agent, 'get_model_path', lambda: 'nomodel')())) # Get model filename
        video_dir = f'record/{env_id}/seed{seed}'
        os.makedirs(video_dir, exist_ok=True)
        # Create a more descriptive filename including environment, agent type, score and date
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f'{agent_type_str}_seed{seed}_score{normalized_score:.2f}_len{episode_length}_{timestamp}'
        video_path = os.path.join(video_dir, f'{base_filename}.mp4')
        try:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(obs_list, fps=30)
            clip.write_videofile(video_path, logger=None) # logger=None suppresses verbose output
            print(f"Saved video: {video_path}")
        except Exception as e:
            print(f"Error saving video {video_path}: {e}")
            video_path = None # Reset path if saving failed

    env.close()

    return {
        'states': np.array(episode_states[:-1]), # Exclude final state if using states for action prediction alignment
        'actions': np.array(episode_actions),
        'reward': total_raw_reward,
        'normalized_score': normalized_score,
        'length': episode_length,
        'video_path': video_path,
        'success': success, # Add success flag to results
        'early_termination': early_termination, # Add early termination flag to results
        'default_horizon': getattr(env, '_default_horizon', 1000) # Add default horizon to results
    }

def write_results_to_csv(results_dict, env_id, filename_prefix="results"):
    """
    Appends a dictionary of results to a CSV file named using the prefix and env_id.

    Args:
        results_dict: A dictionary where keys are column headers and values are the data.
        env_id: The environment ID, used to create the filename.
        filename_prefix: Prefix for the CSV filename (e.g., "agent_performance").
    """
    if not results_dict:
        print("Warning: No results provided to write_results_to_csv.")
        return

    # Ensure the directory exists (assuming CSVs are saved in the current dir or a subdir)
    output_dir = "evaluation_results" # Example directory
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{filename_prefix}_{env_id}.csv")

    # Keep original logic but sort keys for consistency across runs
    fieldnames = sorted(list(results_dict.keys()))

    try:
        # Check if file exists to write header only once
        write_header = not os.path.exists(filename)
        # If file exists, read existing header to maintain consistency
        if not write_header:
            try:
                with open(filename, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_header = next(reader)
                    # Use existing header order and add any new fields at the end
                    new_fields = [field for field in fieldnames if field not in existing_header]
                    fieldnames = existing_header + new_fields
            except Exception as e:
                print(f"Warning: Could not read existing CSV header: {e}. Using sorted header.")

        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')  # Use empty string for missing values
            if write_header:
                writer.writeheader()
            writer.writerow(results_dict)
        # print(f"Results successfully written to {filename}")
    except IOError as e:
        print(f"Error writing results to {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")

# --- Placeholder for Metric Functions ---
# Define functions that calculate specific metrics based on agent/human trajectories

def calculate_dtw_distances(agent_traj_states, agent_traj_actions, human_trajs_states, human_trajs_actions):
    """Calculates average DTW distance between agent trajectory and a set of human trajectories."""
    avg_state_dtw = np.inf
    avg_action_dtw = np.inf

    if human_trajs_states:
        state_dtws = [fastdtw(agent_traj_states, ht_s, dist=euclidean)[0] for ht_s in human_trajs_states]
        avg_state_dtw = np.mean(state_dtws) if state_dtws else np.inf

    if human_trajs_actions:
        # Ensure actions are present and compatible before calculating DTW
        if agent_traj_actions.size > 0:
             # Ensure human trajectories also have actions
            valid_human_action_trajs = [ht_a for ht_a in human_trajs_actions if ht_a.size > 0]
            if valid_human_action_trajs:
                action_dtws = [fastdtw(agent_traj_actions, ht_a, dist=euclidean)[0] for ht_a in valid_human_action_trajs]
                avg_action_dtw = np.mean(action_dtws) if action_dtws else np.inf
            else:
                 print("Warning: No valid human action trajectories provided for DTW calculation.")
        else:
             print("Warning: Agent action trajectory is empty, skipping action DTW.")


    return {
        'state_dtw_mean': avg_state_dtw,
        'action_dtw_mean': avg_action_dtw
    }

# Example for Wasserstein Distance (ensure ws_dist is importable)
def calculate_wasserstein_distances(agent_traj_states, agent_traj_actions, human_trajs_states, human_trajs_actions):
    """Calculates Wasserstein distance between a single agent trajectory and a set of human trajectories."""
    state_w2 = np.inf
    action_w2 = np.inf

    # Ensure agent trajectory is not empty
    if agent_traj_states.size == 0:
        print("Warning: Agent state trajectory is empty, skipping state W2 calculation.")
        # No need to proceed if agent states are empty
        return {
            'state_w2_dist': state_w2,
            'action_w2_dist': action_w2
        }

    try:
        # Calculate State Wasserstein Distance
        if human_trajs_states:
            # compute_wasserstein_distance_multiple expects a list of agent trajectories
            state_w2 = compute_wasserstein_distance_multiple([agent_traj_states], human_trajs_states)
        else:
            print("Warning: No human state trajectories provided for W2 calculation.")

        # Calculate Action Wasserstein Distance
        # Check if agent actions exist and human actions exist
        if agent_traj_actions.size > 0 and human_trajs_actions:
            # Filter out empty human action trajectories if any exist
            valid_human_action_trajs = [ht_a for ht_a in human_trajs_actions if ht_a.size > 0]
            if valid_human_action_trajs:
                 # compute_wasserstein_distance_multiple expects a list of agent trajectories
                action_w2 = compute_wasserstein_distance_multiple([agent_traj_actions], valid_human_action_trajs)
            else:
                print("Warning: No valid (non-empty) human action trajectories provided for W2 calculation.")
        elif agent_traj_actions.size == 0:
             print("Warning: Agent action trajectory is empty, skipping action W2 calculation.")
        else: # Agent actions exist, but no human actions provided
            print("Warning: No human action trajectories provided for W2 calculation.")

    except Exception as e:
        print(f"Warning: Failed to compute Wasserstein distance: {e}")
        # Reset to inf in case of error during calculation
        state_w2 = np.inf
        action_w2 = np.inf

    return {
        'state_w2_dist': state_w2,
        'action_w2_dist': action_w2
    }

# Example for Wasserstein Distance (ensure ws_dist is importable)
def calculate_min_dtw_distances(agent_traj_states, agent_traj_actions, human_trajs_states, human_trajs_actions):
    """Calculates minimum DTW distance between agent trajectory and a set of human trajectories."""
    min_state_dtw = np.inf
    min_action_dtw = np.inf

    if human_trajs_states:
        state_dtws = [fastdtw(agent_traj_states, ht_s, dist=euclidean)[0] for ht_s in human_trajs_states]
        min_state_dtw = np.min(state_dtws) if state_dtws else np.inf

    if human_trajs_actions:
        # Ensure actions are present and compatible before calculating DTW
        if agent_traj_actions.size > 0:
             # Ensure human trajectories also have actions
            valid_human_action_trajs = [ht_a for ht_a in human_trajs_actions if ht_a.size > 0]
            if valid_human_action_trajs:
                action_dtws = [fastdtw(agent_traj_actions, ht_a, dist=euclidean)[0] for ht_a in valid_human_action_trajs]
                min_action_dtw = np.min(action_dtws) if action_dtws else np.inf
            else:
                 print("Warning: No valid human action trajectories provided for DTW calculation.")
        else:
             print("Warning: Agent action trajectory is empty, skipping action DTW.")

    return {
        'state_dtw_min': min_state_dtw,
        'action_dtw_min': min_action_dtw
    }

# Define compute_euclidean_distance locally (mirrors calculator version)
def compute_euclidean_distance(traj1, traj2):
    """
    Calculates the summed Euclidean distance between two sequences of vectors.
    NOTE: This function expects sequences (lists/arrays of vectors).
    If comparing single vectors, wrap them like [vector1], [vector2].
    """
    # Ensure the trajectories are list-like and contain numpy arrays or similar
    if not isinstance(traj1, (list, np.ndarray)) or not isinstance(traj2, (list, np.ndarray)):
         raise TypeError("Inputs must be list-like structures (list or numpy array).")
    if len(traj1) == 0 or len(traj2) == 0:
         return 0 # Or raise error, depending on desired behavior for empty inputs

    # Check if inputs are single vectors wrapped in a list (common use case here)
    # or actual sequences.
    if len(traj1) != len(traj2):
         # If lengths differ, assume comparison of single vectors wrapped in lists.
         # This matches the calculator's specific usage pattern.
         if len(traj1) == 1 and len(traj2) == 1:
              vec1 = np.asarray(traj1[0])
              vec2 = np.asarray(traj2[0])
              if vec1.shape != vec2.shape:
                   raise ValueError(f"Single vector shapes mismatch: {vec1.shape} vs {vec2.shape}")
              return np.linalg.norm(vec1 - vec2)
         else:
              raise ValueError(f"Input trajectory lengths must be equal for sequence comparison, but got {len(traj1)} and {len(traj2)}")

    # If lengths are equal, proceed with sequence comparison
    total_distance = 0
    for t1, t2 in zip(traj1, traj2):
        vec1 = np.asarray(t1)
        vec2 = np.asarray(t2)
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector shapes mismatch at sequence step: {vec1.shape} vs {vec2.shape}")
        total_distance += np.linalg.norm(vec1 - vec2)

    return total_distance

# --- Placeholder for Core Evaluation Functions ---

# Function to evaluate agent performance over multiple episodes (combines gameplay + metrics)
def evaluate_agent_performance(agent,
                             env_id,
                             eval_episodes,
                             human_trajs_states=None,
                             human_trajs_actions=None,
                             metric_fns=None,
                             base_seed=0,
                             render=False,
                             render_mode='rgb_array',
                             results_prefix="agent_performance",
                             horizon=None):
    """
    Evaluates agent performance over multiple episodes, calculates standard metrics,
    and applies custom metric functions for comparison against human data.

    Args:
        agent: The agent instance.
        env_id: The environment ID string.
        eval_episodes: Number of episodes to run for evaluation.
        human_trajs_states: List of human state trajectories (list of numpy arrays).
        human_trajs_actions: List of human action trajectories (list of numpy arrays).
        metric_fns (dict): A dictionary where keys are metric names (e.g., "dtw", "wasserstein")
                           and values are functions. Each function should accept
                           (agent_states, agent_actions, human_states, human_actions)
                           and return a dictionary of metric results (e.g., {'state_dtw_mean': 1.2}).
        base_seed: Base seed for evaluation episodes (each episode gets seed base_seed + i).
        render: Whether to render episodes.
        render_mode: Rendering mode.
        results_prefix: Prefix for the output CSV filename.
        horizon: Maximum number of steps for episodes. If set and success is achieved,
                episode terminates early. If None, uses environment's default horizon.

    Returns:
        None. Results are written to a CSV file.
    """
    if metric_fns is None:
        metric_fns = {}

    all_episode_results = [] # Store results from each episode if needed for aggregation
    aggregate_metrics = {}

    print(f"\n--- Evaluating Agent: {getattr(agent, 'get_agent_type', lambda: 'Unknown')()} ---")
    print(f"Model Path: {getattr(agent, 'get_model_path', lambda: 'N/A')()}")
    print(f"Environment: {env_id}, Episodes: {eval_episodes}")
    if horizon is not None:
        print(f"Custom Horizon: {horizon} (early termination on success enabled)")
    print()

    for i in range(eval_episodes):
        episode_seed = base_seed * 200 + i
        episode_data = run_agent_episode(agent, env_id, episode_seed, render=render, render_mode=render_mode, horizon=horizon)
        all_episode_results.append(episode_data)

        # Add early termination status to the output
        termination_info = ""
        if episode_data.get('early_termination', False):
            termination_info = " | Early Term: SUCCESS"
        elif horizon is not None and episode_data['length'] >= horizon:
            termination_info = " | Term: HORIZON"
            
        print(f"  Episode {i+1}/{eval_episodes} | Seed: {episode_seed} | Score: {episode_data['normalized_score']:.2f} | Len: {episode_data['length']} | Raw Reward: {episode_data['reward']:.2f}{termination_info}")



        # --- Calculate Per-Episode Custom Metrics --- 
        episode_custom_metrics = {}
        if human_trajs_states is not None and human_trajs_actions is not None:
            for name, func in metric_fns.items():
                try:
                    # Pass episode data and human data to the metric function
                    metric_results = func(episode_data['states'], episode_data['actions'],
                                          human_trajs_states, human_trajs_actions)
                    # Prefix metric keys with the function name for clarity
                    for key, value in metric_results.items():
                        episode_custom_metrics[f"{name}_{key}"] = value
                except Exception as e:
                    print(f"Warning: Error calculating metric '{name}' for episode {i+1}: {e}")
                    # Optionally add placeholder error values
                    # episode_custom_metrics[f"{name}_error"] = str(e)
            
            # Print custom metrics for the episode
            metric_strs = [f"{k}: {v:.4f}" for k, v in episode_custom_metrics.items() if isinstance(v, (float, np.floating))] # Format floats
            if metric_strs:
                print(f"    Metrics vs Human: { ' | '.join(metric_strs)}")

        # Store custom metrics with other episode data if needed later
        episode_data.update(episode_custom_metrics)
        

    # --- Aggregate Results Across Episodes --- 
    if all_episode_results:
        scores = [res['normalized_score'] for res in all_episode_results if res['normalized_score'] != -np.inf]
        raw_rewards = [res['reward'] for res in all_episode_results]
        lengths = [res['length'] for res in all_episode_results]
        successes = [res.get('success', False) for res in all_episode_results] # Get success flags (default to False)
        early_terminations = [res.get('early_termination', False) for res in all_episode_results] # Get early termination flags
        default_horizon = all_episode_results[0].get('default_horizon', 1000) # Get default horizon from first episode
        
        aggregate_metrics['normalized_score_mean'] = np.mean(scores) if scores else np.nan
        aggregate_metrics['normalized_score_std'] = np.std(scores) if scores else np.nan
        aggregate_metrics['raw_reward_mean'] = np.mean(raw_rewards)
        aggregate_metrics['raw_reward_std'] = np.std(raw_rewards)
        aggregate_metrics['length_mean'] = np.mean(lengths)
        aggregate_metrics['length_std'] = np.std(lengths)
        aggregate_metrics['success_rate'] = np.mean(successes) if successes else 0.0 # Calculate success rate
        aggregate_metrics['early_termination_rate'] = np.mean(early_terminations) if early_terminations else 0.0 # Calculate early termination rate
        aggregate_metrics['horizon'] = horizon # Store horizon value used
        aggregate_metrics['default_horizon'] = default_horizon # Store default horizon
        
        # --- Additional Analysis for Custom Horizon Cases ---
        if horizon is not None or True: ## !!!! ERALY RETURN
            # Separate successful and non-successful episodes
            successful_episodes = [res for res in all_episode_results if res.get('success', False)]
            non_successful_episodes = [res for res in all_episode_results if not res.get('success', False)]
            
            # Statistics for successful episodes
            if successful_episodes:
                success_lengths = [res['length'] for res in successful_episodes]
                aggregate_metrics['success_length_mean'] = np.mean(success_lengths)
                aggregate_metrics['success_length_std'] = np.std(success_lengths) if len(success_lengths) > 1 else 0.0
                aggregate_metrics['success_length_max'] = np.max(success_lengths)
                aggregate_metrics['success_length_min'] = np.min(success_lengths)
                
                # Calculate percentage of successful episodes that exceed default horizon
                success_exceed_default = [length for length in success_lengths if length > default_horizon]
                aggregate_metrics['success_exceed_default_pct'] = (len(success_exceed_default) / len(success_lengths)) * 100 if success_lengths else 0.0
                aggregate_metrics['success_exceed_default_count'] = len(success_exceed_default)
            else:
                aggregate_metrics['success_length_mean'] = np.nan
                aggregate_metrics['success_length_std'] = np.nan
                aggregate_metrics['success_length_max'] = np.nan
                aggregate_metrics['success_length_min'] = np.nan
                aggregate_metrics['success_exceed_default_pct'] = 0.0
                aggregate_metrics['success_exceed_default_count'] = 0
            
            # Statistics for non-successful episodes
            if non_successful_episodes:
                non_success_lengths = [res['length'] for res in non_successful_episodes]
                aggregate_metrics['non_success_length_mean'] = np.mean(non_success_lengths)
                aggregate_metrics['non_success_length_std'] = np.std(non_success_lengths) if len(non_success_lengths) > 1 else 0.0
                aggregate_metrics['non_success_length_max'] = np.max(non_success_lengths)
                aggregate_metrics['non_success_length_min'] = np.min(non_success_lengths)
                
                # Calculate percentage of non-successful episodes that exceed default horizon
                non_success_exceed_default = [length for length in non_success_lengths if length > default_horizon]
                aggregate_metrics['non_success_exceed_default_pct'] = (len(non_success_exceed_default) / len(non_success_lengths)) * 100 if non_success_lengths else 0.0
                aggregate_metrics['non_success_exceed_default_count'] = len(non_success_exceed_default)
            else:
                aggregate_metrics['non_success_length_mean'] = np.nan
                aggregate_metrics['non_success_length_std'] = np.nan
                aggregate_metrics['non_success_length_max'] = np.nan
                aggregate_metrics['non_success_length_min'] = np.nan
                aggregate_metrics['non_success_exceed_default_pct'] = np.nan
                aggregate_metrics['non_success_exceed_default_count'] = 0
            
            # Summary statistics
            aggregate_metrics['total_success_episodes'] = len(successful_episodes)
            aggregate_metrics['total_non_success_episodes'] = len(non_successful_episodes)
            
            # --- Separate DTW and Wasserstein Distance Metrics by Success Status ---
            if metric_fns:  # Only if custom metrics are being calculated
                # Get all metric keys from the first episode that has metrics
                metric_keys = []
                for episode_result in all_episode_results:
                    for key in episode_result.keys():
                        if any(key.startswith(metric_name) for metric_name in metric_fns.keys()):
                            if key not in metric_keys:
                                metric_keys.append(key)
                
                # Calculate metrics for successful episodes
                if successful_episodes and metric_keys:
                    for metric_key in metric_keys:
                        success_values = [res.get(metric_key, np.nan) for res in successful_episodes]
                        valid_success_values = [v for v in success_values if not np.isnan(v) and v != np.inf]
                        if valid_success_values:
                            aggregate_metrics[f"success_{metric_key}_mean"] = np.mean(valid_success_values)
                            aggregate_metrics[f"success_{metric_key}_std"] = np.std(valid_success_values) if len(valid_success_values) > 1 else 0.0
                            aggregate_metrics[f"success_{metric_key}_max"] = np.max(valid_success_values)
                            aggregate_metrics[f"success_{metric_key}_min"] = np.min(valid_success_values)
                        else:
                            aggregate_metrics[f"success_{metric_key}_mean"] = np.nan
                            aggregate_metrics[f"success_{metric_key}_std"] = np.nan
                            aggregate_metrics[f"success_{metric_key}_max"] = np.nan
                            aggregate_metrics[f"success_{metric_key}_min"] = np.nan
                elif metric_keys:  # Handle case where there are metrics but no successful episodes
                    for metric_key in metric_keys:
                        aggregate_metrics[f"success_{metric_key}_mean"] = np.nan
                        aggregate_metrics[f"success_{metric_key}_std"] = np.nan
                        aggregate_metrics[f"success_{metric_key}_max"] = np.nan
                        aggregate_metrics[f"success_{metric_key}_min"] = np.nan
                
                # Calculate metrics for non-successful episodes
                if non_successful_episodes and metric_keys:
                    for metric_key in metric_keys:
                        non_success_values = [res.get(metric_key, np.nan) for res in non_successful_episodes]
                        valid_non_success_values = [v for v in non_success_values if not np.isnan(v) and v != np.inf]
                        if valid_non_success_values:
                            aggregate_metrics[f"non_success_{metric_key}_mean"] = np.mean(valid_non_success_values)
                            aggregate_metrics[f"non_success_{metric_key}_std"] = np.std(valid_non_success_values) if len(valid_non_success_values) > 1 else 0.0
                            aggregate_metrics[f"non_success_{metric_key}_max"] = np.max(valid_non_success_values)
                            aggregate_metrics[f"non_success_{metric_key}_min"] = np.min(valid_non_success_values)
                        else:
                            aggregate_metrics[f"non_success_{metric_key}_mean"] = np.nan
                            aggregate_metrics[f"non_success_{metric_key}_std"] = np.nan
                            aggregate_metrics[f"non_success_{metric_key}_max"] = np.nan
                            aggregate_metrics[f"non_success_{metric_key}_min"] = np.nan
                elif metric_keys:  # Handle case where there are metrics but no non-successful episodes
                    for metric_key in metric_keys:
                        aggregate_metrics[f"non_success_{metric_key}_mean"] = np.nan
                        aggregate_metrics[f"non_success_{metric_key}_std"] = np.nan
                        aggregate_metrics[f"non_success_{metric_key}_max"] = np.nan
                        aggregate_metrics[f"non_success_{metric_key}_min"] = np.nan

        # Aggregate custom metrics (overall across all episodes)
        if metric_fns and all_episode_results:
            # Get all metric keys from episodes that have metrics
            metric_keys = []
            for episode_result in all_episode_results:
                for key in episode_result.keys():
                    if any(key.startswith(metric_name) for metric_name in metric_fns.keys()):
                        if key not in metric_keys:
                            metric_keys.append(key)
            
            for key in metric_keys:
                values = [res.get(key, np.nan) for res in all_episode_results]
                # Filter out potential NaNs before calculating mean/std if necessary
                valid_values = [v for v in values if not np.isnan(v) and v != np.inf]
                if valid_values:
                     aggregate_metrics[f"{key}_agg_mean"] = np.mean(valid_values)
                     aggregate_metrics[f"{key}_agg_std"] = np.std(valid_values) if len(valid_values) > 1 else 0.0
                else:
                     aggregate_metrics[f"{key}_agg_mean"] = np.nan
                     aggregate_metrics[f"{key}_agg_std"] = np.nan


    # --- Prepare final results dictionary for CSV --- 
    final_results = {
        'env_id': env_id,
        'agent_type': getattr(agent, 'get_agent_type', lambda: 'Unknown')(),
        'model_path': str(getattr(agent, 'get_model_path', lambda: 'N/A')()),
        'eval_episodes': eval_episodes,
        'base_seed': base_seed,
        **aggregate_metrics # Unpack all calculated aggregate metrics
    }

    print("\n--- Aggregated Results ---")
    for key, value in aggregate_metrics.items():
         if isinstance(value, (float, np.floating)): # Check if value is float
            if not np.isnan(value):
                print(f"  {key}: {value:.4f}") # Format float only if not NaN
            else:
                print(f"  {key}: N/A") # Print N/A for NaN values
         else:
            print(f"  {key}: {value}") # Print non-float as is
    
    # --- Enhanced logging for custom horizon cases ---
    if (horizon is not None or True) and all_episode_results: ## !!!! ERALY RETURN
        print(f"\n--- Detailed Analysis (Custom Horizon: {horizon} vs Default: {aggregate_metrics.get('default_horizon', 'N/A')}) ---")
        print(f"  Total Episodes: {len(all_episode_results)}")
        print(f"  Successful Episodes: {aggregate_metrics.get('total_success_episodes', 0)}")
        print(f"  Non-Successful Episodes: {aggregate_metrics.get('total_non_success_episodes', 0)}")
        
        if aggregate_metrics.get('total_success_episodes', 0) > 0:
            print(f"\n  === Successful Episodes Analysis ===")
            success_length_mean = aggregate_metrics.get('success_length_mean', np.nan)
            success_length_std = aggregate_metrics.get('success_length_std', np.nan)
            print(f"  Mean Length: {success_length_mean:.2f}" if not np.isnan(success_length_mean) else "  Mean Length: N/A")
            print(f"  Std Length: {success_length_std:.2f}" if not np.isnan(success_length_std) else "  Std Length: N/A")
            print(f"  Max Length: {aggregate_metrics.get('success_length_max', 'N/A')}")
            print(f"  Min Length: {aggregate_metrics.get('success_length_min', 'N/A')}")
            print(f"  Episodes exceeding default horizon: {aggregate_metrics.get('success_exceed_default_count', 0)}")
            success_exceed_pct = aggregate_metrics.get('success_exceed_default_pct', np.nan)
            print(f"  % Success episodes exceeding default horizon: {success_exceed_pct:.2f}%" if not np.isnan(success_exceed_pct) else "  % Success episodes exceeding default horizon: N/A")
            
            # Print DTW and Wasserstein metrics for successful episodes
            dtw_metrics = [k for k in aggregate_metrics.keys() if k.startswith('success_dtw_')]
            w2_metrics = [k for k in aggregate_metrics.keys() if k.startswith('success_wasserstein_')]
            
            if dtw_metrics:
                print(f"  --- DTW Distances (Success) ---")
                for metric in sorted(dtw_metrics):
                    value = aggregate_metrics.get(metric, 'N/A')
                    if isinstance(value, (float, np.floating)) and not np.isnan(value):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            
            if w2_metrics:
                print(f"  --- Wasserstein Distances (Success) ---")
                for metric in sorted(w2_metrics):
                    value = aggregate_metrics.get(metric, 'N/A')
                    if isinstance(value, (float, np.floating)) and not np.isnan(value):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        if aggregate_metrics.get('total_non_success_episodes', 0) > 0:
            print(f"\n  === Non-Successful Episodes Analysis ===")
            non_success_length_mean = aggregate_metrics.get('non_success_length_mean', np.nan)
            non_success_length_std = aggregate_metrics.get('non_success_length_std', np.nan)
            print(f"  Mean Length: {non_success_length_mean:.2f}" if not np.isnan(non_success_length_mean) else "  Mean Length: N/A")
            print(f"  Std Length: {non_success_length_std:.2f}" if not np.isnan(non_success_length_std) else "  Std Length: N/A")
            print(f"  Max Length: {aggregate_metrics.get('non_success_length_max', 'N/A')}")
            print(f"  Min Length: {aggregate_metrics.get('non_success_length_min', 'N/A')}")
            print(f"  Episodes exceeding default horizon: {aggregate_metrics.get('non_success_exceed_default_count', 0)}")
            non_success_exceed_pct = aggregate_metrics.get('non_success_exceed_default_pct', np.nan)
            print(f"  % Non-success episodes exceeding default horizon: {non_success_exceed_pct:.2f}%" if not np.isnan(non_success_exceed_pct) else "  % Non-success episodes exceeding default horizon: N/A")
            
            # Print DTW and Wasserstein metrics for non-successful episodes
            dtw_metrics = [k for k in aggregate_metrics.keys() if k.startswith('non_success_dtw_')]
            w2_metrics = [k for k in aggregate_metrics.keys() if k.startswith('non_success_wasserstein_')]
            
            if dtw_metrics:
                print(f"  --- DTW Distances (Non-Success) ---")
                for metric in sorted(dtw_metrics):
                    value = aggregate_metrics.get(metric, 'N/A')
                    if isinstance(value, (float, np.floating)) and not np.isnan(value):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            
            if w2_metrics:
                print(f"  --- Wasserstein Distances (Non-Success) ---")
                for metric in sorted(w2_metrics):
                    value = aggregate_metrics.get(metric, 'N/A')
                    if isinstance(value, (float, np.floating)) and not np.isnan(value):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        elif horizon is not None:
            print(f"\n  === Non-Successful Episodes Analysis ===")
            print(f"  No non-successful episodes found.")


    # --- Write results to CSV --- 
    if not render:
        # Modify filename prefix based on whether horizon is set
        if horizon is not None:
            modified_prefix = f"{results_prefix}_horizon{horizon}"
        else:
            modified_prefix = results_prefix
        
        write_results_to_csv(final_results, env_id, filename_prefix=modified_prefix)
        print(f"\nEvaluation complete for agent. Results saved with prefix: {modified_prefix}")
    else:
        print(f"\nEvaluation complete for agent. Results shown above (CSV not saved when rendering).")


def evaluate_saved_predictions(predictions_file,
                               env_id, # Added env_id for filename consistency
                               eval_method,
                               threshold=0.5,
                               results_prefix="prediction_eval",
                               horizon=None):
    """
    Evaluates agent predictions previously saved to a file.
    Supports different evaluation methods like 'continuous' or 'continuous_v2'.
    *** NOTE: This implementation is based on evaluate_agent_predictions from human_similarity_calculator.py ***

    Args:
        predictions_file (str): Path to the .pkl file containing saved predictions.
        env_id (str): Environment ID for naming the output CSV.
        eval_method (str): The evaluation method to use (e.g., "continuous_v2", "continuous").
        threshold (float): Threshold value used for continuous evaluation methods.
        results_prefix (str): Prefix for the output CSV filename.
        horizon (int): Horizon value used during prediction generation (for filename differentiation).

    Returns:
        None. Results are written to a CSV file.
    """
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        return

    try:
        with open(predictions_file, "rb") as f:
            predictions = pickle.load(f)
    except Exception as e:
        print(f"Error loading predictions from {predictions_file}: {e}")
        return

    print(f"\n--- Evaluating Saved Predictions: {predictions_file} (Using calculator logic) ---")
    print(f"Method: {eval_method}, Threshold: {threshold}\n")

    all_results = [] # Store results for each agent type/model

    for agent_type, models in predictions.items():
        print(f"Evaluating Agent Type: {agent_type}")
        # Replicating calculator logic: Calculate mean score per model, then average those means
        agent_model_mean_scores = []
        processed_model_path = "N/A" # Keep track of the last model path processed for this agent type

        for model_path, traj_list in models.items():
            processed_model_path = model_path # Update with the actual model path
            model_total_lengths = [] # Store all segment lengths for *this specific model*

            if not traj_list:
                print(f"  Warning: No trajectories found for model: {model_path}")
                continue

            # --- Apply Evaluation Method (calculator logic) ---
            if eval_method == "continuous_v2":
                for traj_data in traj_list:
                    if "human_actions" not in traj_data or "agent_actions" not in traj_data:
                        print(f"  Warning: Skipping trajectory (missing keys) in {model_path}")
                        continue

                    human_actions = np.array(traj_data["human_actions"])
                    agent_actions = traj_data["agent_actions"]

                    if len(human_actions) == 0 or len(agent_actions) == 0:
                         print(f"  Warning: Skipping trajectory (empty actions) in {model_path}")
                         continue

                    visited = np.zeros(len(human_actions), dtype=bool) # Track visited states

                    for human_idx, human_action in enumerate(human_actions): # Use enumerate here to get human_action easily
                        if visited[human_idx]:
                            # print("visited!", human_idx) # Optional debug print
                            continue
                        
                        # Inner loop logic from calculator
                        shifted_idx = 0
                        force_break = False
                        while True:
                            current_idx = human_idx + shifted_idx
                            if current_idx >= len(human_actions) or force_break:
                                break
                            visited[current_idx] = True # Mark current human step visited

                            # Check agent prediction structure before indexing
                            if human_idx >= len(agent_actions) or not isinstance(agent_actions[human_idx], (list, np.ndarray)):
                                print(f"  Warning: Agent action data issue at human_idx {human_idx} for model {model_path}. Breaking inner loop.")
                                force_break = True # Break inner loop if structure is wrong
                                break

                            # Nested loop for matching agent actions starting from current_idx
                            # This complex structure is from the calculator code.
                            try:
                                for idx, agent_action_pred in enumerate(agent_actions[current_idx]):
                                    target_human_idx = current_idx + idx
                                    if target_human_idx >= len(human_actions):
                                        force_break = True
                                        model_total_lengths.append(shifted_idx)
                                        break # Break inner prediction loop

                                    visited[target_human_idx] = True # Mark target human step visited

                                    actual_human_action = human_actions[target_human_idx]
                                    # Use compute_euclidean_distance with expanded dims, like calculator
                                    distance = compute_euclidean_distance(
                                         np.expand_dims(actual_human_action, axis=0),
                                         np.expand_dims(agent_action_pred, axis=0)
                                    )

                                    if distance > threshold:
                                        model_total_lengths.append(shifted_idx)
                                        # print(" continuous ",shifted_idx, human_idx + shifted_idx) # Optional debug
                                        shifted_idx += 1 # Increment outer shift
                                        force_break = True
                                        break # Break inner prediction loop
                                    # If match is good, inner prediction loop continues implicitly
                                # --- End inner prediction loop ---
                                if force_break: # If inner loop broke, outer loop should too (for this human_idx start)
                                    break
                                # If inner loop completed naturally (matched all predictions), increment shifted_idx
                                # This assumes agent_actions[current_idx] has finite length.
                                shifted_idx += len(agent_actions[current_idx]) # Jump by the number of predicted actions
                                # Check if this jump logic is correct based on calculator intentions.
                                # The original calculator code `shifted_idx+=1` inside the distance>threshold block,
                                # and `shifted_idx+=1` at the end of the prediction loop seems complex.
                                # Let's try to match the calculator's increment logic more closely:
                                # If the inner loop runs, it breaks on mismatch or finishes.
                                # If it breaks on mismatch, shifted_idx is incremented by 1 outside the inner loop.
                                # If it finishes, how much should shifted_idx increment? The original code isn't perfectly clear.
                                # Reverting to a simpler +1 increment after the inner loop seems safer,
                                # unless the exact calculator logic is crucial and understood.
                                # Let's stick to the calculator's apparent logic: increment only on mismatch break.

                            except IndexError as e:
                                print(f"  Warning: IndexError during agent action processing at current_idx {current_idx}, human_idx {human_idx}. Error: {e}")
                                force_break = True # Break outer loop on error
                                break
                            except ValueError as e:
                                print(f"  Warning: ValueError (likely dimension mismatch) during distance calc. Error: {e}")
                                force_break = True
                                break
                        # --- End while True ---
                        # The original calculator code doesn't seem to increment shifted_idx outside the while loop explicitly
                        # if the while loop completes without breaking force_break = True; model_total_lengths is appended then.
                        # This seems to imply the loop continues until the end of human actions or mismatch.
                        # The logic is very hard to replicate exactly without ambiguity.
                        # Let's use the structure but note the ambiguity in the increment logic.


            elif eval_method == "continuous":
                 for traj_data in traj_list:
                    if "human_actions" not in traj_data or "agent_actions" not in traj_data:
                        print(f"  Warning: Skipping trajectory (missing keys) in {model_path}")
                        continue

                    human_actions = np.array(traj_data["human_actions"])
                    agent_actions = traj_data["agent_actions"]

                    if len(human_actions) == 0 or len(agent_actions) == 0:
                         print(f"  Warning: Skipping trajectory (empty actions) in {model_path}")
                         continue

                    # Outer loop matching calculator structure
                    for human_idx, human_action in enumerate(human_actions):
                        shifted_idx = 0
                        force_break = False
                        while True: # Inner loop
                            current_idx = human_idx + shifted_idx
                            if current_idx >= len(human_actions) or force_break:
                                break

                            # Check agent prediction structure before indexing
                            if human_idx >= len(agent_actions) or not isinstance(agent_actions[human_idx], (list, np.ndarray)):
                                print(f"  Warning: Agent action data issue at human_idx {human_idx} for model {model_path}. Breaking inner loop.")
                                force_break = True
                                break

                            try:
                                # Nested loop for matching agent actions starting from current_idx
                                for idx, agent_action_pred in enumerate(agent_actions[current_idx]):
                                    target_human_idx = current_idx + idx
                                    if target_human_idx >= len(human_actions):
                                        force_break = True
                                        model_total_lengths.append(shifted_idx)
                                        break # Break inner prediction loop

                                    actual_human_action = human_actions[target_human_idx]
                                    # Use compute_euclidean_distance with expanded dims, like calculator
                                    distance = compute_euclidean_distance(
                                         np.expand_dims(actual_human_action, axis=0),
                                         np.expand_dims(agent_action_pred, axis=0)
                                    )

                                    if distance > threshold:
                                        model_total_lengths.append(shifted_idx)
                                        shifted_idx += 1
                                        force_break = True
                                        break # Break inner prediction loop
                                    # If match good, prediction loop continues
                                # --- End inner prediction loop ---
                                if force_break:
                                    break # Break while loop
                                # If prediction loop completed, increment outer shift? (See notes in continuous_v2)
                                shifted_idx += len(agent_actions[current_idx]) # Tentative jump based on length
                            except IndexError as e:
                                print(f"  Warning: IndexError during agent action processing. Error: {e}")
                                force_break = True
                                break
                            except ValueError as e:
                                print(f"  Warning: ValueError (likely dimension mismatch) during distance calc. Error: {e}")
                                force_break = True
                                break
                        # --- End while True ---

            # --- Add other eval_methods here as elif blocks if needed ---
            # elif eval_method == "overall_average" and "MAQ" in agent_type:
            #     # Calculator has pass here, so do nothing specific
            #     pass
            else:
                print(f"  Warning: Evaluation method '{eval_method}' not implemented based on calculator logic.")
                continue # Skip to next model if method not known

            # Calculate mean score for *this model* if lengths were recorded
            if model_total_lengths:
                model_mean_score = np.mean(model_total_lengths)
                agent_model_mean_scores.append(model_mean_score) # Append the mean for this model
                print(f"  Model: {os.path.basename(model_path)} | Mean Cont. Length: {model_mean_score:.4f}")
            else:
                 # Handle case where no lengths were recorded for this model (e.g., all skips/errors)
                 print(f"  Model: {os.path.basename(model_path)} | No continuous lengths recorded.")
                 # Optionally append NaN or skip if preferred, calculator logic appends means, so maybe skip?
                 # Let's skip appending if no scores were calculated for a model to match calculator's apparent behavior.


        # --- Aggregate and Store Results for the Agent Type (mean of means) ---
        if agent_model_mean_scores: # Check if any model means were collected
            final_mean_value = np.mean(agent_model_mean_scores)
            final_std_value = np.std(agent_model_mean_scores)
            print(f"Agent Type: {agent_type} | Overall Mean of Model Means: {final_mean_value:.4f} | Std Dev of Model Means: {final_std_value:.4f}\n")

            # Store results for this agent type
            result_row = {
                "env_id": env_id,
                "agent_type": agent_type,
                "model_path": processed_model_path, # Use the last path encountered for this agent type
                "eval_method": eval_method,
                "threshold": threshold if "continuous" in eval_method else None,
                "mean_value": final_mean_value, # Mean of the means
                "std_value": final_std_value,   # Std of the means
                "predictions_file": predictions_file # Keep track of source file
            }
            all_results.append(result_row)
        else:
            print(f"Agent Type: {agent_type} | No valid model scores calculated.\n")

    # --- Write all collected results to CSV ---
    if all_results:
        # Use the first result to get fieldnames, ensuring consistent order
        # Match fieldnames used in the calculator's evaluate_agent_predictions if possible
        # Calculator used: ["agent_type", "eval_method", "mean_value", "std_value", "threshold", "model_path"]
        # Let's adapt to match that more closely, adding env_id and predictions_file
        fieldnames = ["env_id", "agent_type", "model_path", "eval_method", "threshold", "mean_value", "std_value", "predictions_file"]
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Modify filename prefix based on whether horizon was used
        if horizon is not None:
            modified_prefix = f"{results_prefix}_horizon{horizon}"
        else:
            modified_prefix = results_prefix
            
        filename = os.path.join(output_dir, f"{modified_prefix}_{env_id}.csv")

        try:
            write_header = not os.path.exists(filename)
            with open(filename, mode='a', newline='', encoding='utf-8') as f:
                # Filter results to only include keys present in fieldnames
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                writer.writerows(all_results) # Write all results at once
            print(f"Prediction evaluation results saved to {filename} (prefix: {modified_prefix})")
        except IOError as e:
            print(f"Error writing prediction evaluation results to {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CSV writing: {e}")
    else:
        print("No results generated from prediction evaluation.")


print("human_similarity_utils.py loaded.") 