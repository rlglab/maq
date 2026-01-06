import argparse
import numpy as np
import torch
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import os
import sys
import re
sys.path.append("human_similarity")
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("O2ORL")
import gym 
import d4rl
import json
import csv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import pprint
import pandas as pd
import matplotlib.pyplot as plt
import os
import moviepy.video.io.ImageSequenceClip
# from ws_dist import *
import pickle
from human_similarity.utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import glob
from tensorboard.backend.event_processing import event_accumulator
sys.path.append('../offline_data')
from human_similarity_utils import (
    run_agent_episode,
    evaluate_agent_performance,
    evaluate_saved_predictions,
    calculate_dtw_distances,
    calculate_wasserstein_distances,
    calculate_min_dtw_distances
)

DEFAULT_SEQLEN_IQL = 3
DEFAULT_K_IQL = 16
MAQ_IQL_CSV_FILENAME = "experiment_short.csv" 

# --- Environment Horizon Constant ---
# Set this to None to use environment's default horizon, or set to a specific number
# If set, episodes will terminate early if success is achieved before the horizon
ENVIRONMENT_HORIZON = None  # You can change this value (e.g., 1000, 500, etc.) 

def load_trajectories(file):
    """從檔案加載 trajectory 數據"""
    with open(file, 'rb') as f:
        return pickle.load(f)

def get_d4rl_dataset(env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
    dataset = d4rl.qlearning_dataset(gym.make(env))
    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    dones_float[-1] = 1

    return dataset['observations'].astype(np.float32), dataset['actions'].astype(np.float32), dataset['rewards'].astype(np.float32), dones_float.astype(np.float32), dataset['next_observations'].astype(np.float32)


def normalized_score(env,score):
    return env.get_normalized_score(score)*100

def Calc_Distance(A, B):
    # get L1 distance
    assert A.shape == B.shape
    return np.sum(np.abs(A - B))

def compute_dtw_distance(traj1, traj2):
    assert traj1[0].shape == traj2[0].shape
    distance, _ = fastdtw(traj1, traj2, dist=euclidean)
    return distance

def compute_euclidean_distance(traj1, traj2):
    """
    用簡單的歐幾里得距離比較兩個序列的相似性。
    :param traj1: 第一個軌跡 (list of numpy arrays)
    :param traj2: 第二個軌跡 (list of numpy arrays)
    :return: 累計歐幾里得距離
    """
    # 確保兩個軌跡的長度相同
    assert len(traj1) == len(traj2), "The trajectories must have the same length."
    
    # 確保每一步的 shape 相同
    for t1, t2 in zip(traj1, traj2):
        assert t1.shape == t2.shape, "Each step in the trajectories must have the same shape."

    # 計算每一步的歐幾里得距離並累加
    total_distance = sum(np.linalg.norm(t1 - t2) for t1, t2 in zip(traj1, traj2))
    
    return total_distance


def similarity_metrics(env, env_id, testing_dataset, eval_seqlen, eval_episodes, report_file, 
                        agents = [],
                        render=False,
                        horizon=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define which metric functions to use during performance evaluation
    # The keys ('dtw') will be used in the output CSV headers
    metrics_to_calculate = {
        'dtw': calculate_dtw_distances,
        # Add other metric functions here, e.g.:
        'wasserstein': calculate_wasserstein_distances,
        'min_dtw': calculate_min_dtw_distances
    }

    # --- Load Human Trajectories (Common for all agents) --- 
    # Assuming only one seed is needed for human data comparison, use the first agent's seed?
    # Or perhaps load a generic human dataset not tied to a specific agent seed.
    # Using seed 0 as a placeholder, adjust if needed.

    # --- Evaluate Each Agent --- 
    for agent in agents:
        agent_seed = agent.get_seed()
        agent_type = agent.get_agent_type()

        human_traj_states = []
        human_traj_actions = []
        test_trajectories = load_trajectories(f"offline_data/{testing_dataset}")
        for traj in test_trajectories:
            human_traj_states.append(traj['observations'])
            human_traj_actions.append(traj['actions'])
        print(f"Loaded {len(human_traj_states)} human trajectories for comparison (seed {agent_seed}).")
    
        print(f"Processing Agent: {agent_type}, Seed: {agent_seed}")

        # --- Run Performance Evaluation (includes gameplay and metrics vs human) ---
        # This function now handles running episodes, calculating scores, and optionally metrics.
        # Results are saved to CSV inside the function.
        evaluate_agent_performance(
            agent=agent,
            env_id=env_id,
            eval_episodes=eval_episodes,
            human_trajs_states=human_traj_states,
            human_trajs_actions=human_traj_actions,
            metric_fns=metrics_to_calculate,
            base_seed=agent_seed, # Use agent's seed for its eval episodes
            render=render,
            results_prefix="agent_performance", # Specific prefix for performance results
            horizon=horizon # Pass horizon parameter
        )

        # --- Save Agent Predictions (Optional Step) ---
        # Keep this if you need the raw predictions saved separately for later analysis
        # with different methods/thresholds not covered by evaluate_agent_performance.

    print("\n--- similarity_metrics function finished ---")

def evaluate_agent_predictions(report_file, save_file, eval_method, threshold=0.5, render=False):
    """
    支援四種不同的評估方法，根據指定的 eval_method 進行分析。
    """
    # 從檔案中讀取資料
    with open(save_file, "rb") as f:
        predictions = pickle.load(f)

    results = []
    for agent_type, models in predictions.items():
        print(f"Evaluating Agent Type: {agent_type}")
        agent_results = []
        score = []
        for model_path, traj_list in models.items():
            if eval_method == "continuous_v2":
                total_length = []
                for traj_data in traj_list:
                    human_actions = np.array(traj_data["human_actions"])
                    agent_actions = traj_data["agent_actions"]
                    visited = np.array([False]*len(human_actions))
                    for human_idx, human_action in enumerate(human_actions):
                        shifted_idx = 0
                        if visited[human_idx]:
                            print("visited!",human_idx)
                            continue 
                        force_break = False
                        while True:
                            current_idx = human_idx + shifted_idx
                            if current_idx >= len(human_actions) or force_break:
                                break
                            visited[current_idx] = True
                            distance = compute_euclidean_distance(
                                np.expand_dims(agent_actions[human_idx][0], axis=0), 
                                np.expand_dims(human_action, axis=0))
                            for idx, agent_action in enumerate(agent_actions[current_idx]):
                                if current_idx+idx >= len(human_actions):
                                    force_break = True
                                    total_length.append(shifted_idx)
                                    break
                                visited[current_idx+idx] = True
                                # print(agent_action)
                                # print(human_actions[current_idx])
                                distance = compute_euclidean_distance(
                                    np.expand_dims(human_actions[current_idx+idx], axis=0), 
                                    np.expand_dims(agent_actions[current_idx][idx], axis=0))
                                if distance > threshold:
                                    total_length.append(shifted_idx)
                                    # print(" continuous ",shifted_idx, human_idx + shifted_idx)
                                    shifted_idx+=1
                                    force_break = True
                                    break
                                shifted_idx+=1
                # print(total_length)
                mean_value = np.mean(total_length)
                std_value = np.std(total_length)
                score.append(mean_value)

            elif eval_method == "continuous":
                total_length = []
                for traj_data in traj_list:
                    human_actions = np.array(traj_data["human_actions"])
                    agent_actions = traj_data["agent_actions"]

                    for human_idx, human_action in enumerate(human_actions):
                        shifted_idx = 0
                        force_break = False
                        while True:
                            current_idx = human_idx + shifted_idx
                            if current_idx >= len(human_actions) or force_break:
                                break
                            
                            distance = compute_euclidean_distance(
                                np.expand_dims(agent_actions[human_idx][0], axis=0), 
                                np.expand_dims(human_action, axis=0))
                            for idx, agent_action in enumerate(agent_actions[current_idx]):
                                if current_idx+idx >= len(human_actions):
                                    force_break = True
                                    total_length.append(shifted_idx)
                                    break
                                # print(agent_action)
                                # print(human_actions[current_idx])
                                distance = compute_euclidean_distance(
                                    np.expand_dims(human_actions[current_idx+idx], axis=0), 
                                    np.expand_dims(agent_actions[current_idx][idx], axis=0))
                                if distance > threshold:
                                    total_length.append(shifted_idx)
                                    # print(" continuous ",shifted_idx, human_idx + shifted_idx)
                                    shifted_idx+=1
                                    force_break = True
                                    break
                                shifted_idx+=1
                mean_value = np.mean(total_length)
                std_value = np.std(total_length)
                score.append(mean_value)
                        # print(shifted_idx)
            elif eval_method == "overall_average" and "MAQ" in agent_type:
                pass
        mean_value = np.mean(score)
        std_value = np.std(score)

        # 儲存結果
        results.append({
            "agent_type": agent_type,
            "eval_method": eval_method,
            "mean_value": mean_value,
            "std_value": std_value,
            "threshold": threshold,
            "model_path": model_path,
        })
    # 將結果寫入 CSV 文件，除非啟用了渲染
    if not render:
        fieldnames = ["agent_type", "eval_method", "mean_value", "std_value", "threshold", "model_path"]
        with open(report_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:  # 如果文件是空的，寫入表頭
                writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"Results saved to {report_file}")
    else:
        print("Rendering enabled - CSV results not saved")

    return results

def save_agent_predictions(human_traj_states, human_traj_actions, agent, save_file):
    """
    儲存單一 agent 的預測結果到指定的 pkl 檔案，支援增量追加資料。
    """
    # 檢查檔案是否存在，若不存在則初始化空結構
    if os.path.exists(save_file):
        with open(save_file, "rb") as f:
            predictions = pickle.load(f)
    else:
        predictions = {}

    # 獲取 agent 的類型與模型路徑
    agent_type = agent.get_agent_type()
    model_path = str(agent.get_model_path())

    # 初始化結構（若 agent_type 或 model_path 不存在）
    if agent_type not in predictions:
        predictions[agent_type] = {}
    if model_path not in predictions[agent_type]:
        predictions[agent_type][model_path] = []
    else:
        print("SKIP",agent_type,model_path,"!")
        return 
    # 新增 trajectory 資料
    for traj_idx in range(len(human_traj_states)):
        human_traj_state = human_traj_states[traj_idx]
        human_traj_action = human_traj_actions[traj_idx]

        agent_predictions = []
        for human_idx, human_state in enumerate(human_traj_state):
            agent_actions = agent.inference(human_state)
            agent_predictions.append(agent_actions)

        # 將結果附加到現有資料
        predictions[agent_type][model_path].append({
            "traj_id": traj_idx,
            "human_states": human_traj_state,
            "human_actions": human_traj_action,
            "agent_actions": agent_predictions
        })

    # 將更新後的資料寫回檔案
    with open(save_file, "wb") as f:
        pickle.dump(predictions, f)

    print(f"Predictions for {agent_type} ({model_path}) appended to {save_file}")


def calc_accuracy(human_traj_states,human_traj_actions,agent,eval_seqlen,report_file,env_id,fieldnames,first_only=False,continuous_match=False,continuous_thresholds=[0.1,0.25,0.5,0.75,0.9,1.0]):
    # first_only : only compare first
    # continuous_match : check if continuous if not same then stopped
    # if no first_only and no continuous_match , its all match to calc the distance
    print("First only",first_only)
    print("Continuous match",continuous_match,"thresholds",continuous_thresholds)
    mode = ""
    if first_only:
        mode = "first_only"
        print("ONLY COMPARE FIRST ACTION DISTANCE")
    elif continuous_match:
        mode = "continuous_match"
        print("ONLY COMPARE CONTINUOUS MATCHING WITH THRESHOLDS",continuous_thresholds)
    else:
        mode = "all_match"
        print("ONLY COMPARE EACH MACRO ACTION DISTANCE")
    
    for traj_idx in range(len(human_traj_states)):
        total_distance = []
        continuous_acc = []
        human_traj_state = human_traj_states[traj_idx]
        human_traj_action = human_traj_actions[traj_idx]
        
        for human_idx in range(len(human_traj_state)-eval_seqlen):
            human_state = human_traj_state[human_idx]
            human_actions = []
            for shift_idx in range(eval_seqlen):
                human_actions.append(human_traj_action[human_idx + shift_idx])
            
            agent_actions = agent.inference(human_state)
            human_actions = np.array(human_actions)

            # print(human_actions.shape,agent_actions.shape)
            if first_only:
                distance = compute_euclidean_distance(
                    np.expand_dims(agent_actions[0], axis=0), 
                    np.expand_dims(human_actions[0], axis=0))

                total_distance.append(distance)
            elif continuous_match:
                continuous = [True] * len(continuous_thresholds)
                acc = [0] * len(continuous_thresholds)
                for idx in range(eval_seqlen):
                    distance = compute_euclidean_distance(
                        np.expand_dims(agent_actions[idx], axis=0), 
                        np.expand_dims(human_actions[idx], axis=0))
                    for c_idx in range(len(continuous_thresholds)):
                        if continuous[c_idx] and distance < continuous_thresholds[c_idx]:
                            acc[c_idx] += 1
                        else:
                            continuous[c_idx] = False
                continuous_acc.append(acc)
            else:
                distance = compute_euclidean_distance(agent_actions, human_actions)
                total_distance.append(distance)
        value = 0.0   
        if continuous_match:
            continuous_acc = np.array(continuous_acc)
            continuous_acc = continuous_acc / 3 * 100
            col_sums = np.sum(continuous_acc, axis=0)
            col_means = np.mean(continuous_acc, axis=0)
            col_std = np.std(continuous_acc, axis=0)


            # diff_value = col_means / 3 * 100
            print(col_means) 
            for value_idx in range(len(continuous_thresholds)):
                with open(report_file, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(
                        {'env_id': env_id, 
                         'agent_type': f"{agent.get_agent_type()}", 
                        'eval_method': f'{mode}_{continuous_thresholds[value_idx]}', 
                        'eval_source': f'human_traj_{traj_idx}', 
                        'model_path': agent.get_model_path(), 
                        'value_mean': col_means[value_idx],
                        'value_std': col_std[value_idx],
                        'action_seqlen': eval_seqlen})
        else:
            mean_value = np.mean(total_distance)
            std_value = np.std(total_distance)

            with open(report_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(
                    {'env_id': env_id, 
                     'agent_type': f"{agent.get_agent_type()}", 
                    'eval_method': f'{mode}', 
                    'eval_source': f'human_traj_{traj_idx}', 
                    'model_path': agent.get_model_path(), 
                    'value_mean': mean_value, 
                    'value_std': std_value,
                    'action_seqlen': eval_seqlen})

def simple_gameplay_w2_dist(env_id, eval_episodes, agent, human_traj_states, human_traj_actions, render=False, seed=0):
    if "RLPD" in agent.get_agent_type():
        print("RLPD!")
        env = gym.make(env_id)
        env = wrap_gym(env, rescale_actions=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        env.seed(100+42)
    else:
        env = gym.make(env_id)
    total_rewards = []  # 用於儲存每個 episode 的總獎勵
    
    eval_state_traj = []
    eval_action_traj = []
    for i in range(eval_episodes):
        obs_list = []
        if render:
            env = gym.make(env_id)
            env.seed(seed+i)
            viewer = env.mj_viewer_setup()
            env.render()
            

        state = env.reset()
        done = False
        total_reward = 0
        state_traj = []
        action_traj = []
        state_traj.append(state)
        length = 0
        if render:
            # obs_list.append(env.render(mode='rgb_array'))
            obs_list.append(env.viewer._read_pixels_as_in_window())  # for Adroit
        
        while not done:
            action = agent.inference(state) # (1,28)
            for a in action:
                state, reward, done, _ = env.step(a)
                total_reward += reward
                state_traj.append(state)
                action_traj.append(a)
                length += 1
                
                if render:
                    # obs_list.append(env.render(mode='rgb_array'))
                    obs_list.append(env.viewer._read_pixels_as_in_window())  # for Adroit
                if done:
                    break
        eval_state_traj.append(np.array(state_traj))
        eval_action_traj.append(np.array(action_traj))
        raw_reward = total_reward
        total_reward = normalized_score(env, total_reward)
        total_rewards.append(total_reward)  # 儲存每個 episode 的總獎勵
        print(f"Episode {i+1} finished reward: {total_reward} from {agent.get_agent_type()} len:{length}")
        if render:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use agent's short name if available
            agent_name = agent.get_agent_short_name() if hasattr(agent, 'get_agent_short_name') else agent.get_agent_type()
            
            os.makedirs(f'record/{env_id}/seed{seed}', exist_ok=True)

            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(obs_list, fps=30)
            video_path = f'record/{env_id}/seed{seed}/{agent_name}_{i}_{timestamp}_temp.mp4'
            clip.write_videofile(video_path)

    
    total_state_distance = [compute_wasserstein_distance_multiple(eval_state_traj,human_traj_states)]
    total_action_distance = [compute_wasserstein_distance_multiple(eval_action_traj,human_traj_actions)]
    print(f"wasserestin distance  state: {total_state_distance[0]},  action: {total_action_distance[0]}")

    # 計算均值和標準差
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Evaluation completed for {eval_episodes} episodes.")
    print(f"Mean reward: {mean_reward}")
    print(f"Standard deviation of rewards: {std_reward}")
    if render:
        for i in range(eval_episodes):
            temp_fname = f'record/{env_id}/seed{seed}/{agent_name}_{i}_{timestamp}_temp.mp4'
            final_fname = f'record/{env_id}/seed{seed}/{agent_name}_score_{total_rewards[i]:.2f}_state_{total_state_distance[i]:.2f}_action_{total_action_distance[i]:.2f}_seed{seed+i}_{timestamp}.mp4'
            try:
                os.rename(temp_fname, final_fname)
            except: 
                pass
        return
    
    # Only write to CSV if not rendering
    # save results to csv file
    fieldnames = ['env_id', 'agent_type','eval_method','model_path', 'value_mean','value_std','action_seqlen','state_distance_mean','state_distance_std','action_distance_mean','action_distance_std']
    with open(f"D4RL_{env_id}_score_w2_dist.csv", 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {'env_id': env_id,
             'agent_type': f"{agent.get_agent_type()}", 
            'eval_method': f'SCORE {eval_episodes}', 
            'model_path': agent.get_model_path(), 
            'value_mean': f'{mean_reward}', 
            'value_std': f'{std_reward}',
            'action_seqlen': eval_seqlen,
            'state_distance_mean': np.mean(total_state_distance),
            'state_distance_std': np.std(total_state_distance,ddof=0), 
            'action_distance_mean': np.mean(total_action_distance),
            'action_distance_std': np.std(total_action_distance,ddof=0), 
        })
    return total_rewards, mean_reward, std_reward



def simple_gameplay(env_id, eval_episodes, agent, human_traj_states, human_traj_actions, render=False, seed=0):
    if "RLPD" in agent.get_agent_type():
        print("RLPD!")
        env = gym.make(env_id)
        env = wrap_gym(env, rescale_actions=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        env.seed(100+42)
    else:
        env = gym.make(env_id)
    total_rewards = []  # 用於儲存每個 episode 的總獎勵
    
    eval_state_traj = []
    eval_action_traj = []
    for i in range(eval_episodes):
        obs_list = []
        if render:
            env = gym.make(env_id)
            env.seed(seed+i)
            viewer = env.mj_viewer_setup()
            env.render()
            

        state = env.reset()
        done = False
        total_reward = 0
        state_traj = []
        action_traj = []
        state_traj.append(state)
        length = 0
        if render:
            # obs_list.append(env.render(mode='rgb_array'))
            obs_list.append(env.viewer._read_pixels_as_in_window())  # for Adroit
        
        while not done:
            action = agent.inference(state) # (1,28)
            for a in action:
                state, reward, done, _ = env.step(a)
                total_reward += reward
                state_traj.append(state)
                action_traj.append(a)
                length += 1
                
                if render:
                    # obs_list.append(env.render(mode='rgb_array'))
                    obs_list.append(env.viewer._read_pixels_as_in_window())  # for Adroit
                if done:
                    break
        eval_state_traj.append(np.array(state_traj))
        eval_action_traj.append(np.array(action_traj))
        raw_reward = total_reward
        total_reward = normalized_score(env, total_reward)
        total_rewards.append(total_reward)  # 儲存每個 episode 的總獎勵
        print(f"Episode {i+1} finished reward: {total_reward} from {agent.get_agent_type()} len:{length}")
        if render:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use agent's short name if available
            agent_name = agent.get_agent_short_name() if hasattr(agent, 'get_agent_short_name') else agent.get_agent_type()
            
            os.makedirs(f'record/{env_id}/seed{seed}', exist_ok=True)

            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(obs_list, fps=30)
            video_path = f'record/{env_id}/seed{seed}/{agent_name}_{i}_{timestamp}_temp.mp4'
            clip.write_videofile(video_path)

    
    
    total_state_distance = []
    total_action_distance = []
    for i in range(eval_episodes):
        episode_state_distance = 0
        episode_action_distance = 0
        for traj in human_traj_states:
            episode_state_distance += compute_dtw_distance(eval_state_traj[i], traj)
        for traj in human_traj_actions:
            episode_action_distance += compute_dtw_distance(eval_action_traj[i], traj)
        episode_state_distance /= len(human_traj_states)
        episode_action_distance /= len(human_traj_actions)
        print(f"Episode {i+1} has avg state distance: {episode_state_distance}, avg action distance: {episode_action_distance}")
        total_state_distance.append(episode_state_distance)
        total_action_distance.append(episode_action_distance)

    # 計算均值和標準差
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Evaluation completed for {eval_episodes} episodes.")
    print(f"Mean reward: {mean_reward}")
    print(f"Standard deviation of rewards: {std_reward}")
    if render:
        for i in range(eval_episodes):
            temp_fname = f'record/{env_id}/seed{seed}/{agent_name}_{i}_{timestamp}_temp.mp4'
            final_fname = f'record/{env_id}/seed{seed}/{agent_name}_score_{total_rewards[i]:.2f}_state_{total_state_distance[i]:.2f}_action_{total_action_distance[i]:.2f}_seed{seed+i}_{timestamp}.mp4'
            try:
                os.rename(temp_fname, final_fname)
            except: 
                pass
        return
    
    # Only write to CSV if not rendering
    # save results to csv file
    fieldnames = ['env_id', 'agent_type','eval_method','model_path', 'value_mean','value_std','action_seqlen','state_distance_mean','state_distance_std','action_distance_mean','action_distance_std']
    with open(f"D4RL_{env_id}_score.csv", 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {'env_id': env_id,
             'agent_type': f"{agent.get_agent_type()}", 
            'eval_method': f'SCORE {eval_episodes}', 
            'model_path': agent.get_model_path(), 
            'value_mean': f'{mean_reward}', 
            'value_std': f'{std_reward}',
            'action_seqlen': eval_seqlen,
            'state_distance_mean': np.mean(total_state_distance),
            'state_distance_std': np.std(total_state_distance,ddof=0), 
            'action_distance_mean': np.mean(total_action_distance),
            'action_distance_std': np.std(total_action_distance,ddof=0), 
        })
    return total_rewards, mean_reward, std_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating methods.')
    parser.add_argument('--env', type=str,help='')
    parser.add_argument('--gpuid', type=str,help='',default='3')
    parser.add_argument('--eval_agent', type=str,help='',default='')
    parser.add_argument('--render', type=bool, default=False, help='Render and save video of agent performance')
    parser.add_argument('--suffix', type=str,help='',default='')
    parser.add_argument('--model_path', type=str,help='',default='')
    parser.add_argument('--seed', type=str,help='',default='')
    parser.add_argument('--training_dataset', type=str,help='',default='')
    parser.add_argument('--testing_dataset', type=str,help='',default='')
    
    

    args = parser.parse_args()
    if args.gpuid != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid  
    
    

    # Parameters
    eval_seqlen = 3
    eval_episodes = 100 
    if args.render:
        eval_episodes = 5
    
    seed = args.seed
    env_id = args.env
    method = args.eval_agent
    training_dataset = args.training_dataset
    testing_dataset = args.testing_dataset
    suffix = args.suffix
    if not method in ["MAQ+DSAC","MAQ+RLPD","MAQ+IQL","SAC","RLPD","IQL"]:
        print("Method must be one of MAQ+DSAC, MAQ+RLPD, MAQ+IQL, SAC, RLPD, IQL")
        exit(0)

    generic_agent_type = None
    if method == "MAQ+RLPD":
        base_paths = [f"RLPD_MAQ/log/exp_{suffix}"]
        target_keywords = ["RLPDMAQ"]
        print(f"Base Paths: {base_paths}")

        agents = get_best_MAQ_agent(
            base_paths, # Use the constructed base paths
            args.env,
            env_id,
            agent_type=RLPDMAQAgent,
            target_keywords=target_keywords,
            tag=suffix,
            seed=str(seed)
        )
        generic_agent_type=RLPDMAQAgent

    elif method == "MAQ+DSAC":
        base_paths = [f"RLPD_MAQ/log/exp_{suffix}"]
        target_keywords = ["DSACMAQ"]
        print(f"Base Paths: {base_paths}")

        agents = get_best_MAQ_agent(
            base_paths, # Use the constructed base paths
            args.env,
            env_id,
            agent_type=DSACMAQAgent,
            target_keywords=target_keywords,
            tag=suffix,
            seed=str(seed)
        )
        generic_agent_type=DSACMAQAgent
        
    elif method == "MAQ+IQL":
        base_paths = [
            f"IQL/log/{env_id}/MAQ_iql{suffix}_seed{seed}",
        ]
        print(f"Base Paths: {base_paths}")

        agents = get_best_MAQ_IQL_agent(
            base_paths,
            args.env,
            env_id,
            tag=suffix
        )
        generic_agent_type=MAQIQLAgent
        
        
    elif method == "SAC":
        agents = []
        ckpt = find_best_checkpoint_SAC(f"SAC/log/{args.env}/SAC_seed{seed}")
        agents.append(FlexibleAgentInterface(SB3Agent, ckpt, env_id))

        generic_agent_type=SB3Agent
        
    elif method == "RLPD":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        import d4rl.gym_mujoco
        import d4rl.locomotion
        import dmcgym
        from rlpd.wrappers import wrap_gym
        from human_similarity.agent_rlpd import *
        agents = []
        #checkpoint_0.orbax-checkpoint-tmp-0
        ckpt = f"/workspace/rlpd/log_door_seed1/s1_0pretrain_LN/checkpoints/checkpoint_1000000"
        agents.append(FlexibleAgentInterface(RLPDAgent, ckpt, env_id))
        generic_agent_type=RLPDAgent
    elif method == "IQL":  
        agents = get_best_IQL_agent(
            [
                f"IQL/log/{args.env}/iql_seed{seed}"
            ],
            args.env,
            env_id
        )
        generic_agent_type=IQLAgent
    
    if args.model_path != '':
        # load specific model path
        agents = []
        agents.append(FlexibleAgentInterface(generic_agent_type, args.model_path, env_id))
    
    if len(agents) == 0:
        print("No agents found!")
        exit(0)
    # print(">>>",agents)
    # tmp_check = []
    # for agent in agents:
    #     s = agent.test()
    #     tmp_check.append(s)
    #     agent.close()
    # print("\n".join(tmp_check))    
    # print("total agents:",len(agents)) # current version do not support multiple agents at same time
    assert len(agents) == 1 # must equals to one
    

    horizon = ENVIRONMENT_HORIZON
    
    similarity_metrics(args.env, env_id, testing_dataset, eval_seqlen, eval_episodes, f"total_human_sim_report_{args.env}.csv", agents, render=args.render, horizon=horizon)