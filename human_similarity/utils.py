import os
import pandas as pd
from AbstractAgent import *
from human_similarity.agent_IQL import *
from human_similarity.agent_MAQ_IQL import *
from human_similarity.agent_SB3 import *
from human_similarity.agent_Rnd import *
from human_similarity.agent_DSACMAQ import *
from human_similarity.agent_RLPDMAQ import *
from tensorboard.backend.event_processing import event_accumulator
import re
import glob

def find_best_checkpoints(base_path, game_type, target_keywords = ["Top1", "Top3", "MAQB", "coef0"],seed="1"):
    result = {}
    # 定義需要搜尋的關鍵字
    
    
    # 遍歷 base_path 下所有子目錄
    for root, dirs, files in os.walk(base_path):
        # 檢查是否包含 game_type 和目標關鍵字
        # print(root)
        if game_type in root and any(keyword in root for keyword in target_keywords):
            # 找到包含 experiment_data.csv 的目錄
            if "experiment_data.csv" in files:
                # 讀取 experiment_data.csv
                csv_path = os.path.join(root, "experiment_data.csv")
                try:
                    experiment_data = pd.read_csv(csv_path)
                    if 'episode_reward' not in experiment_data.columns or 'training_step' not in experiment_data.columns:
                        print(f"Warning: Missing required columns in {csv_path}. Skipping.")
                        continue
                    if experiment_data.empty:
                        print(f"Warning: Empty CSV file {csv_path}. Skipping.")
                        continue

                    # Find the maximum reward value
                    max_reward = experiment_data['episode_reward'].max()

                    # Get all rows that have this maximum reward
                    best_rows = experiment_data[experiment_data['episode_reward'] == max_reward]

                    if best_rows.empty:
                         print(f"Warning: Could not find rows with max reward in {csv_path}. Skipping.")
                         continue

                    # Select the first row among the best ones (handles ties by picking first occurrence)
                    best_row = best_rows.iloc[0]

                    # 找到最高 episode_reward 的對應 training_step
                    # best_row = experiment_data.loc[experiment_data['episode_reward'].idxmax()]
                    best_training_step = int(best_row['training_step'])

                    # 在目錄中搜尋對應的 checkpoint 檔案
                    found_checkpoint = False
                    for file in files:
                        if file.startswith(f"model_{best_training_step}_") and file.endswith(".pth"):
                            checkpoint_path = os.path.join(root, file)
                            # Extract seed name from the parent directory name
                            seed_match = re.search(r'_seed(\d+)', os.path.basename(root))
                            if seed_match:
                                seed_name = seed_match.group(1)
                            else:
                                # Fallback or warning if seed pattern not found
                                print(f"Warning: Could not extract seed from directory name {os.path.basename(root)}")
                                seed_name = str(seed)  # Use provided seed as fallback
                                

                            # 存儲結果
                            if seed_name not in result:
                                result[seed_name] = {}
                            result[seed_name][root] = checkpoint_path
                            found_checkpoint = True
                            break # Found the checkpoint for this step
                    if not found_checkpoint:
                         print(f"Warning: Found max reward step {best_training_step} in {csv_path}, but no matching model file found.")

                except pd.errors.EmptyDataError:
                    print(f"Warning: CSV file {csv_path} is empty. Skipping.")
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}. Skipping.")
    
    return result
def find_last_checkpoints_VQPrior(vqvae_dir, prior_dir, env):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 找到 VQVAE 目錄中最後一個模型檔案
    vqvae_files = [filename for filename in os.listdir(vqvae_dir) if filename.endswith(".pth")]
    # 按照迭代次數排序
    sorted_vqvae = sorted(vqvae_files, key=lambda x: int(x.split('_')[1]))
    # 取最後一個檔案
    last_vqvae_cp = os.path.join(vqvae_dir, sorted_vqvae[-1]) if sorted_vqvae else None

    # 找到 Prior 目錄中最後一個模型檔案
    prior_files = [filename for filename in os.listdir(prior_dir) if filename.endswith(".pth")]
    # 按照準確率排序
    sorted_prior = sorted(prior_files, key=lambda x: int(x.split('_')[2]))
    # 取最後一個檔案
    last_prior_cp = os.path.join(prior_dir, sorted_prior[-1]) if sorted_prior else None

    return {
        "VQVAE": last_vqvae_cp,
        "Prior": last_prior_cp,
        "env": env,
        "device": device,
    }

def find_best_checkpoints_BC(directory):
    # 用於儲存最佳 checkpoint 的字典，key 為 model 前綴，value 為對應的最佳檔案
    best_checkpoints = {}

    # 遍歷目錄內的所有檔案
    for filename in os.listdir(directory):
        # 匹配檔案名稱格式 model_{epoch}_{loss}.pth
        match = re.match(r'model_(\d+)_(\d*\.\d*)\.pth', filename)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            
            # 提取模型名稱前綴 (可選，根據具體需求設計)
            model_key = "model"

            # 更新最佳 checkpoint
            if model_key not in best_checkpoints or loss < best_checkpoints[model_key][1]:
                best_checkpoints[model_key] = [directory+"/"+filename, loss]

    return best_checkpoints['model'][0]
def find_last_checkpoints_BC(model_dir):
    model_files = [filename for filename in os.listdir(model_dir) if filename.endswith(".pth")]
    sorted_BC = sorted(model_files, key=lambda x: int(x.split('_')[1]))
    last_BC_cp =  os.path.join(model_dir, sorted_BC[-1])
    return last_BC_cp
    
def find_best_checkpoint_SAC(directory):
    # 讀取 CSV 檔案（沒有 header）
    csv_path = os.path.join(directory, "SAC.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # 讀取 CSV，指定無 header，並手動給列命名
    df = pd.read_csv(csv_path, header=None, names=["epoch", "reward", "normalized_reward"])
    
    # 找到最大 reward 的行
    max_reward_row = df.loc[df['reward'].idxmax()]
    max_epoch = int(max_reward_row['epoch'])  # 最大 reward 對應的 epoch
    max_reward = max_reward_row['reward']  # 最大 reward
    
    print(f"Max reward: {max_reward}, Epoch: {max_epoch}")
    directory = directory + "/checkpoints"
    # 在資料夾中搜尋對應的 zip 檔案
    best_checkpoint = None
    for file in os.listdir(directory):
        # 匹配檔案命名格式: SAC_model_{epoch}_{reward}.zip
        match = re.match(r'SAC_model_(\d+)_(.*).zip', file)
        if match:
            file_epoch = int(match.group(1))
            file_reward = int(match.group(2))  # 注意 reward 是整數
            # 確認是否與找到的 epoch 和 reward 數值一致
            if file_epoch == max_epoch:
                best_checkpoint = file
                break

    if best_checkpoint:
        print(f"Best checkpoint found: {best_checkpoint}")
    else:
        print("No matching checkpoint found for the maximum reward.")
    
    return directory+"/"+best_checkpoint


def export_all_scalars_to_csv_append(folder_path, csv_filename="experiment.csv"):
    """
    1) 搜尋指定資料夾下的所有 .tfevents 檔。
    2) 讀取所有 scalar (tag, step, value)。
    3) 以 'append' (a) 模式將新資料追加到同一個 CSV 檔底部。
    """

    # 尋找所有 tfevents 檔案
    event_files = glob.glob(os.path.join(folder_path, "events.out.tfevents.*"))
    if not event_files:
        print(f"找不到任何 .tfevents 檔案於: {folder_path}")
        return

    # 預計輸出 CSV 檔的路徑
    csv_path = os.path.join(folder_path, csv_filename)

    # 檢查 csv 檔案是否已存在（用來決定是否需要寫入表頭）
    file_exists = os.path.isfile(csv_path)

    # 以 append 模式開啟 CSV
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 若是第一次建立檔案，寫入表頭
        if not file_exists:
            writer.writerow(["event_file", "tag", "step", "value"])

        # 逐一讀取每個 tfevents 檔案
        for ef in event_files:
            print(f"正在讀取檔案: {ef}")
            ea = event_accumulator.EventAccumulator(ef)
            ea.Reload()

            scalar_tags = ea.Tags().get("scalars", [])
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                for e in events:
                    writer.writerow([os.path.basename(ef), tag, e.step, e.value])

    print(f"\n已將 scalar 資料追加寫入: {csv_path}")
    
def find_best_checkpoint_O2ORL(
    folder_path, csv_filename="experiment.csv", tag="eval/d4rl_normalized_score", min_step=1e6
):
    """
    根據 CSV 檔案和條件，尋找符合條件的 checkpoint。
    1) 限制 step > min_step。
    2) 找到指定 tag (如 eval/score) 最大 value 對應的 step。
    3) 檢查同資料夾中是否存在 `checkpoint_{step}.pt`，若無則找距離最近的。

    Args:
        folder_path (str): 資料夾路徑，包含 CSV 與 checkpoint。
        csv_filename (str): CSV 檔名 (預設: "experiment.csv")。
        tag (str): 要篩選的標籤 (如: "eval/score")。
        min_step (float): 限制 step 必須大於此值 (預設: 1e6)。

    Returns:
        str: 最佳 checkpoint 的完整路徑。
    """
    # 1) 讀取 CSV 檔案
    csv_path = os.path.join(folder_path, csv_filename)
    if not os.path.isfile(csv_path):
        
        export_all_scalars_to_csv_append(folder_path,"experiment.csv")
        # raise FileNotFoundError(f"找不到 CSV 檔案: {csv_path}")
    
    data = pd.read_csv(csv_path)

    # 2) 篩選 step > min_step 並且符合指定 tag
    filtered_data = data[(data["step"] > min_step) & (data["tag"] == tag)]
    if filtered_data.empty:
        raise ValueError(f"在 step > {min_step} 且 tag 為 {tag} 的情況下，找不到任何資料。")

    # 3) 找到最大 value 的對應 step
    best_row = filtered_data.loc[filtered_data["value"].idxmax()]
    best_step = int(best_row["step"])
    print(f"找到最大值的 step: {best_step}, value: {best_row['value']}")

    # 4) 嘗試在資料夾內找到 checkpoint_{step}.pt
    checkpoint_path = os.path.join(folder_path, f"checkpoint_{best_step}.pt")
    if os.path.isfile(checkpoint_path):
        print(f"找到對應的 checkpoint 檔案: {checkpoint_path}")
        return checkpoint_path

    # 5) 如果找不到完全匹配的檔案，則找距離最近的 checkpoint
    print(f"未找到 checkpoint_{best_step}.pt，開始尋找距離最近的 checkpoint...")
    all_checkpoints = glob.glob(os.path.join(folder_path, "checkpoint_*.pt"))
    if not all_checkpoints:
        raise FileNotFoundError(f"資料夾內沒有任何符合 `checkpoint_*.pt` 的檔案。")

    # 從檔名中提取 step 並計算距離
    checkpoint_steps = []
    for ckpt in all_checkpoints:
        try:
            step = int(os.path.basename(ckpt).split("_")[1].split(".")[0])
            checkpoint_steps.append((step, ckpt))
        except (IndexError, ValueError):
            continue

    if not checkpoint_steps:
        raise FileNotFoundError(f"找不到任何符合 step 的 checkpoint 檔案。")

    # 根據 step 計算距離並排序
    closest_checkpoint = min(checkpoint_steps, key=lambda x: abs(x[0] - best_step))
    print(f"找到距離最近的 checkpoint: {closest_checkpoint[1]} (step: {closest_checkpoint[0]})")
    return closest_checkpoint[1]

def find_best_returns_rlpd(csv_file_path, env, seeds=[1, 10, 100]):
    # 1. 讀取 CSV
    df = pd.read_csv(csv_file_path)
    
    # 2. 依據 env + seed 自動組出欄位名稱
    #    欄位名稱格式為：「{env} seed {seed} - evaluation/return」
    result = {}
    for seed in seeds:
        col_name = f"{env} seed {seed} - evaluation/return"
        
        # 如果在 CSV 裡找不到這個欄位，先跳過或做提示
        if col_name not in df.columns:
            print(f"警告：找不到欄位 {col_name}")
            continue
        
        # 3. 找出最大值所在的 index
        idx_of_max = df[col_name].idxmax()
        best_step = df.loc[idx_of_max, 'Step']
        best_value = df.loc[idx_of_max, col_name]
        
        # 4. 儲存結果
        result[seed] = (best_step, best_value)
    
    return result

# env     door
# env id  door-human-v1
# sourc_path ["RL/log/paper_experiment_results_bind_source_seed1"]
# target_keywords  
def get_MAQ_agent(source_paths,env,env_id,tag="",target_keywords = ["Top1", "Top3", "MAQB", "coef0", "DSAC", "RLPD"]):
    agents = []
    for source_path in source_paths:
        result  = find_best_checkpoints(source_path,env,target_keywords)
        for seed, paths in result.items():
            print(f"Seed: {seed}")
            for directory, checkpoint in paths.items():
                print(f"Directory: {directory}")
                print(f"Checkpoint: {checkpoint}")
                if "Top" in directory or "coef0" in directory:
                    agent = FlexibleAgentInterface(MAQTopkAgent, checkpoint, env_id, tag=tag)
                elif "DSACMAQ" in directory:
                    agent = FlexibleAgentInterface(DSACMAQAgent, checkpoint, env_id, tag=tag)
                elif "RLPDMAQ" in directory:
                    agent = FlexibleAgentInterface(RLPDMAQAgent, checkpoint, env_id, tag=tag)
                else:
                    agent = FlexibleAgentInterface(MAQBAgent, checkpoint, env_id, tag=tag)
                
                agents.append(agent)
    
    return agents

def get_best_MAQ_agent(source_paths,env,env_id,agent_type,target_keywords= ["Top1", "Top3", "MAQB", "coef0", "DSAC", "RLPD"],tag="",seed="1"):
    agents = []
    for source_path in source_paths:
        result  = find_best_checkpoints(source_path,env,target_keywords,seed=seed)
        for seed, paths in result.items():
            print(f"Seed: {seed}")
            
            for directory, checkpoint in paths.items():
                print(f"Directory: {directory}")
                print(f"Checkpoint: {checkpoint}")
                if "DSACMAQ" in directory:
                    agent = FlexibleAgentInterface(agent_type, checkpoint, env_id, tag=tag)
                elif "RLPDMAQ" in directory:
                    agent = FlexibleAgentInterface(agent_type, checkpoint, env_id, tag=tag)
                
                agents.append(agent)
    
    return agents

def get_MAQ_agent_limit_j(source_paths,env,env_id,tag="",target_keywords = ["Top1", "Top3", "MAQB", "coef0", "DSAC", "RLPD"]):
    agents = []
    for source_path in source_paths:
        result  = find_best_checkpoints(source_path,env,target_keywords)
        for seed, paths in result.items():
            print(f"Seed: {seed}")
            for directory, checkpoint in paths.items():
                print(f"Directory: {directory}")
                print(f"Checkpoint: {checkpoint}")
                if "DSACMAQ" in directory:
                    agent = FlexibleAgentInterface(DSACMAQLimitJAgent, checkpoint, env_id, tag=tag)
                elif "RLPDMAQ" in directory:
                    agent = FlexibleAgentInterface(RLPDMAQLimitJAgent, checkpoint, env_id, tag=tag)
                
                agents.append(agent)
    
    return agents 

def get_Dynamic_MAQ_agent(source_paths,env,env_id,tag="",target_keywords = ["Top1", "Top3", "MAQB", "coef0"]):
    agents = []
    for source_path in source_paths:
        result  = find_best_checkpoints(source_path,env,target_keywords)
        for seed, paths in result.items():
            print(f"Seed: {seed}")
            for directory, checkpoint in paths.items():
                print(f"Directory: {directory}")
                print(f"Checkpoint: {checkpoint}")
                if "Top" in directory:
                    agent = FlexibleAgentInterface(OnlineMAQTopkAgent, checkpoint, env_id, tag=tag)
                else:
                    agent = FlexibleAgentInterface(OnlineMAQBAgent  , checkpoint, env_id, tag=tag)
                agents.append(agent)
    
    return agents 


def get_BC_agent(source_paths,env,env_id,tag=""):
    agents = []
    for source_path in source_paths:
        ckpt = find_best_checkpoints_BC(source_path)
        agents.append(FlexibleAgentInterface(BCAgent, ckpt, env_id, tag=tag))
    return agents 

def get_VQPrior_agent(source_paths,env,env_id,tag=""):
    agents = []
    for source_path in source_paths:
        ckpt = find_last_checkpoints_VQPrior(source_path[0],source_path[1], env)
        agents.append(FlexibleAgentInterface(MAQPriorAgent, ckpt, env_id, tag=tag))
    return agents

def get_AWAC_agent(source_paths,env,env_id,tag=""):
    agents = []
    for source_path in source_paths:
        
        agents.append(FlexibleAgentInterface(AWACAgent,source_path, env_id, tag=tag))
    return agents

def get_IQL_agent(source_paths,env,env_id,tag=""):
    agents = []
    for source_path in source_paths:
        
        agents.append(FlexibleAgentInterface(IQLAgent,source_path, env_id, tag=tag))
    return agents

def get_best_AWAC_agent(source_paths,env,env_id,tag=""):
    agents = []
    for source_path in source_paths:
        ckpt = find_best_checkpoint_O2ORL(source_path)
        agents.append(FlexibleAgentInterface(AWACAgent,ckpt, env_id, tag=tag))
    return agents


def get_best_IQL_agent(source_paths,env,env_id,tag=""):
    agents = []
    for source_path in source_paths:
        ckpt = find_best_checkpoint_O2ORL(source_path)
        agents.append(FlexibleAgentInterface(IQLAgent,ckpt, env_id, tag=tag))
    return agents

def get_RLPD_agent(env,env_id,tag=""):
    agents = []
    result = find_best_returns_rlpd("rlpd/rlpd.csv",env)
    for seed_val, (step, val) in result.items():
        print(f"{env} (seed={seed_val}) 的最佳分數出現在 Step={step}, 分數={val}")

        ckpt = f"rlpd/log_{env}_seed{seed_val}/s{seed_val}_0pretrain_LN/checkpoints/checkpoint_{step}"
        print(ckpt)
        agents.append(FlexibleAgentInterface(RLPDAgent, ckpt, env_id))
        
    return agents 
    

def get_best_MAQ_IQL_agent(source_paths,env,env_id,tag="", csv_filename="experiment.csv"):
    agents = []
    for source_path in source_paths:
        ckpt = find_best_checkpoint_O2ORL(source_path, csv_filename=csv_filename)
        agents.append(FlexibleAgentInterface(MAQIQLAgent,ckpt, env_id, tag=tag))
    return agents