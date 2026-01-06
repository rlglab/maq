import os
import json
import numpy as np
from abc import ABC, abstractmethod
import gc
import re
import csv
import time

# 抽象類別
class AbstractAgent(ABC):
    save_time = True
    inited = False
    time_csv = "agent_time.csv"
    
    def __init__(self, model_path, env_id):
        self.agent = self.load_agent(model_path, env_id)
    
    def _init_csv(self, header):
        """Initialize CSV file with headers"""
        if not self.inited:
            csv_exists = os.path.exists(self.time_csv)
            with open(self.time_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not csv_exists:
                    writer.writerow(header)
            self.inited = True
    
    def _save_timing_to_csv(self, inference_time):
        """Save timing data to CSV file"""
        if self.save_time:
            self._init_csv(['timestamp', 'inference_time_seconds'])
            with open(self.time_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([time.time(), inference_time])
    
    def _save_detailed_timing_to_csv(self, policy_time, vqvae_time=None, total_time=None):
        """Save detailed timing data to CSV file"""
        if self.save_time:
            self._init_csv(['timestamp', 'policy_time_seconds', 'vqvae_time_seconds', 'total_time_seconds'])
            with open(self.time_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if vqvae_time is not None and total_time is not None:
                    writer.writerow([time.time(), policy_time, vqvae_time, total_time])
                else:
                    writer.writerow([time.time(), policy_time])
    
    @abstractmethod
    def load_agent(self, model_path, env_id):
        """加載 agent 的抽象方法"""
        pass
    
    @abstractmethod
    def inference(self, state):
        """推理方法"""
        pass
    
class FlexibleAgentInterface:
    def __init__(self, agent_class, model_path, env_id, tag=""):
        """
        :param agent_class: 具體 agent 的類型 (如 MAQReg1Agent, MAQReg2Agent)
        :param model_path: 模型路徑
        :param env_id: 環境 ID
        """
        self.model_path = model_path
        self.env_id = env_id
        self.init = False
        self.agent_class = agent_class
        self.tag = tag
    def get_model_path(self):
        return self.model_path

    def get_seed(self):
        if "seed100" in self.model_path:
            return 100
        if "seed10" in self.model_path:
            return 10
        if "seed1" in self.model_path:
            return 1
        if "VQVAE" in self.model_path:
            if "seed100" in self.model_path['VQVAE']:
                return 100
            if "seed10" in self.model_path['VQVAE']:
                return 10
            if "seed1" in self.model_path['VQVAE']:
                return 1

    def inference(self, state):
        if not self.init:
            self.agent_instance = self.agent_class(self.model_path, self.env_id)
            self.init = True
        return self.agent_instance.inference(state)
    def test(self):
        print(self.agent_class,self.env_id)
        self.agent_instance = self.agent_class(self.model_path, self.env_id)
        self.init = True
        return "testing "+self.get_agent_short_name()+" OK"

    def close(self):
        if self.init == False:
            return
        del self.agent_instance
        self.agent_instance = None
        gc.collect()

        self.init = False
        
    def get_agent_short_name(self):
        if "MAQIQLearn" in self.agent_class.__name__:
            return "MAQIQLearn_seq9_k8"
        if "MAQ" in self.agent_class.__name__:
            # Extract key parameters from model path
            short_name = []
            
            # Base MAQ type
            if "MAQB" in self.agent_class.__name__ or "MAQB" in self.model_path:
                short_name.append("MAQB")
            elif "Top" in self.agent_class.__name__ or "Top" in self.model_path:
                # Check for Top1 or Top3
                if "Top1" in self.model_path:
                    short_name.append("MAQTop1")
                elif "Top3" in self.model_path:
                    short_name.append("MAQTop3") 
                else:
                    short_name.append("MAQTopk")
            elif "Prior" in self.agent_class.__name__:
                short_name.append("MAQPrior")
            elif "DSACMAQ" in self.agent_class.__name__:
                short_name.append("MAQDSAC")
            elif "RLPDMAQ" in self.agent_class.__name__:
                short_name.append("MAQRLPD")
            elif "IQL" in self.agent_class.__name__:
                short_name.append("MAQIQL") 
            else:
                short_name.append("MAQ")
            
            # Online/Offline indicator
            if "Online" in self.agent_class.__name__:
                short_name.append("online")
            
            # Extract sequence length
            seq_match = re.search(r'(?:_sq|SEQ)(\d+)', str(self.model_path))
            if seq_match:
                short_name.append(f"seq{seq_match.group(1)}")
            else:
                short_name.append("seq3") # init
            
            # Extract codebook size (k value)
            k_match = re.search(r'(?:_k|k)(\d+)', str(self.model_path))
            if k_match:
                short_name.append(f"k{k_match.group(1)}")
            else:
                short_name.append("k16") # init
            
            # Extract coefficient (c value)
            c_match = re.search(r'(?:_c|c)([\d\.]+)', str(self.model_path))
            if c_match:
                short_name.append(f"c{c_match.group(1)}")
            elif "coef0" in self.model_path:
                short_name.append("c0")
            else:
                short_name.append("c0.1") # init

            if "ranked" in self.tag:
                short_name.append("ranked")
            if "ratio" in self.tag:
                # find ratio
                ratio = re.search(r'(?:_ratio|ratio)([\d\.]+)', str(self.model_path))   
                if ratio:
                    short_name.append(f"ratio{ratio.group(1)}")
                
            return "_".join(short_name)
        else:
            # For non-MAQ agents, return simpler names
            if "AWAC" in self.agent_class.__name__:
                step = int(self.model_path.split("_")[-1].replace(".pt","")) if ".pt" in self.model_path else 0
                return f"AWAC_{'online' if step >= 1e6 else 'offline'}"
            elif "IQLearn" in self.agent_class.__name__:
                return "IQLearn"
            elif "IQL" in self.agent_class.__name__:
                step = int(self.model_path.split("_")[-1].replace(".pt","")) if ".pt" in self.model_path else 0
                return f"IQL_{'online' if step >= 1e6 else 'offline'}"
            elif "BC" in self.agent_class.__name__:
                return "BC"
            elif "SAC" in self.agent_class.__name__:
                return "SAC"
            elif "RLPD" in self.agent_class.__name__:
                return "RLPD"
            
            # Default to first part of class name
            return self.agent_class.__name__.split("Agent")[0]
            
    
    def get_agent_type(self):
        if "MAQIQLearn" in self.agent_class.__name__:
            return "MAQIQLearn" 
        if "IQLearn" in self.agent_class.__name__:
            return "IQLearn" + self.tag
        full_name = self.agent_class.__name__
        if "MAQ" in self.agent_class.__name__ and "c0.05" in self.model_path:
            full_name += "_LAMBDA_0.05"
        elif "MAQ" in self.agent_class.__name__ and "c0.5" in self.model_path:
            full_name += "_LAMBDA_0.5"
        elif "MAQ" in self.agent_class.__name__ and "c1.0" in self.model_path:
            full_name += "_LAMBDA_1.0"
        elif "MAQ" in self.agent_class.__name__ and "c0" in self.model_path:
            full_name += "_LAMBDA_0.0"
        elif "AWAC" in self.agent_class.__name__:
            step = int(self.model_path.split("_")[-1].replace(".pt",""))
            if step >= 1e6:
                full_name += "_ONLINE"
            else:
                full_name += "_OFFLINE"
        
        elif "IQL" in self.agent_class.__name__:
            step = int(self.model_path.split("_")[-1].replace(".pt",""))
            if step >= 1e6:
                full_name += "_ONLINE"
            else:
                full_name += "_OFFLINE"
        
        full_name += self.tag
        if "Top1" in self.model_path:
            full_name = full_name.replace("Topk","Top1")
        if "Top3" in self.model_path:
            full_name = full_name.replace("Topk","Top3")
        return full_name
    
