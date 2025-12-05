"""
Contains utility functions. 
Currently contains a seed setter, save checkpoint, and load checkpoint.
Author: Pietro Paniccia
"""
import torch 
import random 
import numpy as np 
import os 
import pickle, gzip
from config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda_is_avilable():
        torch.cuda.manual_seed_all(seed)
        
def save_checkpoint(agent, episode, path="checkpoint.pth", include_memory=False):
    """
    Saves a checkpoint file called checkpoint.pth and current episode to resume training from.
    Optionally includes the replay memory as it takes a lot of storage and time to save.
    """
    checkpoint = {
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode": episode
    }
    
    if include_memory:
        mem = agent.memory
        max_mem = min(mem.mem_cntr, mem.capacity)
        last_n = 10000
        start = max(0,max_mem - last_n)
        checkpoint["memory"] = {
            "states": mem.states[start:max_mem],
            "actions": mem.actions[start:max_mem],
            "rewards": mem.rewards[start:max_mem],
            "next_states": mem.next_states[start:max_mem],
            "dones": mem.dones[start:max_mem],
        }
    
    torch.save(checkpoint, path)
    
    print(f"Checkpoint saved at episode {episode} -> {path}")
    
def load_checkpoint(agent, path="checkpoint.pth", device=Config.device):
    """
    Loads a checkpoint from checkpoint.pth if exists. 
    Takes in an agent as input.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]
        
                 
        if "memory" in checkpoint:
            memory_data = checkpoint["memory"]
            mem = agent.memory 
            data_len = len(memory_data["states"])
            mem.states[:data_len] = memory_data["states"]
            mem.actions[:data_len] = memory_data["actions"]
            mem.rewards[:data_len] = memory_data["rewards"]
            mem.next_states[:data_len] = memory_data["next_states"]
            mem.dones[:data_len] = memory_data["dones"]
            mem.mem_cntr = data_len
            print(f"Loaded replay memory ({data_len} samples)")
        
        print(f"Checkpoint loaded for episode {checkpoint['episode']}")       
        return checkpoint["episode"]
    
    else:
        print("No checkpoint found, starting fresh.")
        return 0