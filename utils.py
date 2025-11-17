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
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at episode {episode} -> {path}")
    
    # Include replay memory buffer
    if include_memory:
        mem_path = path.replace(".pth","_memory.pkl.gz")
        with gzip.open(mem_path, "wb") as f:
            pickle.dump(list(agent.memory.buffer)[-10000:], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved Replay memory separately -> {mem_path}")
    
def load_checkpoint(agent, path="checkpoint.pth", device=Config.device):
    """
    Loads a checkpoint from checkpoint.pth if exists. 
    Takes in an agent as input.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]
        

        mem_path = path.replace(".pth","_memory.pkl.gz")
        if os.path.exists(mem_path):
            try: 
                with gzip.open(mem_path, "rb") as f:
                    memory_data = pickle.load(f)
                agent.memory.buffer.clear()
                agent.memory.buffer.extend(memory_data)
                print(f"Loaded replay memory ({len(memory_data)} samples from {mem_path})")
            except Exception as e:
                print(f"Could not load replay memory ({e}), continuing without it")
        return checkpoint["episode"]
    
    else:
        print("No checkpoint found, starting fresh.")
        return 0