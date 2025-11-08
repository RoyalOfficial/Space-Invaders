import torch 
import random 
import numpy as np 
import os 
from config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda_is_avilable():
        torch.cuda.manual_seed_all(seed)
        
def save_checkpoint(agent, episode, path="checkpoint.pth", include_memory=False):
    checkpoint = {
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode": episode
    }
    
    # Include replay memory buffer
    if include_memory:
        checkpoint["memory_buffer"] = list(agent.memory.buffer)
        
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at episode {episode} -> {path}")
    
def load_checkpoint(agent, path="checkpoint.pth", device=Config.device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint["episode"]
    
        if "memory_buffer" in checkpoint:
            agent.memory.buffer.clear()
            agent.memory.buffer.extend(checkpoint["memory_buffer"])
            print(f"Loaded replay memory with {len(agent.memory.buffer)} samples")
    
    else:
        print("No checkpoint found, starting fresh.")
        return 0