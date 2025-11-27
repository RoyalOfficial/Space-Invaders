"""
Contains all hyperparameters and other variable values.
Author: Pietro Paniccia
"""
import torch

class Config:
    env_name = "ALE/SpaceInvaders-v5"
    render_mode = "rgb_array"
    frameskip = 4
    num_actions = None
    obs_shape = None
    
    # Hyper parameters
    learning_rate = 1e-4
    gamma = 0.99
    batch_size = 32 
    memory_size = 100000 # Amount of replay memory
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1e-6 # Amount of decay for linear epsilon decay
    target_update = 1000
    num_episodes = 10000
    epsilon_decay_steps = 250000
    
    # Uses cuda if available and if not cpu
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"