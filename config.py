import torch
import gymnasium as gym

class Config:
    env_name = "ALE/SpaceInvaders-v5"
    render_mode = "rgb_array"
    frameskip = 4
    #env = gym.make(env_name, render_mode=render_mode)
    #num_actions = env.action_space.n
    #obs_shape = env.observation_space.shape
    num_actions = None
    obs_shape = None
    
    # Hyper parameters
    learning_rate = 1e-4
    gamma = 0.99
    batch_size = 32 
    memory_size = 100000
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1e-6
    target_update = 1000
    num_episodes = 10000
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"