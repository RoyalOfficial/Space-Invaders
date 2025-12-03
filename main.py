"""
Main python file that is responsible for running the training for Space Invaders.
Current implementation uses a deep q learning network and isn't very good at the game.
Author: Pietro Paniccia
"""
from config import Config
from model import DQN
from memory import ReplayMemory
from train import train_loop, eval_loop
from agent import Agent
from utils import load_checkpoint
import gymnasium as gym
import ale_py

if __name__ == "__main__":
    Train = True # True = resume training, False = eval mode
    if Train:
        config = Config() # Creates a config that holds all the hyper parameters
        
        gym.register_envs(ale_py) # Makes the arcade learning environment games work
        env = gym.make(config.env_name,  render_mode=config.render_mode, frameskip=config.frameskip)
        config.num_actions = env.action_space.n
        config.obs_shape = env.observation_space.shape
        
        memory = ReplayMemory(config.memory_size, config.state_shape)
        model = DQN(config).to(config.device)
        agent = Agent(model, memory, config)
        
        train_loop(agent, config, env)
    else:
        config = Config()
        
        gym.register_envs(ale_py)
        env = gym.make(config.env_name, render_mode="human")
        config.num_actions = env.action_space.n
        config.obs_shape = env.observation_space.shape
        
        memory = ReplayMemory(config.memory_size)
        model = DQN(config).to(config.device)
        agent = Agent(model, memory, config)
        
        # Load trained weights
        loaded_episode = load_checkpoint(agent, path="checkpoint.pth", device=config.device)
        print(f"Loaded model from episode {loaded_episode}")
        
        # Eval settings
        agent.epsilon = 0.0 # No random actions
        num_eval_episodes = 5
        
        for ep in range(num_eval_episodes):
            total_reward = eval_loop(agent, env, config.device)  
            print(f"Episode {ep+1} finsihed with total reward {total_reward}")
        
        env.close()