from config import Config
from model import DQN, NeuralNetwork
from memory import ReplayMemory
from train import train_loop
from agent import Agent
import gymnasium as gym
import ale_py

if __name__ == "__main__":
    config = Config()
    
    gym.register_envs(ale_py)
    env = gym.make(config.env_name,  render_mode=config.render_mode, frameskip=config.frameskip)
    config.num_actions = env.action_space.n
    config.obs_shape = env.observation_space.shape
    
    memory = ReplayMemory(config.memory_size)
    model = DQN(config).to(config.device)
    agent = Agent(model, memory, config)
    
    train_loop(agent, config, env)