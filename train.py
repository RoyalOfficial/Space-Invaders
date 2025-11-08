import torch
from collections import deque 
import numpy as np
import cv2 
from utils import save_checkpoint, load_checkpoint

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def train_loop(agent, config, env):
    start_episode = load_checkpoint(agent, device=config.device)
    
    for episode in range(start_episode, config.num_episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        frame_stack = deque([state]*4, maxlen=4) # Stacks 4 frames
        done = False
        total_reward = 0 
        
        while not done:
            stacked_state = np.concatenate(frame_stack, axis=0) # (shape 4,84,84)
            # Get action from agent
            action = agent.act(torch.tensor(stacked_state).unsqueeze(0).to(config.device))
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            frame_stack.append(next_state)
            done = terminated or truncated
            stacked_next_state = np.concatenate(frame_stack, axis=0)
            
            # Store transition in replay memory 
            agent.memory.push(stacked_state, action, reward, stacked_next_state, done)
            
            # Learn from replay buffer
            agent.learn()
            
            # Prepare next iteration
            stacked_state = stacked_next_state
            total_reward += reward
            
        print(f"Epsiode {episode+1}/{config.num_episodes} - Total Reward: {total_reward:.2f}")
        
        with open("exampleout.txt", "a") as f:
            f.write(f"Episode {episode+1}/{config.num_episodes} - Total Reward: {total_reward:.2f}\n")
       
        if (episode + 1) % 50 == 0:
            save_checkpoint(agent, episode + 1, include_memory=False)
            
    env.close()