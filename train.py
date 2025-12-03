"""
Contains the train loop and evalution loop.
Author: Pietro Paniccia
"""
import torch
from collections import deque 
import numpy as np
import time
import cv2 
from utils import save_checkpoint, load_checkpoint

def preprocess_frame(frame):
    """
    Takes a frame and converts it to grayscale for simplified training
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def train_loop(agent, config, env):
    """
    Training loop that takes in the agent, config, and env 
    """
    start_episode = load_checkpoint(agent, device=config.device)
    
    for episode in range(start_episode, config.num_episodes):
        state, info = env.reset()
        state = preprocess_frame(state)
        frame_stack = deque([state]*4, maxlen=4) # Stacks 4 frames
        done = False
        total_reward = 0 
        
        previous_lives = info.get("ale.lives", 3) # init previous lives 3 if not present as it is space invaders default
        while not done:
            stacked_state = np.concatenate(frame_stack, axis=0) # (shape 4,84,84)
            # Get action from agent
            action = agent.act(torch.tensor(stacked_state).unsqueeze(0).to(config.device))
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_frame(next_state)
            frame_stack.append(next_state)
            done = terminated or truncated
            stacked_next_state = np.concatenate(frame_stack, axis=0)
            shaped_reward = reward
            
            # Punish dying 
            current_lives = info.get("ale.lives", previous_lives) # Defaults previous if not present
            if current_lives < previous_lives:
                shaped_reward -= 0.5
            else:
                shaped_reward += 0.0001 # small reward for living
            previous_lives = current_lives
            
            # Slight encouragement for moving 
            if action in [2,3]:
                shaped_reward += 0.001 # moved
            
            # Store transition in replay memory 
            agent.memory.push(stacked_state, action, shaped_reward, stacked_next_state, done)
            
            # Learn from replay buffer
            if len(agent.memory) > 50000:
                agent.learn()
            
            # Decay epsilon once per step
            agent.update_epsilon()
            
            # Prepare next iteration
            stacked_state = stacked_next_state
            total_reward += shaped_reward
            
        print(f"Epsiode {episode+1}/{config.num_episodes} - Total Reward: {total_reward:.2f}")
        
        with open("exampleout.txt", "a") as f:
            f.write(f"Episode {episode+1}/{config.num_episodes} - Total Reward: {total_reward:.2f}\n")

        # Save every 500 episodes include_memory decides if replay memory included
        if (episode + 1) % 500 == 0:
            save_checkpoint(agent, episode + 1, include_memory=True)
            
    env.close()
    
def eval_loop(agent, env, device, frame_stack_size=4, sleep=0.02):
    """
    Evalution loop for a trained model. 
    Removes the randomness to test if the model has learned the game well.
    """
    frame_stack = deque(maxlen=frame_stack_size)
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Initialize frame stack
    frame = preprocess_frame(obs)[0, :, :]
    
    for _ in range(frame_stack_size):
        frame_stack.append(frame)
        
    while not done:
        state = np.stack(frame_stack, axis=0)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                
        action = agent.act(state)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        frame_stack.append(preprocess_frame(obs)[0, :, :])
        if sleep:
            time.sleep(sleep)
        
    return total_reward