"""
Contains class for the replay memory. 
Saves the previous state, action, reward, next_state, and done.
Author: Pietro Paniccia
"""
from collections import deque
import random 

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Saves as zip file to reduce storage usage
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones 
    
    def __len__(self):
        return len(self.buffer)