"""
Contains class for the replay memory. 
Saves the previous state, action, reward, next_state, and done.
Author: Pietro Paniccia
"""
from collections import deque
import random 
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.mem_cntr = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = float(done)
        
        self.mem_cntr += 1
        
    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        return (
            self.states[batch],
            self.actions[batch],
            self.rewards[batch],
            self.next_states[batch],
            self.dones[batch]
        )
    
    def __len__(self):
        return min(self.mem_cntr, self.capacity)