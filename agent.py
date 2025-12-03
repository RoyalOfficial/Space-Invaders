"""
Contains code for the agent and contains learning algorithm. 
Also contains the way the action chooses an action.
Author: Pietro Paniccia
"""
import random
import torch
import numpy as np
import copy

class Agent:
    def __init__(self, model, memory, config):
        """
        Initializes the agent with given model, memory, and config
        """
        self.model = model
        self.memory = memory 
        self.config = config
        self.epsilon = config.epsilon_start
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.target_model = copy.deepcopy(model)
        self.learn_step = 0
    
    def act(self, state):
        """
        Either picks move based on the state or a random move based on current epsilon
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.config.num_actions - 1)
        with torch.no_grad():
            q_vals = self.model(state)
            return torch.argmax(q_vals).item()
        
    def update_epsilon(self):
        # Decay epsilon - expoenational decays too fast
        # self.epsilon = max(self.config.epsilon_min, self.epsilon * 0.99995)
        
        # Linear decay epsilon
        decay_rate = (self.config.epsilon_start - self.config.epsilon_mid) / self.config.epsilon_decay_steps
        self.epsilon = max(self.config.epsilon_min, self.epsilon - decay_rate)
        
    def learn(self):
        """
        Contains the learning algorithm for the agent
        """
        # Only learn if enough samples
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample batch from Replay Memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.config.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.config.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.config.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.config.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.config.device)
        
        # Compute Q(s, a)
        q_vals = self.model(states).gather(1, actions)
        
        # Compute max Q(s', a') for next states
        with torch.no_grad():
            max_next_q_vals = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q_vals = rewards + self.config.gamma * max_next_q_vals * (1 - dones)
            
        # Update target network 
        self.learn_step += 1
        if self.learn_step % self.config.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        # Compute loss
        loss = self.criterion(q_vals, target_q_vals)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        