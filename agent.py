import random
import torch
import numpy as np

class Agent:
    def __init__(self, model, memory, config):
        self.model = model
        self.memory = memory 
        self.config = config
        self.epsilon = config.epsilon_start
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.learning_rate)
        self.criterion = torch.nn.MSELoss()
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.config.num_actions - 1)
        with torch.no_grad():
            q_vals = self.model(state)
            return torch.argmax(q_vals).item()
    
    def learn(self):
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
            max_next_q_vals = self.model(next_states).max(1, keepdim=True)[0]
            target_q_vals = rewards + self.config.gamma * max_next_q_vals * (1 - dones)
            
        # Compute loss
        loss = self.criterion(q_vals, target_q_vals)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * 0.99995)