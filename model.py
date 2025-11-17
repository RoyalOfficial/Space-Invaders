"""
Contains the various options for different types of neural networks
Author: Pietro Paniccia
"""
import torch
from torch import nn

# Sets device to gpu(cuda) if available otherwise uses the cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    """
    Basic neural network. 
    MNIST style would take a lot more neurons to work to a degree.
    Not effective for atari games
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 * 84 * 84, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class DQN(nn.Module):
    """
    Deep Q learning network that uses convolution 
    
    """
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 84, 84)
            conv_out_size = self.conv(dummy_input).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, config.num_actions)
        )
    
    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)