import torch.nn as nn
import torch.optim as optim

"""
This module defines a set of PyTorch neural network classes for various applications in deep learning.

1. `SimpleNetwork`: A basic feedforward neural network with a specified number of layers and neurons, using the Adam optimizer.

2. `MoreLayersNetwork`: A deeper feedforward neural network with multiple hidden layers, designed for more complex learning tasks.

3. `SimpleNetworkWithDiffrentOptimizer`: A simple feedforward neural network with a different optimizer (Adagrad) than the default Adam optimizer.

4. `MoreLayersNetworkDiffrentOptimizer`: A deeper feedforward neural network with a different optimizer (Adagrad) for tasks requiring a different optimization approach.

5. `SimpleDiffrentLossFunction`: A simple neural network with a customizable loss function, allowing flexibility in the choice of loss metric.

6. `MoreLayerDiffrentLossFunction`: A deeper neural network with a customizable loss function, suitable for more complex learning tasks with varied loss requirements.

7. `Qnetwork`: A neural network designed for reinforcement learning tasks, specifically in the context of Q-learning, featuring customizable architecture and loss function.

These classes are intended to serve as modular components that can be adapted for specific deep learning applications by adjusting hyperparameters, architecture, and optimization strategies.
"""
class SimpleNetwork(nn.Module):

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(SimpleNetwork, self).__init__()
        
        # Set hyperparameters
        self.lr = lr
        self.loss = loss
        
        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action)
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions



class MoreLayersNetwork(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(MoreLayersNetwork, self).__init__()
        
        # Set hyperparameters
        self.lr = lr
        self.loss = loss
        
        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim),
            nn.ReLU(),
            nn.Linear(fc3_dim, n_action)
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions
    

class SimpleNetworkWithDiffrentOptimizer(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(SimpleNetworkWithDiffrentOptimizer, self).__init__()
        
        # Set hyperparameters
        self.lr = lr
        self.loss = loss
        
        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )

        # Define the optimizer (using Adagrad)
        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        
    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions


class MoreLayersNetworkDiffrentOptimizer(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(MoreLayersNetworkDiffrentOptimizer, self).__init__()
        
        # Set hyperparameters
        self.lr = lr
        self.loss = loss
        
        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim),
            nn.ReLU(),
            nn.Linear(fc3_dim, n_action),
        )

        # Define the optimizer (using Adagrad)
        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        
    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions



class SimpleDiffrentLossFunction(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(SimpleDiffrentLossFunction, self).__init__()

        # Set hyperparameters
        self.lr = lr
        self.loss = loss

        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions

class MoreLayerDiffrentLossFunction(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(MoreLayerDiffrentLossFunction, self).__init__()

        # Set hyperparameters
        self.lr = lr
        self.loss = loss

        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim),
            nn.ReLU(),
            nn.Linear(fc3_dim, n_action),
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions

class Qnetwork(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(Qnetwork, self).__init__()

        # Set hyperparameters
        self.lr = lr
        self.loss = loss

        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5),  # Add dropout with a probability of 0.5
            nn.Linear(fc1_dim, n_action),
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # Forward pass through the network
        actions = self.network(state)
        return actions


