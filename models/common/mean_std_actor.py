import torch
from torch import nn
import torch.nn.functional as F

class ActorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ActorMLP, self).__init__()
        # Ensure hidden_dims is a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Combine all layers into a Sequential module
        self.hidden_layers = nn.Sequential(*layers)

        # Output layers for mean and standard deviation
        self.mean = nn.Linear(hidden_dims[-1], output_dim)
        self.std = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Pass input through hidden layers
        x = self.hidden_layers(x)
        x = F.relu(x)  # Apply ReLU to the output of the last hidden layer

        # Compute mean and standard deviation
        mean = self.mean(x)
        std = torch.exp(self.std(x))  # Ensure std is positive

        return mean, std
