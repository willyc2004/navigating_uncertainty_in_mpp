import torch
from torch import nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden, init_embed = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out

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

class ResidualBlock(nn.Module):
    """Residual Block with normalization, activation, and optional dropout."""

    def __init__(self, dim, activation, norm_fn, dropout_rate=None):
        super().__init__()

        # Define layers
        self.linear = nn.Linear(dim, dim)
        self.norm = norm_fn
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        residual = x  # Save residual connection
        x = self.norm(x)  # Apply normalization
        x = self.linear(x)  # Linear transformation
        x = self.activation(x)  # Activation function
        x = self.dropout(x)  # Dropout (if any)
        return x + residual  # Add residual connection

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        if x.dim() == 2:
            return x
        elif x.dim() == 3:
            return x.permute(*self.dims)
        else:
            raise ValueError("Invalid dimensions.")

def add_normalization_layer(normalization, embed_dim):
    """
    Adds a normalization layer based on the specified type and handles input shape compatibility.
    """
    if normalization == "batch":
        return nn.Sequential(
            Permute((0, 2, 1)),  # Permute for BatchNorm1d
            nn.BatchNorm1d(embed_dim),
            Permute((0, 2, 1)),  # Revert permutation
        )
    elif normalization == "layer":
        return nn.LayerNorm(embed_dim)
    else:
        return nn.Identity()