from typing import List, Union
import torch
from torch import nn

from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.ops import Normalization
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden, init_embed = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out

class CustomMLP(MLP):
    @staticmethod
    def _get_norm_layer(norm_method, dim):
        if norm_method == "Batch":
            # Use BatchNorm1d but ensure compatibility with vectorized operations
            in_norm = nn.BatchNorm1d(dim, track_running_stats=False)
        elif norm_method == "Layer":
            # Use LayerNorm
            in_norm = nn.LayerNorm(dim)
        elif norm_method == "None":
            # No normalization
            in_norm = nn.Identity()
        else:
            raise RuntimeError(
                f"Not implemented normalization layer type {norm_method}"
            )
        return in_norm


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
            nn.BatchNorm1d(embed_dim, track_running_stats=False),
            Permute((0, 2, 1)),  # Revert permutation
        )
    elif normalization == "layer":
        return nn.LayerNorm(embed_dim)
    else:
        return nn.Identity()