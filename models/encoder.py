from typing import Tuple, Union
from tensordict import TensorDict
from torch import Tensor
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.zoo.am.encoder import AttentionModelEncoder

# Custom
from models.common import CustomMLP

class MLPEncoder(AutoregressiveEncoder):
    """MLP Encoder.
    First embed the input and then process it with a feedforward network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super(MLPEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )
        normalization = "Batch" if normalization == "batch" else normalization
        normalization = "Layer" if normalization == "layer" else normalization

        self.net = (
            CustomMLP(
                embed_dim,
                embed_dim,
                [feedforward_hidden] * num_layers,
                hidden_act="ReLU",
                out_act="Identity",
                input_norm=normalization,
                output_norm=normalization,
            )
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        batch_size = td.batch_size
        init_h = self.init_embedding(td)

        # Process embedding
        h = self.net(init_h.view(-1, init_h.size(-1)))

        # Return latent representation and initial embedding
        return h.view(*batch_size, -1, h.size(-1)), init_h.view(*batch_size, -1, init_h.size(-1))

class AttentionEncoder(AttentionModelEncoder):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super(AttentionEncoder, self).__init__(
            embed_dim,
            init_embedding,
            env_name,
            num_heads,
            num_layers,
            normalization,
            feedforward_hidden,
            net,
            sdpa_fn,
            moe_kwargs,
        )
        replace_norm_layers(self.net, normalization)

def replace_norm_layers(layer, norm_type="batch", track_running_stats=False):
    """
    Recursively replace normalization layers in the given layer/module.

    Args:
        layer (nn.Module): The layer/module to process.
        norm_type (str): Type of normalization to use ("Batch", "Layer", "None").
        track_running_stats (bool): Whether to track running stats for BatchNorm (if used).
    """
    for name, child in layer.named_children():
        # Check if the child is an instance of a normalization layer directly
        if isinstance(child, nn.BatchNorm1d):
            if norm_type == "batch":
                setattr(layer, name, nn.BatchNorm1d(child.num_features, track_running_stats=False))
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")

        replace_norm_layers(child, norm_type, track_running_stats)