from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork
from rl4co.models.nn.mlp import MLP


class MLPEncoder(AutoregressiveEncoder):
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
            MLP(
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