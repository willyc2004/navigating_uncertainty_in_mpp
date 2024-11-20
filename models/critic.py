import copy

from typing import Optional, Union

from tensordict import TensorDict
from torch import Tensor, nn

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CriticNetwork(nn.Module):
    """Create a critic network given an encoder (e.g., as the one in the policy network)
    with a value head to transform the embeddings to a scalar value.

    Args:
        encoder: Encoder module to encode the input
        value_head: Value head to transform the embeddings to a scalar value
        embed_dim: Dimension of the embeddings of the value head
        hidden_dim: Dimension of the hidden layer of the value head
        context_embedding: Embedding layer for context (e.g., graph or environment context)
        customized: Flag to use a customized flow for encoding and value head
    """

    def __init__(
            self,
            encoder: nn.Module,
            value_head: Optional[nn.Module] = None,
            embed_dim: int = 128,
            hidden_dim: int = 512,
            num_layers: int = 1,
            context_embedding: Optional[nn.Module] = None,  # Add context_embedding
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        self.context_embedding = context_embedding  # Store context_embedding

        if value_head is None:
            # Check if embed_dim of encoder is different, if so, use it
            if getattr(encoder, "embed_dim", embed_dim) != embed_dim:
                log.warning(
                    f"Found encoder with different embed_dim {encoder.embed_dim} than the value head {embed_dim}. Using encoder embed_dim for value head."
                )
                embed_dim = getattr(encoder, "embed_dim", embed_dim)

            # Create value head
            value_head = nn.Sequential(
                # nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                *[
                    nn.Sequential(
                        # nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                    )
                    for _ in range(num_layers - 1)
                ],
                nn.Linear(hidden_dim, 1),  # Output layer
            )
        self.value_head = value_head

    def forward(self, x: Union[Tensor, TensorDict], ) -> Tensor:
        """Forward pass of the critic network: encode the input in embedding space and return the value

        Args:
            x: Input containing the environment state. Can be a Tensor or a TensorDict

        Returns:
            Value of the input state
        """
        # Step 1: Encode input using the encoder
        h, _ = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]

        # Step 2: If context_embedding exists, compute and apply it
        if self.context_embedding is not None:
            h = self.context_embedding(h, x)

        # Step 3: Compute value using value_head
        output = self.value_head(h) # [batch_size, N]
        return output


def create_critic_from_actor(
    policy: nn.Module, backbone: str = "encoder", **critic_kwargs
):
    # we reuse the network of the policy's backbone, such as an encoder
    encoder = getattr(policy, backbone, None)
    if encoder is None:
        raise ValueError(
            f"CriticBaseline requires a backbone in the policy network: {backbone}"
        )
    critic = CriticNetwork(copy.deepcopy(encoder), **critic_kwargs).to(
        next(policy.parameters()).device
    )
    return critic
