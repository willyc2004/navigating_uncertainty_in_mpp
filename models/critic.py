import copy
from typing import Optional, Union
from tensordict import TensorDict
from torch import Tensor, nn
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

# Custom imports
from decoder import ResidualBlock

class CriticNetwork(nn.Module):
    """Create a critic network with an encoder and a value head to transform the embeddings to a scalar value."""
    def __init__(
            self,
            encoder: nn.Module,
            value_head: Optional[nn.Module] = None,
            embed_dim: int = 128,
            hidden_dim: int = 512,
            num_layers: int = 1,
            context_embedding: Optional[nn.Module] = None,  # Context embedding for additional input
            normalization: Optional[str] = None,
            dropout_rate: Optional[float] = None,
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        self.context_embedding = context_embedding  # Store context_embedding

        if value_head is None:
            # Adjust embed_dim if encoder has a different dimension
            if getattr(encoder, "embed_dim", embed_dim) != embed_dim:
                log.warning(
                    f"Found encoder with different embed_dim {encoder.embed_dim} than the value head {embed_dim}. "
                    f"Using encoder embed_dim for value head."
                )
                embed_dim = getattr(encoder, "embed_dim", embed_dim)

            # Create value head with residual connections
            ffn_activation = nn.LeakyReLU()
            norm_dict = {
                'layer': nn.LayerNorm,
            }
            assert normalization != 'batch', "BatchNorm1d is not supported in the critic network"
            norm_fn = norm_dict.get(normalization, nn.Identity)

            # Build the value head
            layers = [
                norm_fn(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                ffn_activation,
            ]

            # Add residual blocks
            for _ in range(num_layers - 1):
                layers.append(ResidualBlock(hidden_dim, norm_fn, ffn_activation, dropout_rate))

            # Output layer
            layers.append(nn.Linear(hidden_dim, 1))

            value_head = nn.Sequential(*layers)

        self.value_head = value_head

    def forward(self, x: Union[Tensor, dict]) -> Tensor:
        """Forward pass of the critic network: encode the input and return the value."""
        # Step 1: Encode input using the encoder
        h, _ = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]

        # Step 2: If context_embedding exists, compute and apply it
        if self.context_embedding is not None:
            h = self.context_embedding(h, x)

        # Step 3: Compute value using value_head
        output = self.value_head(h)  # [batch_size, N]
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
