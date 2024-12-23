import copy
from typing import Optional, Union
from tensordict import TensorDict
import torch
from torch import Tensor, nn
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

# Custom imports
from models.common import ResidualBlock, add_normalization_layer

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
            temperature: float = 1.0,
            customized: bool = False,
            use_q_value: Optional[nn.Module] = False,
            action_dim: Optional[nn.Module] = None,
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        self.context_embedding = context_embedding  # Store context_embedding
        self.temperature = temperature
        self.customized = customized

        if use_q_value:
            self.state_action_layer = nn.Linear(embed_dim+action_dim, embed_dim)

        if value_head is None:
            # Adjust embed_dim if encoder has a different dimension
            if getattr(encoder, "embed_dim", embed_dim) != embed_dim:
                log.warning(
                    f"Found encoder with different embed_dim {encoder.embed_dim} than the value head {embed_dim}. "
                    f"Using encoder embed_dim for value head."
                )
                embed_dim = getattr(encoder, "embed_dim", embed_dim)

            # Create value head with residual connections
            ffn_activation = nn.LeakyReLU() # nn.ReLU()
            norm_fn_input = add_normalization_layer(normalization, embed_dim)
            norm_fn_hidden = add_normalization_layer(normalization, hidden_dim)
            # Build the layers
            layers = [
                norm_fn_input,
                nn.Linear(embed_dim, hidden_dim),
                ffn_activation,
            ]
            # Add residual blocks
            for _ in range(num_layers - 1):
                layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate, ))

            # Output layer
            layers.append(nn.Linear(hidden_dim, 1))
            value_head = nn.Sequential(*layers)

        self.value_head = value_head

    def forward(self, obs: Union[Tensor, TensorDict], action:Optional=None, hidden=None) -> Tensor:
        """Forward pass of the critic network: encode the imput in embedding space and return the value

        Args:
            x: Input containing the environment state. Can be a Tensor or a TensorDict

        Returns:
            Value of the input state
        """
        latent, _ = self.encoder(obs)  # [batch_size, N, embed_dim] -> [batch_size, N]
        hid = self.context_embedding(obs, latent)  # Apply context embedding
        if action is not None:
            hidden = self.state_action_layer(torch.cat([hid, action], dim=-1))
        else:
            hidden = hid
        if not self.customized:  # for for most of costructive tasks
            output = self.value_head(hidden).sum(dim=1, keepdims=True)  # [batch_size, N] -> [batch_size, 1]
        else:  # customized encoder and value head with hidden input
            output = self.value_head(hidden) # [batch_size, N] -> [batch_size, N]
        output = output / self.temperature
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