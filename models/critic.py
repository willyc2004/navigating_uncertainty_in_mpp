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
            dual_head: Optional[nn.Module] = None,
            primal_dual: Optional[bool] = False,
            n_constraints: int = 25,
            embed_dim: int = 128,
            hidden_dim: int = 512,
            decoder_layers: int = 1,
            context_embedding: Optional[nn.Module] = None,  # Context embedding for additional input
            dynamic_embedding: Optional[nn.Module] = None,  # Dynamic embedding for additional input
            critic_embedding: Optional[nn.Module] = None,  # Observation embedding for additional input
            normalization: Optional[str] = None,
            dropout_rate: Optional[float] = None,
            critic_temperature: float = 1.0,
            customized: bool = False,
            use_q_value: Optional[nn.Module] = False,
            action_dim: Optional[nn.Module] = None,
            **kwargs
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        self.context_embedding = context_embedding  # Store context_embedding
        self.dynamic_embedding = dynamic_embedding  # Store dynamic_embedding
        self.critic_embedding = critic_embedding
        self.temperature = critic_temperature
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
            for _ in range(decoder_layers - 1):
                layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate, ))

            # Output layer
            layers.append(nn.Linear(hidden_dim, 1))
            value_head = nn.Sequential(*layers)

        self.value_head = value_head

        # todo: Add Lagrangian multiplier predictor head
        if primal_dual and dual_head == None:
            # Create dual head with residual connections
            layers = [
                add_normalization_layer(normalization, embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(),  # nn.ReLU()
            ]
            for _ in range(decoder_layers - 1):
                layers.append(ResidualBlock(hidden_dim, nn.LeakyReLU(), add_normalization_layer(normalization, hidden_dim), dropout_rate, ))

            # Output layer with ReLU activation
            layers.append(nn.Linear(hidden_dim, n_constraints))
            layers.append(nn.Softplus()) # layers.append(MinClampActivation(min_val=0.001))
            dual_head = nn.Sequential(*layers)

        self.dual_head = dual_head

    def forward(self, obs: Union[Tensor, TensorDict], action:Optional=None,) -> (Tensor, Tensor):
        # Encode the input
        h, _ = self.encoder(obs)  # [batch_size, N, embed_dim] -> [batch_size, N]
        h = self.critic_embedding(h, obs)

        # State-action value
        if action is not None and hasattr(self, "state_action_layer"):
            h = torch.cat([h, action.clone().detach()], dim=-1)
            h = self.state_action_layer(h)

        # Compute the value
        if not self.customized:  # for most constructive tasks
            output = self.value_head(h).sum(dim=1, keepdims=True)  # [batch_size, N] -> [batch_size, 1]
        else:  # customized encoder and value head with hidden input
            output = self.value_head(h) # [batch_size, N] -> [batch_size, N]
        output = output / self.temperature

        # Compute the Lagrangian multiplier
        lagrangian_multiplier = self.dual_head(h) # todo: check if this also works for non primal-dual tasks
        return output, lagrangian_multiplier

def create_critic_from_actor(
    policy: nn.Module, backbone: str = "encoder", **critic_kwargs
) -> nn.Module:
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

class MinClampActivation(nn.Module):
    def __init__(self, min_val=0.001):
        super().__init__()
        self.min_val = min_val

    def forward(self, x):
        return torch.maximum(x, torch.tensor(self.min_val, device=x.device, dtype=x.dtype))
