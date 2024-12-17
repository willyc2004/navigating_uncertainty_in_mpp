# Import libraries and modules
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict
from einops import rearrange
import math

# Custom rl4co modules
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import batchify, unbatchify
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from environment.embeddings import MPPInitEmbedding, StaticEmbedding, MPPContextEmbedding
from torch.cuda.amp import autocast
from models.projection_n_step_ppo import check_for_nans, recursive_check_for_nans
from models.common.ffn_block import ResidualBlock, add_normalization_layer

@dataclass
class PrecomputedCache:
    init_embeddings: Tensor
    graph_context: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

class AttentionDecoderWithCache(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 embed_dim: int,
                 total_steps: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 num_hidden_layers: int = 3,  # Number of hidden layers
                 hidden_dim: int = None,  # Dimension for hidden layers (defaults to 4 * embed_dim)
                 normalization: Optional[str] = None,  # Type of normalization layer
                 init_embedding=None,
                 context_embedding=None,
                 dynamic_embedding=None,
                 linear_bias: bool = False,
                 max_context_len: int = 256,
                 use_graph_context: bool = False,
                 mask_inner: bool = False,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 sdpa_fn: Callable = None):
        super(AttentionDecoderWithCache, self).__init__()

        self.context_embedding = context_embedding
        self.dynamic_embedding = dynamic_embedding
        self.is_dynamic_embedding = not isinstance(self.dynamic_embedding, StaticEmbedding)
        self.state_size = state_size
        self.action_size = action_size
        self.total_steps = total_steps
        self.total_step_range = torch.arange(total_steps, device="cuda")
        self.dropout = nn.Dropout(dropout_rate)

        # Attention and Feedforward Layers
        self.attention = FP32Attention(embed_dim, num_heads, batch_first=True)

        # Layer Normalization
        norm_dict = {
            "layer": FP32LayerNorm,
            "batch": nn.BatchNorm1d,
        }
        norm = norm_dict.get(normalization, nn.Identity)
        self.q_layer_norm = norm(embed_dim)
        self.attn_layer_norm = norm(embed_dim)
        self.ffn_layer_norm = norm(embed_dim)
        self.output_layer_norm = norm(embed_dim*2)

        # Configurable Feedforward Network with Variable Hidden Layers
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Default hidden dimension is 4 times the embed_dim

        feed_forward_layers = []
        ffn_activation = nn.LeakyReLU() # nn.GELU(), nn.ReLU(), nn.SiLU(), nn.LeakyReLU()
        for _ in range(num_hidden_layers):
            feed_forward_layers.append(nn.Linear(embed_dim, hidden_dim))
            # feed_forward_layers.append(nn.LayerNorm(hidden_dim))
            feed_forward_layers.append(ffn_activation)
            feed_forward_layers.append(nn.Dropout(dropout_rate))
            feed_forward_layers.append(nn.Linear(hidden_dim, embed_dim))
            # feed_forward_layers.append(nn.LayerNorm(embed_dim))

        self.feed_forward = nn.Sequential(*feed_forward_layers)

        # Projection Layers
        self.project_embeddings_kv = nn.Linear(embed_dim, embed_dim * 3)  # For key, value, and logit
        self.output_projection = nn.Linear(embed_dim * 2, action_size * 2)  # For mean and log_std
        self.output_projection2 = nn.Linear(embed_dim, action_size * 2)  # For mean and log_std

        # Optionally, use graph context
        self.use_graph_context = use_graph_context

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict) -> Tensor:
        """Compute query of static and context embedding for the attention mechanism."""
        node_embeds_cache = cached.init_embeddings
        glimpse_q = self.context_embedding(node_embeds_cache, td)
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute key, value, and logit_key of static embedding for the attention mechanism."""
        return cached.glimpse_key, cached.glimpse_val, cached.logit_key

    def forward(self, td: TensorDict, cached: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor,Tensor]:
        # Multi-head Attention block
        # Compute query, key, and value for the attention mechanism
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)
        glimpse_q = self._compute_q(cached, td)
        glimpse_q = self.q_layer_norm(glimpse_q)
        attn_output, _ = self.attention(glimpse_q, glimpse_k, glimpse_v)

        # Feedforward Network with Residual Connection block
        attn_output = self.attn_layer_norm(attn_output + glimpse_q)
        ffn_output = self.feed_forward(attn_output)

        # Pointer Attention block
        # Compute pointer logits (scores) over the sequence
        # The pointer logits are used to select an index (action) from the input sequence
        ffn_output = self.ffn_layer_norm(ffn_output + attn_output)
        pointer_logits = torch.matmul(ffn_output, glimpse_k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        pointer_probs = F.softmax(pointer_logits, dim=-1)
        # Compute the context vector (weighted sum of values based on attention probabilities)
        pointer_output = torch.matmul(pointer_probs, glimpse_v)  # [batch_size, seq_len, hidden_dim]
        # Combine pointer_output with ffn_output to feed into the output projection
        combined_output = torch.cat([ffn_output, pointer_output], dim=-1)

        # Output block with softplus activation
        combined_output = self.output_layer_norm(combined_output)
        if td["done"].dim() == 2:
            view_transform = lambda x: x.view(td.batch_size[0], self.action_size, 2)
        elif td["done"].dim() == 3:
            view_transform = lambda x: x.view(td.batch_size[0], -1, self.action_size, 2)
        else:
            raise ValueError("Invalid dimension for done tensor.")
        logits = view_transform(self.output_projection(combined_output))
        output_logits = F.softplus(logits)
        return output_logits, td["action_mask"]

    def pre_decoder_hook(self, td: TensorDict, env, embeddings: Tensor, num_starts: int = 0):
        return td, env, self._precompute_cache(embeddings, num_starts)

    def _precompute_cache(self, embeddings: Tensor, num_starts: int = 0) -> PrecomputedCache:
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_embeddings_kv(embeddings).chunk(3, dim=-1)

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            init_embeddings=embeddings,
            graph_context=torch.tensor(0),  # Placeholder, can be extended if graph context is used
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

class MLPDecoderWithCache(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 embed_dim: int,
                 total_steps: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 num_hidden_layers: int = 3,  # Number of hidden layers
                 hidden_dim: int = None,  # Dimension for hidden layers (defaults to 4 * embed_dim)
                 normalization: Optional[str] = None,  # Type of normalization layer
                 init_embedding=None,
                 context_embedding=None,
                 dynamic_embedding=None,
                 temperature: float = 1.0,
                 linear_bias: bool = False,
                 max_context_len: int = 256,
                 use_graph_context: bool = False,
                 mask_inner: bool = False,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 sdpa_fn: Callable = None,):
        super(MLPDecoderWithCache, self).__init__()
        self.context_embedding = context_embedding
        self.dynamic_embedding = dynamic_embedding
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )
        self.state_size = state_size
        self.action_size = action_size

        # Create policy MLP
        ffn_activation = nn.LeakyReLU()
        norm_fn_input = add_normalization_layer(normalization, embed_dim)
        norm_fn_hidden = add_normalization_layer(normalization, hidden_dim)
        # Build the layers
        layers = [
            norm_fn_input,
            nn.Linear(embed_dim, hidden_dim),
            ffn_activation,
        ]
        # Add residual blocks
        for _ in range(num_hidden_layers - 1):
            layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate,))

        # Output layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.policy_mlp = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_size)
        self.std_head = nn.Linear(hidden_dim, action_size)

        # Temperature for the policy
        self.temperature = temperature

    def forward(self, obs, hidden) -> Tensor:
        # Context embedding
        context = self.context_embedding(obs, hidden)

        # Compute mask and logits
        hidden = self.policy_mlp(context)
        mean = self.mean_head(hidden)
        # todo: add std head
        # std = F.softplus(self.std_head(hidden))
        # output_logits = torch.stack([mean, std], dim=-1)
        # todo: add mask
        return mean/self.temperature

    def pre_decoder_hook(self, td, env, embeddings, num_starts: int = 0):
        return td, env, self._precompute_cache(embeddings, num_starts)

    def _precompute_cache(self, embeddings: Tensor, num_starts: int = 0) -> PrecomputedCache:
        return PrecomputedCache(
            init_embeddings=embeddings,
            graph_context=None,
            glimpse_key=None,
            glimpse_val=None,
            logit_key=None,
        )

class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm using FP32 computation and FP16 storage."""
    def __init__(self, normalized_shape, eps=1e-4):
        super(FP32LayerNorm, self).__init__(normalized_shape, eps=eps)

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        normalized_output = super(FP32LayerNorm, self).forward(x_fp32)
        return normalized_output.to(x.dtype)

class FP32Attention(nn.MultiheadAttention):
    """Multi-head Attention using FP32 computation and FP16 storage, with adjusted initialization."""

    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            print(f"Warning: embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads}). Adjusting embed_dim.")
            embed_dim = (embed_dim // num_heads) * num_heads

        # Call superclass with adjusted embed_dim
        super(FP32Attention, self).__init__(embed_dim, num_heads, **kwargs)
        self.embed_dim = embed_dim  # Store adjusted embed_dim if it was changed
        self.num_heads = num_heads  # Ensure num_heads is consistent

    def forward(self, query, key, value, **kwargs):
        # Cast inputs to FP32 for stable attention computation
        query_fp32 = query.float()
        key_fp32 = key.float()
        value_fp32 = value.float()

        # Ensure head_dim is consistent
        head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim == head_dim * self.num_heads, (
            f"embed_dim ({self.embed_dim}) is not compatible with num_heads ({self.num_heads})."
        )

        # Perform multi-head attention in FP32 and cast back to input dtype
        attn_output_fp32, attn_weights_fp32 = super(FP32Attention, self).forward(query_fp32, key_fp32, value_fp32)
        attn_output = attn_output_fp32.to(query.dtype)
        attn_weights = attn_weights_fp32.to(query.dtype)
        return attn_output, attn_weights