# Import libraries and modules
from typing import Tuple, Callable
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

@dataclass
class PrecomputedCache:
    init_embeddings: Tensor
    graph_context: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")


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
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Layer Normalization
        self.q_layer_norm = nn.LayerNorm(embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer_norm = nn.LayerNorm(embed_dim*2)

        # Configurable Feedforward Network with Variable Hidden Layers
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Default hidden dimension is 4 times the embed_dim

        feed_forward_layers = []
        for _ in range(num_hidden_layers):
            feed_forward_layers.append(nn.Linear(embed_dim, hidden_dim))
            # feed_forward_layers.append(nn.LayerNorm(hidden_dim))
            feed_forward_layers.append(nn.ReLU())
            feed_forward_layers.append(nn.Dropout(dropout_rate))
            feed_forward_layers.append(nn.Linear(hidden_dim, embed_dim))
            # feed_forward_layers.append(nn.LayerNorm(embed_dim))

        self.feed_forward = nn.Sequential(*feed_forward_layers)

        # Projection Layers
        self.project_embeddings_kv = nn.Linear(embed_dim, embed_dim * 3)  # For key, value, and logit
        self.output_projection = nn.Linear(embed_dim * 2, action_size * 2)  # For mean and log_std

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
        # Compute query, key, and value for the attention mechanism
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)
        glimpse_q = self._compute_q(cached, td)
        glimpse_q = self.q_layer_norm(glimpse_q)

        # Multi-head Attention
        attn_output, _ = self.attention(glimpse_q, glimpse_k, glimpse_v)
        attn_output = self.dropout(attn_output)
        attn_output = self.attn_layer_norm(attn_output + glimpse_q)

        # # Feedforward Network with Residual Connection
        ffn_output = self.feed_forward(attn_output)
        ffn_output = self.ffn_layer_norm(ffn_output + attn_output)

        # Pointer mechanism: compute pointer logits (scores) over the sequence
        # The pointer logits are used to select an index (action) from the input sequence
        pointer_logits = torch.matmul(ffn_output, glimpse_k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        pointer_probs = F.softmax(pointer_logits, dim=-1)
        # Compute the context vector (weighted sum of values based on attention probabilities)
        pointer_output = torch.matmul(pointer_probs, glimpse_v)  # [batch_size, seq_len, hidden_dim]

        # Combine pointer_output with ffn_output to feed into the output projection
        combined_output = torch.cat([ffn_output, pointer_output], dim=-1)
        combined_output = self.output_layer_norm(combined_output)

        # Project logits to mean and log_std logits (use softplus)
        logits = self.output_projection(combined_output).view(td.batch_size[0], self.action_size, 2)
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
        super(MLPDecoderWithCache, self).__init__()
        self.context_embedding = context_embedding
        self.dynamic_embedding = dynamic_embedding
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )
        self.state_size = state_size
        self.action_size = action_size

        # Create MLP layers with ReLU activation, add some layer parameter
        num_layers = num_hidden_layers
        layers = [nn.Linear(embed_dim, embed_dim), nn.ReLU()] * num_layers
        self.mlp = nn.Sequential(*layers)

        # Output projection to action size
        self.output_projection = nn.Linear(embed_dim, action_size*2)

    def forward(self, td: TensorDict, cached: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor,Tensor]:
        # Get precomputed (cached) embeddings
        init_embeds_cache, _ = cached.init_embeddings, cached.graph_context

        # Compute step context
        step_context = self.context_embedding(init_embeds_cache, td)

        # Compute mask and logits
        logits = self.mlp(step_context)

        # Project logits to mean and log_std logits (use softplus)
        logits = self.output_projection(logits).view(td.batch_size[0], self.action_size, 2)
        output_logits = F.softplus(logits)
        return output_logits, td["action_mask"]

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