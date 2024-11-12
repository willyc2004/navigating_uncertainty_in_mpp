import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from tensordict import TensorDict
from torch.cuda.amp import autocast
from rl4co.utils.ops import gather_by_index

class MPPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, action_dim, env, num_constraints=5):
        super(MPPInitEmbedding, self).__init__()
        # Store environment and sequence size
        self.env = env
        self.seq_size = self.env.T * self.env.K

        # Initialize embeddings and layers
        self.cache_initialized = False  # Flag to check if cache has been created
        # Category embeddings
        self.origin_port = nn.Embedding(self.env.P, embed_dim)
        self.destination_port = nn.Embedding(self.env.P, embed_dim)
        self.cargo_class = nn.Embedding(self.env.K, embed_dim)

        # Continuous embeddings
        self.lhs_A = nn.Linear(action_dim*num_constraints, embed_dim)
        self.weight = nn.Linear(1, embed_dim)
        self.teu = nn.Linear(1, embed_dim)
        self.revenue = nn.Linear(1, embed_dim)
        self.ex_demand = nn.Linear(1, embed_dim)
        self.stdx_demand = nn.Linear(1, embed_dim)
        self.fc = nn.Linear(8 * embed_dim, embed_dim)
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim*8) # 8 is the number of embeddings

    def initialize_cache(self):
        """Initialize cache for fixed embeddings"""
        self.origin_emb_cache = self.origin_port(self.env.pol[:-1].to(torch.int64)).view(1,self.seq_size, -1)
        self.destination_emb_cache = self.destination_port(self.env.pod[:-1].to(torch.int64)).view(1,self.seq_size, -1)
        self.class_embed_cache = self.cargo_class(self.env.k[:-1].to(torch.int64)).view(1,self.seq_size, -1)
        self.weight_emb_cache = self.weight(self.env.weights[self.env.k[:-1]].view(-1, 1)).view(1,self.seq_size, -1)
        self.teu_embd_cache = self.teu(self.env.teus[self.env.k[:-1]].view(-1, 1)).view(1,self.seq_size, -1)
        self.revenue_emb_cache = self.revenue((self.env.revenues[:-1] /
                                               (self.env.pod[:-1] - self.env.pol[:-1])).view(-1, 1)).view(1,self.seq_size, -1)
        self.cache_initialized = True  # Update flag

    def forward(self, td: TensorDict):
        # Get batch size and sequence length
        if td["POL"].dim() == 2:
            batch_size, seq_size = td["POL"].shape
            expected_demand = self.ex_demand(td["expected_demand"].view(batch_size, seq_size, 1))
            std_demand = self.stdx_demand(td["std_demand"].view(batch_size, seq_size, 1))
        elif td["POL"].dim() == 3:
            batch_size, n_step, seq_size = td["POL"].shape
            expected_demand = self.ex_demand(td["expected_demand"][:,0].view(batch_size, seq_size, 1))
            std_demand = self.stdx_demand(td["std_demand"][:,0].view(batch_size, seq_size, 1))
        else:
            raise ValueError("Invalid shape for POL")

        # Initialize cache if not done yet
        if not self.cache_initialized:
            self.initialize_cache()

        # Concatenate all embeddings (using cached values for fixed ones)
        combined_emb = torch.cat([
            expected_demand, std_demand,
            self.origin_emb_cache.expand(batch_size, -1, -1),
            self.destination_emb_cache.expand(batch_size, -1, -1),
            self.class_embed_cache.expand(batch_size, -1, -1),
            self.weight_emb_cache.expand(batch_size, -1, -1),
            self.teu_embd_cache.expand(batch_size,-1, -1),
            self.revenue_emb_cache.expand(batch_size, -1, -1),
        ], dim=-1)
        # Final projection
        positional_emb = self.positional_encoding(combined_emb)
        initial_embedding = self.fc(positional_emb)
        return initial_embedding

class MPPContextEmbedding(nn.Module):
    """Context embedding of the MPP;
    - Selects the initial embedding based on the episodic step
    - Embeds the state of the MPP for the context
    """

    def __init__(self, action_dim, embed_dim, env, demand_aggregation="self_attn",):
        super(MPPContextEmbedding, self).__init__()
        self.env = env

        # Categorical embeddings with linear layers
        self.origin_location = nn.Linear(action_dim, embed_dim)
        self.destination_location = nn.Linear(action_dim, embed_dim)
        # Continuous embeddings
        self.current_demand = nn.Linear(1, embed_dim)
        self.residual_capacity = nn.Linear(action_dim, embed_dim)
        self.project_context = nn.Linear(embed_dim * 8, embed_dim, )

        # Self-attention layer
        self.demand_aggregation = demand_aggregation
        if self.demand_aggregation == "sum":
            self.observed_demand = nn.Linear(1, embed_dim)
            self.expected_demand = nn.Linear(1, embed_dim)
            self.std_demand = nn.Linear(1, embed_dim)
        elif self.demand_aggregation == "self_attn":
            self.observed_demand = SelfAttentionStateMapping(embed_dim=embed_dim,)
            self.expected_demand = SelfAttentionStateMapping(embed_dim=embed_dim,)
            self.std_demand = SelfAttentionStateMapping(embed_dim=embed_dim,)
        else:
            raise ValueError(f"Invalid demand aggregation: {demand_aggregation}")

    def forward(self,
                init_embeddings: Tensor,
                td: TensorDict):
        """Embed the context for the MPP"""

        # Get init embedding and state embedding
        select_init_embedding = gather_by_index(init_embeddings, td["episodic_step"])
        state_embedding = self._state_embedding(td)

        # Project state, concat embeddings, and project concat to output
        context_embedding = torch.cat([select_init_embedding, state_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output

    def _state_embedding(self, td):
        """Embed the state for the MPP.
        Important:
        - The state embedding size should not depend on e.g. voyage length and cargo types.
        - todo: It does depend on vessel size now, but this could be changed.
        """
        # Determine the shape based on dimensionality of current_demand
        if td["done"].dim() == 2:
            batch_size, _ = td["done"].shape
            view_transform = lambda x: x.view(batch_size, -1)
        elif td["done"].dim() == 3:
            batch_size, n_step, _ = td["done"].shape
            view_transform = lambda x: x.view(batch_size, n_step, -1)
        else:
            raise ValueError("Invalid shape of done in td")

        # Demand embeddings - Number of containers
        current_demand = self.current_demand(view_transform(td["obs"]["current_demand"]))
        if self.demand_aggregation == "sum":
            expected_demand = self.expected_demand(torch.sum(view_transform(td["obs"]["expected_demand"]), dim=-1, keepdim=True))
            std_demand = self.std_demand(torch.sum(view_transform(td["obs"]["std_demand"]), dim=-1, keepdim=True))
            observed_demand = self.observed_demand(torch.sum(view_transform(td["obs"]["observed_demand"]), dim=-1, keepdim=True))
        elif self.demand_aggregation == "self_attn":
            expected_demand = self.expected_demand(view_transform(td["obs"]["expected_demand"]))
            std_demand = self.std_demand(view_transform(td["obs"]["std_demand"]))
            observed_demand = self.observed_demand(view_transform(td["obs"]["observed_demand"]))

        # Vessel embeddings - Number of containers
        residual_capacity = self.residual_capacity(view_transform(td["obs"]["residual_capacity"]))
        origin_embed = self.origin_location(view_transform(td["state"]["agg_pol_location"]))
        destination_embed = self.destination_location(view_transform(td["state"]["agg_pod_location"]))

        # Concatenate all embeddings
        state_embed = torch.cat([
            current_demand, expected_demand, std_demand, observed_demand,
            residual_capacity, origin_embed, destination_embed
        ], dim=-1)
        return state_embed

class StaticEmbedding(nn.Module):
    # This defines shape of key, value
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td:TensorDict):
        return 0, 0, 0

class DynamicSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        _, seq_length, _ = x.size()
        position = torch.arange(seq_length, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=x.device).float() * -(math.log(10000.0) / seq_length))
        pe = torch.zeros(seq_length, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe

class SelfAttentionStateMapping(nn.Module):
    def __init__(self, embed_dim, feature_dim=1, device='cuda'):
        super(SelfAttentionStateMapping, self).__init__()
        self.feature_dim = feature_dim  # F (number of features)

        # Learnable linear transformations for Q, K, V
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)
        self.scale_factor = torch.sqrt(torch.tensor(feature_dim, device=device))
        self.final_linear = nn.Linear(feature_dim, embed_dim)

    def forward(self, X):
        # Linearly transform input tensor X to Q, K, V (applied to each sequence in the batch)
        Q = self.W_Q(X)  # (batch_size, N, F)
        K = self.W_K(X)  # (batch_size, N, F)
        V = self.W_V(X)  # (batch_size, N, F)

        # Compute attention scores (dot product Q and K, scaled by sqrt(F))
        # We need to perform batch matrix multiplication
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale_factor  # (batch_size, N, N)
        # Apply softmax to get attention weights for each batch
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, N, N)
        # Compute weighted sum of value vectors (V) for each batch
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, N, F)
        # Return enriched representations for all timesteps
        attention_output = self.final_linear(attention_output)  # (batch_size, N, F)

        return attention_output


class MultiFeatureRunningNormalization:
    def __init__(self, num_features, epsilon=1e-5):
        self.num_features = num_features
        self.count = 0
        self.mean = torch.zeros(num_features, device='cuda')
        self.var = torch.zeros(num_features, device='cuda')
        self.epsilon = epsilon

    def update(self, x):
        batch_count = x.shape[0]
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)

        delta = batch_mean - self.mean
        self.count += batch_count
        self.mean += delta * batch_count / self.count
        self.var += batch_var * batch_count + delta ** 2 * (batch_count * (self.count - batch_count) / self.count)

    def normalize(self, x):
        running_var = self.var / (self.count - 1 + self.epsilon)
        return (x - self.mean) / torch.sqrt(running_var + self.epsilon)