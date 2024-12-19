import math
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from rl4co.utils.ops import gather_by_index

class MPPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, action_dim, env, num_constraints=5):
        super(MPPInitEmbedding, self).__init__()
        # Store environment and sequence size
        self.env = env
        self.seq_size = self.env.T * self.env.K

        # Embedding layers
        num_embeddings = 5  # Number of embeddings
        self.fc = nn.Linear(num_embeddings, embed_dim)
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim)

    def _combine_cargo_parameters(self, batch_size):
        """Prepare cargo parameters for init embedding"""
        norm_features = {
            "pol": (self.env.pol[:-1].clone() / self.env.P).view(1, -1, 1).expand(batch_size, -1, -1),
            "pod": (self.env.pod[:-1].clone() / self.env.P).view(1, -1, 1).expand(batch_size, -1, -1),
            "weights": (self.env.weights[self.env.k[:-1]].clone() / self.env.weights[self.env.k[:-1]].max()).view(1, -1, 1).expand(batch_size, -1, -1),
            "teus": (self.env.teus[self.env.k[:-1]].clone() / self.env.teus[self.env.k[:-1]].max()).view(1, -1, 1).expand(batch_size, -1, -1),
            "revenues": (self.env.revenues[:-1].clone() / self.env.revenues[:-1].max()).view(1, -1, 1).expand(batch_size, -1, -1),
        }
        return norm_features


    def forward(self, obs: Tensor):
        # todo: possibly add more exp, std of demand
        batch_size = obs.shape[0]
        cargo_parameters = self._combine_cargo_parameters(batch_size=batch_size)
        combined_input = torch.cat([*cargo_parameters.values(),], dim=-1)
        combined_emb = self.fc(combined_input)

        # Positional encoding
        # todo: add positional encoding
        # initial_embedding = self.positional_encoding(combined_emb)
        initial_embedding = combined_emb
        return initial_embedding

class MPPContextEmbedding(nn.Module):
    """Context embedding of the MPP;
    - Selects the initial embedding based on the episodic step
    - Embeds the state of the MPP for the context
    """

    def __init__(self, obs_dim, embed_dim, env, demand_aggregation="full",):
        super(MPPContextEmbedding, self).__init__()
        self.env = env
        self.seq_size = self.env.T * self.env.K
        self.project_context = nn.Linear(embed_dim + obs_dim, embed_dim,) # embed_dim +

        # todo: give options for different demand aggregation methods; e.g. sum, self-attention
        self.demand_aggregation = demand_aggregation
        if self.demand_aggregation == "sum":
            self.observed_demand = nn.Linear(1, embed_dim)
            self.expected_demand = nn.Linear(1, embed_dim)
            self.std_demand = nn.Linear(1, embed_dim)
        elif self.demand_aggregation == "full":
            self.observed_demand = nn.Linear(self.seq_size, embed_dim)
            self.expected_demand = nn.Linear(self.seq_size, embed_dim)
            self.std_demand = nn.Linear(self.seq_size, embed_dim)
        elif self.demand_aggregation == "self_attn":
            self.observed_demand = SelfAttentionStateMapping(embed_dim=embed_dim,)
            self.expected_demand = SelfAttentionStateMapping(embed_dim=embed_dim,)
            self.std_demand = SelfAttentionStateMapping(embed_dim=embed_dim,)
        else:
            raise ValueError(f"Invalid demand aggregation: {demand_aggregation}")

    def forward(self, obs: Tensor, latent_state: Tensor):
        """Embed the context for the MPP"""
        # Get relevant init embedding (first element of obs)
        time = (obs[..., 0] * self.seq_size).long()
        select_init_embedding = gather_by_index(latent_state, time)

        # Project state, concat embeddings, and project concat to output
        context_embedding = torch.cat([obs, select_init_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output

def reorder_demand(demand, tau, k, T, K, batch_size):
    """Reorder demand to match the episode ordering"""
    return demand.view(*batch_size, T, K)[..., tau[:-1], k[:-1]]

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
        if x.dim() == 3:
            _, seq_length, _ = x.size()
        else:
            _, seq_length, _, _ = x.size()
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
        # Reshape input if it includes multiple steps
        if X.dim() == 4:  # Expected shape [batch, n_step, seq, feature_dim]
            batch_size, n_step, seq_len, _ = X.shape
            X = X.view(batch_size * n_step, seq_len, self.feature_dim)

        # Linearly transform input tensor X to Q, K, V
        Q = self.W_Q(X)  # (batch_size * n_step, seq_len, F)
        K = self.W_K(X)  # (batch_size * n_step, seq_len, F)
        V = self.W_V(X)  # (batch_size * n_step, seq_len, F)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1,-2)) / self.scale_factor  # (batch_size * n_step, seq_len, seq_len)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (batch_size * n_step, seq_len, seq_len)

        # Compute weighted sum of V
        attention_output = torch.matmul(attention_weights, V)  # (batch_size * n_step, seq_len, F)
        attention_output = self.final_linear(attention_output)  # (batch_size * n_step, seq_len, embed_dim)

        # Restore original batch and step dimensions if reshaped
        if X.dim() == 4:
            attention_output = attention_output.view(batch_size, n_step, seq_len, -1)
            attention_weights = attention_weights.view(batch_size, n_step, seq_len, seq_len)

        return attention_output, attention_weights