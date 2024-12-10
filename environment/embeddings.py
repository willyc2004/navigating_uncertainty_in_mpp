import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from tensordict import TensorDict
from torch.cuda.amp import autocast
from rl4co.utils.ops import gather_by_index
from models.projection_n_step_ppo import check_for_nans, recursive_check_for_nans

class MPPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, action_dim, env, num_constraints=5):
        super(MPPInitEmbedding, self).__init__()
        # Store environment and sequence size
        self.env = env
        self.seq_size = self.env.T * self.env.K

        # Embedding layers
        num_embeddings = 7  # Number of embeddings
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


    def forward(self, td: TensorDict):
        # Normalize demand (based on teu and total capacity)
        # norm_cargo = (self.env.teus[self.env.k[:-1]] / self.env.total_capacity).view(1, -1, 1)
        if td["obs"]["expected_demand"].dim() == 2:
            expected_demand = td["obs"]["expected_demand"].unsqueeze(-1) #* norm_cargo
            std_demand = td["obs"]["std_demand"].unsqueeze(-1) #* norm_cargo
        elif td["obs"]["expected_demand"].dim() == 3:
            expected_demand = td["obs"]["expected_demand"][:,0].unsqueeze(-1) #* norm_cargo
            std_demand = td["obs"]["std_demand"][:,0].unsqueeze(-1) #* norm_cargo
        else:
            raise ValueError("Invalid shape for expected_demand")

        # Concatenate demand and cargo_parameters and pass through linear layer
        cargo_parameters  = self._combine_cargo_parameters(batch_size=td.batch_size[0])
        combined_input = torch.cat([expected_demand, std_demand, *cargo_parameters.values(),], dim=-1)
        combined_emb = self.fc(combined_input)

        # Positional encoding
        # todo: add positional encoding
        # initial_embedding = self.positional_encoding(positional_emb)
        initial_embedding = combined_emb
        return initial_embedding

class MPPContextEmbedding(nn.Module):
    """Context embedding of the MPP;
    - Selects the initial embedding based on the episodic step
    - Embeds the state of the MPP for the context
    """

    def __init__(self, action_dim, embed_dim, env, demand_aggregation="full",):
        super(MPPContextEmbedding, self).__init__()
        self.env = env
        self.seq_size = self.env.T * self.env.K
        self.project_context = nn.Linear(embed_dim + 20, embed_dim, ) # 286, 217

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

    def forward(self,
                init_embeddings: Tensor,
                td: TensorDict):
        """Embed the context for the MPP"""
        # Get init embedding and state embedding
        select_init_embedding = gather_by_index(init_embeddings, td["timestep"])
        state_embedding = self._state_embedding(td)
        # check_for_nans(select_init_embedding, "select_init_embedding")
        # check_for_nans(state_embedding, "state_embedding")

        # Project state, concat embeddings, and project concat to output
        context_embedding = torch.cat([select_init_embedding, state_embedding], dim=-1)
        # check_for_nans(context_embedding, "context_embedding")
        output = self.project_context(context_embedding)
        # check_for_nans(output, "output")
        return output

    def _state_embedding(self, td):
        """Embed the state for the MPP.
        Important:
        - The state embedding size should not depend on e.g. voyage length and cargo types.
        - todo: We have dependency on ports and bays, which is not ideal for generalization.
        """
        # Extract demand
        current_demand = td["obs"]["current_demand"] #* norm_cargo_t
        expected_demand = td["obs"]["expected_demand"] #* norm_cargo
        std_demand = td["obs"]["std_demand"] #* norm_cargo
        observed_demand = td["obs"]["observed_demand"] #* norm_cargo

        # Extract vessel and location embeddings (all in range [0,1])
        residual_capacity = td["obs"]["residual_capacity"]
        residual_lc_capacity = td["obs"]["residual_lc_capacity"]
        origin_embed = td["obs"]["agg_pol_location"]/self.env.P
        destination_embed = td["obs"]["agg_pod_location"]/self.env.P
        # print("t", td["timestep"][0])
        # print("residual_capacity", residual_capacity.mean(dim=0))
        # print("residual_lc_capacity", residual_lc_capacity.mean(dim=0))

        # Concatenate all embeddings
        state_embed = torch.cat([
            current_demand, expected_demand, std_demand, observed_demand,
            residual_capacity, residual_lc_capacity, origin_embed, destination_embed
        ], dim=-1)
        return state_embed

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