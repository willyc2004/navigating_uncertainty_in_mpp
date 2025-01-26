import math
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from typing import Tuple, Callable, Optional, Dict
from rl4co.utils.ops import gather_by_index

class MPPInitEmbedding(nn.Module):
    def __init__(self, action_dim, embed_dim, seq_dim, env):
        super(MPPInitEmbedding, self).__init__()
        # Store environment and sequence size
        self.env = env
        self.seq_dim = seq_dim
        self.train_max_demand = self.env.generator.train_max_demand

        # Embedding layers
        if env.name == "mpp":
            num_embeddings = 7  # Number of embeddings
            self.fc = nn.Linear(num_embeddings, embed_dim)
        elif env.name == "port_mpp":
            num_embeddings = self.seq_dim  # Number of embeddings
            self.fc = nn.Linear(num_embeddings, embed_dim)
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim)
        self.zeros = torch.zeros(1, seq_dim, embed_dim, device=self.env.device, dtype=self.env.float_type)

    def _combine_cargo_parameters(self, batch_size):
        """Prepare cargo parameters for init embedding"""
        if batch_size == torch.Size([]):
            norm_features = {
                "pol": (self.env.pol.clone() / self.env.P).view(1, -1, 1),
                "pod": (self.env.pod.clone() / self.env.P).view(1, -1, 1),
                "weights": (self.env.weights[self.env.k].clone() / self.env.weights[self.env.k].max()).view(1, -1, 1),
                "teus": (self.env.teus[self.env.k].clone() / self.env.teus[self.env.k].max()).view(1, -1, 1),
                "revenues": (self.env.revenues.clone() / self.env.revenues.max()).view(1, -1, 1),
            }
        else:
            norm_features = {
                "pol": (self.env.pol.clone() / self.env.P).view(1, -1, 1).expand(batch_size, -1, -1),
                "pod": (self.env.pod.clone() / self.env.P).view(1, -1, 1).expand(batch_size, -1, -1),
                "weights": (self.env.weights[self.env.k].clone() / self.env.weights[self.env.k].max()).view(1, -1, 1).expand(batch_size, -1, -1),
                "teus": (self.env.teus[self.env.k].clone() / self.env.teus[self.env.k].max()).view(1, -1, 1).expand(batch_size, -1, -1),
                "revenues": (self.env.revenues.clone() / self.env.revenues.max()).view(1, -1, 1).expand(batch_size, -1, -1),
            }
        return norm_features

    def forward(self, td: Tensor,):
        batch_size = td.shape[0]
        cargo_parameters = self._combine_cargo_parameters(batch_size=batch_size)
        max_demand = td["realized_demand"].max() if self.train_max_demand == None else self.train_max_demand
        if td["expected_demand"].dim() == 2:
            expected_demand = td["expected_demand"].unsqueeze(-1) / max_demand
            std_demand = td["std_demand"].unsqueeze(-1) / max_demand
        else:
            expected_demand = td["expected_demand"][..., 0, :].unsqueeze(-1) / max_demand
            std_demand = td["std_demand"][..., 0, :].unsqueeze(-1) / max_demand
        combined_input = torch.cat([
            expected_demand, std_demand, *cargo_parameters.values()], dim=-1)
        combined_emb = self.fc(combined_input)

        # Positional encoding
        initial_embedding = self.positional_encoding(combined_emb)
        return initial_embedding

class MPPObservationEmbedding(nn.Module):
    """Context embedding of the MPP;
    - Selects the initial embedding based on the episodic step
    - Embeds the state of the MPP for the context
    """

    def __init__(self, action_dim, embed_dim, seq_dim, env, demand_aggregation="full"):
        super(MPPObservationEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.project_context = nn.Linear(embed_dim + 143, embed_dim,)
        self.train_max_demand = self.env.generator.train_max_demand

    def normalize_obs(self, td):
        batch_size = td.batch_size
        max_demand = td["realized_demand"].max() if self.train_max_demand == None else self.train_max_demand
        return torch.cat([
            (td["observed_demand"] / max_demand ).view(*batch_size, self.env.T * self.env.K),
            (td["residual_capacity"] / self.env.capacity.view(1, self.env.B * self.env.D)).view(*batch_size, self.env.B * self.env.D),
            (td["residual_lc_capacity"] / td["target_long_crane"].unsqueeze(0)).view(*batch_size, self.env.B - 1),
            td["lcg"],
            td["vcg"],
            td["agg_pol_location"] / self.env.P,
            td["agg_pod_location"] / self.env.P,
        ], dim=-1)

    def forward(self,  latent_state: Tensor, td: TensorDict):
        """Embed the context for the MPP"""
        # Get relevant init embedding
        if td["timestep"].dim() == 1:
            select_init_embedding = gather_by_index(latent_state, td["timestep"][0])
        else:
            select_init_embedding = latent_state.squeeze()

        # Project state, concat embeddings, and project concat to output
        obs = self.normalize_obs(td)
        context_embedding = torch.cat([obs, select_init_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output


class MPPContextEmbedding(nn.Module):
    """Context embedding of the MPP;
    - Selects the initial embedding based on the episodic step
    - Embeds the state of the MPP for the context
    """

    def __init__(self, action_dim, embed_dim, seq_dim, env, demand_aggregation="full",):
        super(MPPContextEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.project_context = nn.Linear(embed_dim + 71, embed_dim,)

    def normalize_obs(self, td):
        batch_size = td.batch_size
        return torch.cat([
            (td["residual_capacity"] / self.env.capacity.view(1, self.env.B * self.env.D)).view(*batch_size, self.env.B * self.env.D),
            (td["residual_lc_capacity"] / td["target_long_crane"].unsqueeze(0)).view(*batch_size, self.env.B - 1),
            td["lcg"],
            td["vcg"],
            td["agg_pol_location"] / self.env.P,
            td["agg_pod_location"] / self.env.P,
        ], dim=-1)

    def forward(self, latent_state: Tensor, td: TensorDict):
        """Embed the context for the MPP"""
        # Get relevant init embedding
        if td["timestep"].dim() == 1:
            select_init_embedding = gather_by_index(latent_state, td["timestep"][0])
        else:
            select_init_embedding = latent_state
        # Project state, concat embeddings, and project concat to output
        obs = self.normalize_obs(td)
        context_embedding = torch.cat([obs, select_init_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output

class MPPDynamicEmbedding(nn.Module):
    def __init__(self, embed_dim, seq_dim, env, demand_aggregation="full",):
        super(MPPDynamicEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.project_dynamic = nn.Linear(embed_dim + 1, 3 * embed_dim)
        self.train_max_demand = self.env.generator.train_max_demand

    def forward(self, latent_state: Optional[Tensor], td: Tensor):
        """Embed the dynamic demand for the MPP"""
        # Get relevant demand embeddings
        max_demand = td["realized_demand"].max() if self.train_max_demand == None else self.train_max_demand
        if td["observed_demand"].dim() == 2:
            observed_demand = td["observed_demand"].unsqueeze(-1) / max_demand
        else:
            observed_demand = td["observed_demand"][...,0,:].unsqueeze(-1) / max_demand

        # Project key, value, and logit to anticipate future steps
        hidden = torch.cat([observed_demand, latent_state], dim=-1)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.project_dynamic(hidden).chunk(3, dim=-1)
        return glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn

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