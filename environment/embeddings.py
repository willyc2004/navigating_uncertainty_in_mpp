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
        self.cache_initialized = False  # Flag to check if cache has been created

        # Categorical embeddings
        self.cargo_class = nn.Embedding(self.env.K, embed_dim)
        # Ordinal embeddings
        self.origin_port = nn.Linear(1, embed_dim)
        self.destination_port = nn.Linear(1, embed_dim)
        # Continuous embeddings
        self.weight = nn.Linear(1, embed_dim)
        self.teu = nn.Linear(1, embed_dim)
        self.revenue = nn.Linear(1, embed_dim)
        self.ex_demand = nn.Linear(1, embed_dim)
        self.stdx_demand = nn.Linear(1, embed_dim)
        # Final projection and positional encoding
        num_embeddings = 8  # Number of embeddings
        self.fc = nn.Linear(num_embeddings * embed_dim, embed_dim)
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(num_embeddings*embed_dim)

    def initialize_cache(self):
        """Initialize cache for fixed embeddings"""
        # Point to integer
        cargo_class = self.env.k[:-1].to(torch.int64)
        # Normalize
        norm_pol = (self.env.pol[:-1] / self.env.P).view(-1, 1)
        norm_pod = (self.env.pod[:-1] / self.env.P).view(-1, 1)
        norm_weights = (self.env.weights[self.env.k[:-1]] / self.env.weights[self.env.k[:-1]].max()).view(-1, 1)
        norm_teus = (self.env.teus[self.env.k[:-1]] / self.env.teus[self.env.k[:-1]].max()).view(-1, 1)
        # todo: consider revenue per capacity!
        norm_revenues = (self.env.revenues[:-1] / self.env.revenues[:-1].max()).view(-1, 1)

        # Category embeddings
        self.class_embed_cache = self.cargo_class(cargo_class).view(1,self.seq_size, -1)
        # Ordinal embeddings
        self.origin_emb_cache = self.origin_port(norm_pol).view(1,self.seq_size, -1)
        self.destination_emb_cache = self.destination_port(norm_pod).view(1,self.seq_size, -1)
        # Continuous embeddings
        self.weight_emb_cache = self.weight(norm_weights).view(1,self.seq_size, -1)
        self.teu_emb_cache = self.teu(norm_teus).view(1,self.seq_size, -1)
        self.revenue_emb_cache = self.revenue(norm_revenues).view(1,self.seq_size, -1)
        self.cache_initialized = True  # Update flag

    def forward(self, td: TensorDict):
        # Get batch size and cargo normalization (based on teu and total capacity)
        batch_size = td.batch_size
        norm_cargo = (self.env.teus[self.env.k[:-1]] / self.env.total_capacity).view(1, -1, 1)
        if td["obs"]["expected_demand"].dim() == 2:
            e_x = reorder_demand(td["obs"]["expected_demand"], self.env.tau, self.env.k, self.env.T, self.env.K,batch_size)
            std_x = reorder_demand(td["obs"]["std_demand"], self.env.tau, self.env.k, self.env.T, self.env.K,batch_size)
            expected_demand = self.ex_demand(e_x.view(*batch_size, -1, 1) * norm_cargo)
            std_demand = self.stdx_demand(std_x.view(*batch_size, -1, 1) * norm_cargo)
        elif td["obs"]["expected_demand"].dim() == 3:
            e_x = reorder_demand(td["obs"]["expected_demand"][:,0], self.env.tau, self.env.k, self.env.T, self.env.K,batch_size)
            std_x = reorder_demand(td["obs"]["std_demand"][:,0], self.env.tau, self.env.k, self.env.T, self.env.K,batch_size)
            expected_demand = self.ex_demand(e_x.view(*batch_size, -1, 1) * norm_cargo)
            std_demand = self.stdx_demand(std_x.view(*batch_size, -1, 1) * norm_cargo)
        else:
            raise ValueError("Invalid shape for POL")

        # Initialize cache if not done yet
        if not self.cache_initialized:
            self.initialize_cache()

        # Concatenate all embeddings (using cached values for fixed ones)
        combined_emb = torch.cat([
            expected_demand, std_demand,
            self.origin_emb_cache.expand(*batch_size, -1, -1),
            self.destination_emb_cache.expand(*batch_size, -1, -1),
            self.class_embed_cache.expand(*batch_size, -1, -1),
            self.weight_emb_cache.expand(*batch_size, -1, -1),
            self.teu_emb_cache.expand(*batch_size, -1, -1),
            self.revenue_emb_cache.expand(*batch_size, -1, -1),
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
        self.seq_size = self.env.T * self.env.K

        # Categorical embeddings with linear layers
        self.origin_location = nn.Linear(action_dim, embed_dim)
        self.destination_location = nn.Linear(action_dim, embed_dim)
        self.binary_origin_location = nn.Linear(action_dim*self.env.P, embed_dim)
        self.binary_destination_location = nn.Linear(action_dim*self.env.P, embed_dim)
        # Continuous embeddings
        self.current_demand = nn.Linear(1, embed_dim)
        self.residual_capacity = nn.Linear(action_dim, embed_dim)
        self.long_crane_capacity = nn.Linear(self.env.B - 1, embed_dim)
        self.project_context = nn.Linear(embed_dim * 9, embed_dim, )

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

        # todo: add state normalization!!
        # todo: check if gather_by_index is correct

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
        if td["obs"]["expected_demand"].dim() == 2:
            batch_size, _ = td["obs"]["expected_demand"].shape
            view_transform = lambda x: x.view(batch_size, -1)
        elif td["obs"]["expected_demand"].dim() == 3:
            batch_size, n_step, _ = td["obs"]["expected_demand"].shape
            view_transform = lambda x: x.view(batch_size, n_step, -1)
        else:
            raise ValueError("Invalid shape of done in td")

        # Process demand embeddings based on demand aggregation type
        def process_demand_embedding(embedding_func, view_transform, demand_type, td, aggregation):
            demand = view_transform(td[demand_type])
            if aggregation == "sum":
                return embedding_func(torch.sum(demand, dim=-1, keepdim=True))
            elif aggregation == "self_attn":
                demand, _ = embedding_func(demand.unsqueeze(-1))
                if td["expected_demand"].dim() == 3:
                    return demand.view(batch_size, n_step, self.seq_size, -1)
                return demand
            raise ValueError(f"Invalid demand aggregation: {aggregation}")

        # Extract timestep-specific demand embedding based on episodic step
        def extract_demand_at_timestep(demand, td):
            dim, index = (2, td["episodic_step"][:, 0]) if td["done"].dim() == 3 else (1, td["episodic_step"])
            return gather_by_index(demand, index, dim=dim)

        # Compute demand embeddings
        current_demand = self.current_demand(view_transform(td["obs"]["current_demand"]))
        expected_demand_t = process_demand_embedding(self.expected_demand, view_transform, "expected_demand", td["obs"],
                                                     self.demand_aggregation)
        std_demand_t = process_demand_embedding(self.std_demand, view_transform, "std_demand", td["obs"],
                                                self.demand_aggregation)
        observed_demand_t = process_demand_embedding(self.observed_demand, view_transform, "observed_demand", td["obs"],
                                                     self.demand_aggregation)

        if self.demand_aggregation == "self_attn":
            expected_demand_t = extract_demand_at_timestep(expected_demand_t, td)
            std_demand_t = extract_demand_at_timestep(std_demand_t, td)
            observed_demand_t = extract_demand_at_timestep(observed_demand_t, td)

        # Compute vessel and location embeddings
        residual_capacity = self.residual_capacity(view_transform(td["obs"]["residual_capacity"]))
        # origin_embed = self.origin_location(view_transform(td["obs"]["agg_pol_location"]))
        # destination_embed = self.destination_location(view_transform(td["obs"]["agg_pod_location"]))
        origin_embed = self.binary_origin_location(view_transform(td["obs"]["pol_location"]))
        destination_embed = self.binary_destination_location(view_transform(td["obs"]["pod_location"]))
        residual_lc_capacity = self.long_crane_capacity(view_transform(td["obs"]["residual_lc_capacity"]))

        # Concatenate all embeddings
        state_embed = torch.cat([
            current_demand, expected_demand_t, std_demand_t, observed_demand_t,
            residual_capacity, residual_lc_capacity, origin_embed, destination_embed
        ], dim=-1)

        return state_embed

def reorder_demand(demand, tau, k, T, K, batch_size):
    """Reorder demand to match the episode ordering"""
    return demand.view(*batch_size, T, K)[:, tau[:-1], k[:-1]]

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
        # Reshape input if it includes multiple steps
        if X.dim() == 4:  # Expected shape [batch, n_step, seq, feature_dim]
            batch_size, n_step, seq_len, _ = X.shape
            X = X.view(batch_size * n_step, seq_len, self.feature_dim)

        # Linearly transform input tensor X to Q, K, V
        Q = self.W_Q(X)  # (batch_size * n_step, seq_len, F)
        K = self.W_K(X)  # (batch_size * n_step, seq_len, F)
        V = self.W_V(X)  # (batch_size * n_step, seq_len, F)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1,
                                                       -2)) / self.scale_factor  # (batch_size * n_step, seq_len, seq_len)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (batch_size * n_step, seq_len, seq_len)

        # Compute weighted sum of V
        attention_output = torch.matmul(attention_weights, V)  # (batch_size * n_step, seq_len, F)
        attention_output = self.final_linear(attention_output)  # (batch_size * n_step, seq_len, embed_dim)

        # Restore original batch and step dimensions if reshaped
        if X.dim() == 4:
            attention_output = attention_output.view(batch_size, n_step, seq_len, -1)
            attention_weights = attention_weights.view(batch_size, n_step, seq_len, seq_len)

        return attention_output, attention_weights