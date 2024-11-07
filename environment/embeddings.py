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
        self.env = env
        # Initialize embeddings and layers
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
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim)
        self.cache_initialized = False  # Flag to check if cache has been created

    def initialize_cache(self,):
        # Compute and store fixed embeddings in cache
        self.origin_emb_cache = self.origin_port(self.env.pol[:-1].to(torch.int64)).unsqueeze(0)
        self.destination_emb_cache = self.destination_port(self.env.pod[:-1].to(torch.int64)).unsqueeze(0)
        self.class_embed_cache = self.cargo_class(self.env.k[:-1].to(torch.int64)).unsqueeze(0)
        self.weight_emb_cache = self.weight(self.env.weights[self.env.k[:-1]].view(-1, 1)).unsqueeze(0)
        self.teu_embd_cache = self.teu(self.env.teus[self.env.k[:-1]].view(-1, 1)).unsqueeze(0)
        self.revenue_emb_cache = self.revenue((self.env.revenues[:-1] /
                                               (self.env.pod[:-1] - self.env.pol[:-1])).view(-1, 1)).unsqueeze(0)
        self.cache_initialized = True  # Update flag

    def forward(self, td: TensorDict):
        # Get batch size and sequence length
        batch_size, step_size = td["POL"].shape

        # Initialize cache if not done yet
        if not self.cache_initialized:
            self.initialize_cache()

        # Compute only demand-related embeddings dynamically
        expected_demand = self.ex_demand(td["expected_demand"].view(batch_size, step_size, 1))
        std_demand = self.stdx_demand(td["std_demand"].view(batch_size, step_size, 1))

        # Concatenate all embeddings (using cached values for fixed ones)
        combined_emb = torch.cat([
            expected_demand, std_demand,
            self.origin_emb_cache.expand(batch_size, -1, -1),
            self.destination_emb_cache.expand(batch_size, -1, -1),
            self.class_embed_cache.expand(batch_size, -1, -1),
            self.weight_emb_cache.expand(batch_size, -1, -1),
            self.teu_embd_cache.expand(batch_size, -1, -1),
            self.revenue_emb_cache.expand(batch_size, -1, -1),
        ], dim=-1)
        # Final projection
        initial_embedding = self.fc(combined_emb)
        return initial_embedding

class MPPContextEmbedding(nn.Module):

    def __init__(self, action_dim, embed_dim, env, device, demand_dim=3,):
        super(MPPContextEmbedding, self).__init__()
        self.env = env

        # Categorical embeddings with linear layers
        self.origin_location = nn.Linear(action_dim, embed_dim)
        self.destination_location = nn.Linear(action_dim, embed_dim)
        # Continuous embeddings
        self.expected_demand = nn.Linear(1, embed_dim)
        self.std_demand = nn.Linear(1, embed_dim)
        self.observed_demand = nn.Linear(1, embed_dim)
        self.residual_capacity = nn.Linear(action_dim, embed_dim)
        self.total_loaded = nn.Linear(1, embed_dim)
        self.overstowage = nn.Linear(self.env.B, embed_dim)
        self.excess_crane_moves = nn.Linear(self.env.B - 1, embed_dim)
        self.violation = nn.Linear(5, embed_dim)
        self.rhs = nn.Linear(5, embed_dim)
        self.lhs_A = nn.Linear(action_dim * 5, embed_dim)
        self.project_context = nn.Linear(embed_dim * 12, embed_dim, )

        # Self-attention layer
        # self.demand = SelfAttentionStateMapping(feature_dim=demand_dim, embed_dim=embed_dim, device=device)

    def forward(self,
                init_embeddings: Tensor,
                td: TensorDict):
        """Embed the context for the MPP"""

        # Get init embedding and state embedding
        select_init_embedding = gather_by_index(init_embeddings, td["episodic_step"])
        state_embedding = self._state_embedding(init_embeddings, td)

        # Project state, concat embeddings, and project concat to output
        context_embedding = torch.cat([select_init_embedding, state_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output

    def _state_embedding(self, embeddings, td):
        """Embed the state for the MPP.
        Important:
        - The state embedding size should not depend on e.g. voyage length and cargo types.
        - todo: It does depend on vessel size now, but this could be changed.
        """
        # Demand
        current_demand = self.expected_demand(td["obs"]["current_demand"].view(td.batch_size[0], -1))
        expected_demand = self.expected_demand(
            torch.sum(td["obs"]["expected_demand"].view(td.batch_size[0], -1), dim=-1, keepdim=True))
        std_demand = self.std_demand(
            torch.sum(td["obs"]["std_demand"].view(td.batch_size[0], -1), dim=-1, keepdim=True))
        observed_demand = self.observed_demand(
            torch.sum(td["obs"]["observed_demand"].view(td.batch_size[0], -1), dim=-1, keepdim=True))

        # Vessel
        residual_capacity = self.residual_capacity(td["obs"]["residual_capacity"].view(td.batch_size[0], -1))
        # origin_embed = self.origin_location(td["state"]["agg_pol_location"].view(td.batch_size[0], -1))
        # destination_embed = self.destination_location(td["state"]["agg_pod_location"].view(td.batch_size[0], -1))

        # Performance
        total_loaded = self.total_loaded(td["obs"]["total_loaded"].view(td.batch_size[0], -1))
        overstowage = self.overstowage(td["obs"]["overstowage"].view(td.batch_size[0], -1))
        excess_crane_moves = self.excess_crane_moves(td["obs"]["excess_crane_moves"].view(td.batch_size[0], -1))

        # Feasibility
        violation = self.violation(td["obs"]["violation"].view(td.batch_size[0], -1))
        rhs = self.rhs(td["rhs"].view(td.batch_size[0], -1))
        lhs_A = self.lhs_A(td["lhs_A"].view(td.batch_size[0], -1))

        # Concatenate all embeddings
        state_embed = torch.cat([
            current_demand, expected_demand, std_demand, observed_demand,
            residual_capacity, #origin_embed, destination_embed,
            total_loaded, overstowage, excess_crane_moves,
            violation, rhs, lhs_A
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
        seq_length = x.size(1)
        position = torch.arange(seq_length, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / seq_length))
        pe = torch.zeros(seq_length, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe

class SelfAttentionStateMapping(nn.Module):
    def __init__(self, feature_dim, embed_dim, device):
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