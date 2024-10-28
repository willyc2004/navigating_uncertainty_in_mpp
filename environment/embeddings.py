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
    def __init__(self, embed_dim, env):
        super(MPPInitEmbedding, self).__init__()
        self.env = env

        # Category embeddings
        self.embed_dim = embed_dim
        self.origin_port = nn.Embedding(self.env.P, embed_dim)
        self.destination_port = nn.Embedding(self.env.P, embed_dim)
        self.cargo_class = nn.Embedding(self.env.K, embed_dim)

        # Continuous embeddings
        self.weight = nn.Linear(1, embed_dim)
        self.teu = nn.Linear(1, embed_dim)
        self.revenue = nn.Linear(1, embed_dim)
        self.ex_demand = nn.Linear(1, embed_dim)
        self.stdx_demand = nn.Linear(1, embed_dim)
        self.fc = nn.Linear(8 * embed_dim, embed_dim)
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim)

    def forward(self, td:TensorDict):
        batch_size, step_size = td["POL"].shape

        # Embed categorical features
        origin_emb = self.origin_port(td["POL"].to(torch.int64))
        destination_emb = self.destination_port(td["POD"].to(torch.int64))
        class_embed = self.cargo_class(td["cargo_class"].to(torch.int64))

        # Embed other features
        weight_emb = self.weight(td["weight"].view(batch_size, step_size, 1))
        capacity_emb = self.teu(td["TEU"].view(batch_size, step_size, 1))
        revenue_emb = self.revenue((td["revenue"] / (td["POD"] - td["POL"])).view(batch_size, step_size, 1))
        expected_demand = self.ex_demand(td["expected_demand"].view(batch_size, step_size, 1))
        std_demand = self.stdx_demand(td["std_demand"].view(batch_size, step_size, 1))

        # Concatenate all embeddings
        combined_emb = torch.cat(
            [origin_emb, destination_emb, class_embed,
             weight_emb, capacity_emb, revenue_emb, expected_demand, std_demand],
            dim=-1)

        # Concatenate and project to shared embedding space
        initial_embedding = self.fc(combined_emb)
        return initial_embedding

class MPPContextEmbedding(nn.Module):

    def __init__(self, action_dim, embed_dim, env, device, demand_dim=3,):
        super(MPPContextEmbedding, self).__init__()
        self.env = env

        # Categorical embeddings with linear layers
        self.origin_location = nn.Linear(action_dim * self.env.P, embed_dim)
        self.destination_location = nn.Linear(action_dim * self.env.P, embed_dim)
        # Continuous embeddings
        self.expected_demand = nn.Linear(1, embed_dim)
        self.std_demand = nn.Linear(1, embed_dim)
        self.actual_demand = nn.Linear(1, embed_dim)
        self.residual_capacity = nn.Linear(action_dim, embed_dim)
        self.total_loaded = nn.Linear(1, embed_dim)
        self.overstowage = nn.Linear(self.env.B, embed_dim)
        self.excess_crane_moves = nn.Linear(self.env.B - 1, embed_dim)
        self.lhs_A = nn.Linear(action_dim * 5, embed_dim)
        self.rhs = nn.Linear(5, embed_dim)
        self.violation = nn.Linear(5, embed_dim)
        self.project_context = nn.Linear(embed_dim * 11, embed_dim, )

        # Self-attention layer
        # self.demand = SelfAttentionStateMapping(feature_dim=demand_dim, embed_dim=embed_dim, device=device)
        num_features = 3 + action_dim + 1 + self.env.B + self.env.B - 1 + action_dim * 5 + 5 + 5

        # Feature index dict to use a indices to slice the input tensor
        self.feature_index = {
            "expected_demand": slice(0, 1),
            "std_demand": slice(1, 2),
            "observed_demand": slice(2, 3),
            "clip_max": slice(3, 3 + action_dim),
            "total_loaded": slice(3 + action_dim, 4 + action_dim),
            "overstowage": slice(4 + action_dim, 4 + action_dim + self.env.B),
            "long_crane_excess": slice(4 + action_dim + self.env.B,
                                       4 + action_dim + self.env.B + self.env.B - 1),
            "lhs_A": slice(4 + action_dim + self.env.B + self.env.B - 1,
                           4 + action_dim + self.env.B + self.env.B - 1 + action_dim * 5),
            "rhs": slice(4 + action_dim + self.env.B + self.env.B - 1 + action_dim * 5,
                         4 + action_dim + self.env.B + self.env.B - 1 + action_dim * 5 + 5),
            "violation": slice(4 + action_dim + self.env.B + self.env.B - 1 + action_dim * 5 + 5,
                               4 + action_dim + self.env.B + self.env.B - 1 + action_dim * 5 + 5 + 5),
        }
        self.multi_norm = MultiFeatureRunningNormalization(num_features=num_features)

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
        # Normalize features
        features = torch.cat([
            torch.sum(td["state"]["expected_demand"].view(td.batch_size[0], -1, ), dim=-1, keepdim=True),
            torch.sum(td["state"]["std_demand"].view(td.batch_size[0], -1, ), dim=-1, keepdim=True),
            torch.sum(td["state"]["observed_demand"].view(td.batch_size[0], -1, ), dim=-1, keepdim=True),
            td["clip_max"].view(td.batch_size[0], -1, ),
            td["state"]["total_loaded"].view(td.batch_size[0], -1, ),
            td["state"]["overstowage"].view(td.batch_size[0], -1, ),
            td["state"]["long_crane_excess"].view(td.batch_size[0], -1, ),
            td["lhs_A"].view(td.batch_size[0], -1),
            td["rhs"].view(td.batch_size[0], -1),
            td["violation"].view(td.batch_size[0], -1),
        ], dim=-1)
        self.multi_norm.update(features)
        norm_features = self.multi_norm.normalize(features)

        # get slices from norm_features, and apply linear layers
        expected_demand = self.expected_demand(norm_features[:, self.feature_index["expected_demand"]])
        std_demand = self.std_demand(norm_features[:, self.feature_index["std_demand"]])
        actual_demand = self.actual_demand(norm_features[:, self.feature_index["observed_demand"]])
        residual_capacity = self.residual_capacity(norm_features[:, self.feature_index["clip_max"]])
        total_loaded = self.total_loaded(norm_features[:, self.feature_index["total_loaded"]])
        overstowage = self.overstowage(norm_features[:, self.feature_index["overstowage"]])
        excess_crane_moves = self.excess_crane_moves(norm_features[:, self.feature_index["long_crane_excess"]])
        lhs_A = self.lhs_A(norm_features[:, self.feature_index["lhs_A"]])
        rhs = self.rhs(norm_features[:, self.feature_index["rhs"]])
        violation = self.violation(norm_features[:, self.feature_index["violation"]])

        # todo: not having num_classes in one_hot or embedding causes an error
        # origin_onehot = F.one_hot(td["state"]["agg_pol_location"].view(td.batch_size[0], -1, ).to(torch.int64), num_classes=max_pol).float()
        # origin_embed = self.origin_location(origin_onehot.view(td.batch_size[0], -1, ))
        # destination_onehot = F.one_hot(td["state"]["agg_pod_location"].view(td.batch_size[0], -1, ).to(torch.int64), num_classes=max_pod).float()
        # destination_embed = self.destination_location(destination_onehot.view(td.batch_size[0], -1, ))

        # Concatenate all embeddings
        state_embed = torch.cat([expected_demand, std_demand, actual_demand,
                                 residual_capacity, #origin_embed, destination_embed,
                                 total_loaded, overstowage, excess_crane_moves,
                                 lhs_A, rhs, violation], dim=-1)
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