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
        # Env
        self.env = env

        # Layers
        self.embed_dim = embed_dim
        self.origin_port = nn.Linear(1, embed_dim)
        self.destination_port = nn.Linear(1, embed_dim)
        self.cargo_class = nn.Linear(1, embed_dim)
        self.weight = nn.Linear(1, embed_dim)
        self.teu = nn.Linear(1, embed_dim)
        self.revenue = nn.Linear(1, embed_dim)
        self.ex_demand = nn.Linear(1, embed_dim)
        self.stdx_demand = nn.Linear(1, embed_dim)
        self.fc = nn.Linear(8 * embed_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim)

    def forward(self, td:TensorDict):
        batch_size, step_size = td["POL"].shape
        # Convert origin and destination to one-hot encodings
        origin_one_hot = F.one_hot(td["POL"], num_classes=self.env.P).float()
        print(origin_one_hot.shape)
        breakpoint()

        destination_one_hot = F.one_hot(td["POD"], num_classes=self.env.P).float()

        origin_emb = self.origin_port(td["POL"].view(batch_size, step_size, 1).float())
        destination_emb = self.destination_port(td["POD"].view(batch_size, step_size, 1).float())
        type_emb = self.cargo_class(td["cargo_class"].view(batch_size, step_size, 1).float())
        weight_emb = self.weight(td["weight"].view(batch_size, step_size, 1))
        capacity_emb = self.teu(td["TEU"].view(batch_size, step_size, 1))
        revenue_emb = self.revenue((td["revenue"] / (td["POD"] - td["POL"])).view(batch_size, step_size, 1))
        expected_demand = self.ex_demand(td["expected_demand"].view(batch_size, step_size, 1))
        std_demand = self.stdx_demand(td["std_demand"].view(batch_size, step_size, 1))

        # Concatenate all embeddings
        combined_emb = torch.cat(
            [origin_emb, destination_emb, type_emb,
             weight_emb, capacity_emb, revenue_emb, expected_demand, std_demand],
            dim=-1)

        # Concatenate and project to shared embedding space
        initial_embedding = self.fc(combined_emb)
        return initial_embedding

class MPPContextEmbedding(nn.Module):

    def __init__(self, action_dim, embed_dim, env, device, demand_dim=3,):
        super(MPPContextEmbedding, self).__init__()
        # Environment
        self.env = env
        self.capacity_norm = env.total_capacity

        # Layers
        self.origin_location = nn.Linear(action_dim, embed_dim)
        self.destination_location = nn.Linear(action_dim, embed_dim)
        self.residual_capacity = nn.Linear(action_dim, embed_dim)
        self.total_loaded = nn.Linear(1, embed_dim)
        self.overstowage = nn.Linear(self.env.B, embed_dim)
        self.excess_crane_moves = nn.Linear(self.env.B - 1, embed_dim)
        self.lhs_A = nn.Linear(action_dim * 5, embed_dim)
        self.rhs = nn.Linear(5, embed_dim)
        self.violation = nn.Linear(5, embed_dim)
        self.project_context = nn.Linear(embed_dim * 11, embed_dim, )

        # Self-attention layer
        self.demand = SelfAttentionStateMapping(feature_dim=demand_dim, embed_dim=embed_dim, device=device)

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
        # demand_state = torch.cat([td["state"]["observed_demand"].view(td.batch_size[0], -1, 1),
        #                           td["state"]["expected_demand"].view(td.batch_size[0], -1, 1),
        #                           td["state"]["std_demand"].view(td.batch_size[0], -1, 1), ], dim=-1)
        # demand_embed = self.demand(demand_state)  # attention output [BATCH, Seq, F]
        # demand_embed_t = gather_by_index(demand_embed, td["episodic_step"])

        # Vessel
        location_embed = self.residual_capacity(td["clip_max"].view(td.batch_size[0], -1, ) / self.env.total_capacity,)
        origin_embed = self.origin_location(td["state"]["agg_pol_location"].view(td.batch_size[0], -1,).float())
        destination_embed = self.destination_location(td["state"]["agg_pod_location"].view(td.batch_size[0], -1,).float())

        # Performance
        total_loaded = self.total_loaded(td["state"]["total_loaded"].view(td.batch_size[0], -1, ))
        overstowage = self.overstowage(td["state"]["overstowage"].view(td.batch_size[0], -1, ))
        excess_crane_moves = self.excess_crane_moves(td["state"]["long_crane_excess"].view(td.batch_size[0], -1, ))

        # Feasibility
        lhs_A = self.lhs_A(td["lhs_A"].view(td.batch_size[0], -1))
        rhs = self.rhs(td["rhs"].view(td.batch_size[0], -1))
        violation = self.violation(td["violation"].view(td.batch_size[0], -1))

        # Concatenate all embeddings
        state_embed = torch.cat([demand_embed_t,
                                 location_embed, origin_embed, destination_embed,
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

from typing import Tuple

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, axis=0)
        batch_var = torch.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count