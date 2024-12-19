import torch
from torch import Tensor
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

# Modules
from rl4co.rl4co.env import MasterPlanningEnv
from environment.utils import *

# Logger
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

class PortMasterPlanningEnv(MasterPlanningEnv):
    # todo: add comments
    name = "mpp_pol"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pod = torch.triu(self.ports.repeat(self.P, 1), diagonal=1)[:-1,1:].T


    def _make_spec(self, generator:Optional[Generator] = None) -> None:
        """Define the specs for observations, actions, rewards, and done flags."""
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(214)),  # Define shape as needed
        )
        self.action_spec = BoundedTensorSpec(
            shape=(self.B*self.D*self.K*(self.P-1)),  # Define shape as needed
            low=0.0,
            high=10,  # Define high value as needed
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=th.bool)

    def check_solution_validity(self, td, actions) -> th.bool:
        """Check solution validity"""
        # todo: add validity logic
        return True

    def _check_done(self, t: Tensor) -> Tensor:
        """Determine if the episode is done based on the state."""
        return (t == (self.P - 1)).view(-1, 1)

    def _step(self, td: TensorDict) -> TensorDict:
        """Perform a step in the environment and return the next state."""
        ## Extraction and precompute
        action,realized_demand, lhs_A, rhs, t, batch_size = self._extract_from_td(td)
        action_idx, util_idx = self._extract_indices(t) # Shape [B*D*K*P-1] and [B*D*K*T]
        utilization, target_long_crane, total_loaded, total_revenue, total_cost, total_rc = \
            self._extract_from_state(td["state"])
        current_demand, observed_demand, expected_demand, std_demand = self._extract_from_obs(td["obs"], batch_size)

        breakpoint()

        ## Current state
        # Check done, update utilization, and compute violation
        done = self._check_done(t)
        utilization = self._update_state_loading(action, utilization, k, tau,)
        violation = compute_violation(action.view(*batch_size, 1, -1), lhs_A, rhs)

        # Compute overstowage
        pol_locations, pod_locations = self._compute_pol_pod_locations(utilization)
        agg_pol_location, agg_pod_location = self._aggregate_pol_pod_location(pol_locations, pod_locations)
        overstowage = self._compute_hatch_overstowage(utilization, pol)

        # Compute total_loaded and aggregated long crane excess
        sum_action = action.sum(dim=(-2, -1)).unsqueeze(-1)
        total_loaded = total_loaded + th.min(sum_action, current_demand)
        long_crane_moves = self._compute_long_crane(utilization, pol)
        excess_crane_moves = th.clamp(long_crane_moves - target_long_crane.view(-1, 1), min=0)

        ## Reward
        revenue = th.min(sum_action, current_demand) * self.revenues[t[0]]
        profit = revenue.clone()
        if self.next_port_mask[t].any():
            ho_costs = overstowage.sum(dim=-1, keepdim=True) * self.ho_costs
            lc_costs = excess_crane_moves.sum(dim=-1, keepdim=True) * self.lc_costs
            cost = ho_costs + lc_costs
            profit -= cost
        else:
            cost = th.zeros_like(profit)
        # (implemented outside environment with get_reward fn for code efficiency)
        # reward = th.zeros_like(t, dtype=self.float_type)

        ## Transition to next step
        # Update next state
        t = t + 1
        next_state_dict = self._update_next_state(utilization, target_long_crane,
                                                  realized_demand, observed_demand, expected_demand, std_demand, t,)
        # Update feasibility: lhs_A, rhs, clip_max based on next state
        utilization_ = next_state_dict["utilization"]
        lhs_A = self.create_lhs_A(lhs_A, action_idx, util_idx)
        rhs = self.create_rhs(utilization_, current_demand, batch_size)
        # Express residual capacity in number of containers for next step
        residual_capacity = th.clamp(self.norm_capacity - utilization_.sum(dim=-1) @ self.teus, min=self.zero) \
                            / self.teus_episode[t].view(-1,1,1)

        # # Update action mask
        # # action_mask[t] = 0
        # mask_condition = (t[0] % self.K == 0)
        # min_pod = self._compute_min_pod(pol_locations, pod_locations )
        # new_mask = self._prevent_HO_mask(action_mask, pod, pod_locations, min_pod)
        # action_mask = th.where(mask_condition, new_mask, action_mask)

        # Only for final port: compute last port long crane excess
        if t[0] == self.K * self.T:
            # Compute last port long crane excess
            moves_bays_last_port = utilization.sum(dim=(2, 3, 4))
            lc_moves_last_port = (moves_bays_last_port[..., :-1] + moves_bays_last_port[..., 1:])
            lc_excess_last_port = th.clamp(lc_moves_last_port - next_state_dict["target_long_crane"].view(-1,1), min=0)

            # Compute metrics
            excess_crane_moves += lc_excess_last_port
            cost_ = lc_excess_last_port.sum(dim=-1, keepdim=True) * self.lc_costs
            profit -= cost_
            cost += cost_

        # Track metrics
        total_rc += residual_capacity.view(*batch_size, -1)
        total_revenue += revenue
        total_cost += cost
        # Normalize revenue \in [0,1]:
        # revenue_norm = rev_t / max(rev_t) * min(q_t, sum(a_t)) / q_t
        normalize_revenue = self.revenues.max() * current_demand
        # Normalize accumulated cost \in [0, t_cost], where t_cost is the time at which we evaluate cost:
        # cost_norm = cost_{t_cost} / E[q_t]
        normalize_cost = realized_demand.view(*batch_size, -1)[:,:t[0]].sum(dim=-1, keepdims=True) / t[0]
        # Normalize reward: r_t = revenue_norm - cost_norm
        # We have spikes over delayed costs at specific time steps.
        reward = (revenue.clone() / normalize_revenue) - (cost.clone() / normalize_cost)

        # Update td output
        td.update({
            "state":{
                # Vessel
                "utilization": next_state_dict["utilization"],
                "target_long_crane": next_state_dict["target_long_crane"],

                # Performance
                "total_loaded": total_loaded,
                "total_revenue": total_revenue,
                "total_cost": total_cost,
                "total_rc": total_rc,
            },
            "obs":{
                # Demand
                "current_demand": next_state_dict["current_demand"],
                "observed_demand": next_state_dict["observed_demand"].view(*batch_size, self.K * self.T),
                "expected_demand": next_state_dict["expected_demand"].view(*batch_size, self.K * self.T),
                "std_demand": next_state_dict["std_demand"].view(*batch_size, self.K * self.T),
                "residual_capacity": residual_capacity.view(*batch_size, self.B * self.D),
                "agg_pol_location": agg_pol_location.view(*batch_size, self.B * self.D),
                "agg_pod_location": agg_pod_location.view(*batch_size, self.B * self.D),
            },

            # Feasibility and constraints
            "lhs_A": lhs_A,
            "rhs": rhs,
            "violation": violation,
            "clip_max": residual_capacity.view(*batch_size, self.B*self.D),
            # Action, reward, done and step
            "action": action.view(*batch_size, self.B*self.D),
            "reward": reward,
            "profit": profit,
            "revenue": revenue,
            "cost": cost,
            "done": done,
            "timestep":t,
        })
        return td

    def _reset(self,  td: Optional[TensorDict] = None,  batch_size=None) -> TensorDict:
        """Reset the environment to the initial state."""
        # Shape
        td["realized_demand"] = td["realized_demand"].view(*batch_size, self.K, self.T)
        td["observed_demand"] = td["observed_demand"].view(*batch_size, self.K, self.T)

        # Initialize
        # Parameters
        device = td.device
        t = th.zeros(*batch_size, dtype=th.int64)
        action_idx, util_idx = self._extract_indices(t) # Shape [B*D*K*P-1] and [B*D*K*T]
        # Action mask and clipping
        action_mask = th.ones((*batch_size, self.B*self.D), dtype=th.bool, device=device)
        # Vessel state
        residual_capacity = (self.norm_capacity / self.teus[t[0]]).unsqueeze(0).repeat(*batch_size, 1, 1)
        target_long_crane = self._compute_target_long_crane(td["realized_demand"], self.moves_idx[t[0]])
        utilization = th.zeros((*batch_size, self.B, self.D, self.K, self.T), device=device, dtype=self.float_type)
        locations_utilization = th.zeros_like(action_mask, dtype=self.float_type)
        # Constraints
        # todo: create per pol based on observed_demand
        lhs_A = self.create_lhs_A(th.zeros((*batch_size, self.n_constraints, self.B * self.D * self.K * (self.P-1)),
                                           device=device, dtype=self.float_type),
                                  action_idx, util_idx)
        rhs = self.create_rhs(utilization, current_demand, batch_size)
        # Performance
        total_loaded = th.zeros((*batch_size,1), device=device, dtype=self.float_type)

        # Init tds - state: internal state
        initial_state = TensorDict({
            # Vessel
            "utilization": utilization,
            "target_long_crane": target_long_crane,
            # Performance
            "total_loaded": total_loaded,
            "total_revenue": th.zeros_like(total_loaded, dtype=self.float_type),
            "total_cost": th.zeros_like(total_loaded, dtype=self.float_type),
            "total_rc": th.zeros_like(locations_utilization),
        }, batch_size=batch_size, device=device,)

        # Init tds - obs: observed by embeddings
        initial_obs = TensorDict({
            # Demand
            "observed_demand": td["observed_demand"].view(*batch_size, self.K * self.T),
            "expected_demand": td["expected_demand"].view(*batch_size, self.K * self.T),
            "std_demand": td["std_demand"].view(*batch_size, self.K * self.T),
            # Vessel
            "residual_capacity": residual_capacity.view(*batch_size, self.B*self.D),
            "agg_pol_location": th.zeros_like(locations_utilization),
            "agg_pod_location": th.zeros_like(locations_utilization),
        }, batch_size=batch_size, device=device,)

        # Init tds - full td
        td = TensorDict({
            # State + obs
            "state": initial_state,
            "obs": initial_obs,
            # Action mask and clipping
            "action": th.zeros_like(action_mask, dtype=self.float_type),
            "action_mask": action_mask.view(*batch_size, -1),
            # Constraints
            # "lhs_A": lhs_A,
            # "rhs":  rhs,
            # "violation": th.zeros_like(rhs, dtype=self.float_type),
            # Reward, done and step
            "reward": th.zeros_like(t, dtype=self.float_type),
            "profit": th.zeros_like(t, dtype=self.float_type),
            "revenue": th.zeros_like(t, dtype=self.float_type),
            "cost": th.zeros_like(t, dtype=self.float_type),
            "done": th.zeros_like(t, dtype=th.bool,),
            "timestep": t,
        }, batch_size=batch_size, device=device)
        return td

    # Extraction functions
    def _extract_from_td(self, td) -> Tuple:
        """Extract action, reward and step from the TensorDict."""
        # Must clone to avoid in-place operations!
        batch_size = td.batch_size
        timestep = td["timestep"].clone()
        action = td["action"].clone().view(*batch_size, self.B, self.D, self.K, self.P-1)
        # action_mask = td["action_mask"].clone()
        realized_demand = td["realized_demand"].clone()
        lhs_A = td["lhs_A"].clone()
        rhs = td["rhs"].clone()
        return action, realized_demand, lhs_A, rhs, timestep, batch_size

    def _extract_indices(self, t) -> Tuple:
        """Extract """
        action_idx = self.pod[:, t[0]].repeat(self.B * self.D * self.K) # Shape [B*D*K*P-1]
        # util_idx = # Shape [B*D*K*T]
        return action_idx, util_idx


    def _extract_from_state(self, state) -> Tuple:
        """Extract and clone state variables from the state TensorDict."""
        # Vessel-related variables
        utilization = state["utilization"].clone()
        target_long_crane = state["target_long_crane"].clone()
        # # Additional variables
        total_loaded = state["total_loaded"].clone()
        total_revenue = state["total_revenue"].clone()
        total_cost = state["total_cost"].clone()
        total_rc = state["total_rc"].clone()
        # Return
        return utilization, target_long_crane, total_loaded, total_revenue, total_cost, total_rc

    def _extract_from_obs(self, obs, batch_size) -> Tuple:
        """Extract and clone state variables from the obs TensorDict."""
        current_demand = obs["current_demand"].clone().view(*batch_size, 1)
        observed_demand = obs["observed_demand"].clone().view(*batch_size, self.K, self.T)
        expected_demand = obs["expected_demand"].clone().view(*batch_size, self.K, self.T)
        std_demand = obs["std_demand"].clone().view(*batch_size, self.K, self.T)
        return current_demand, observed_demand, expected_demand, std_demand



    # Constraints
    def create_lhs_A(self, lhs_A:Tensor, action_idx:Tensor, util_idx:Tensor) -> Tensor:
        """Create lhs A_t of compact constraints: A_p x_p <= b_p"""
        lhs_A[:, :, action_idx] = self.A[:, :, util_idx].clone()
        return lhs_A

    def create_rhs(self, utilization:Tensor, observed_demand:Tensor, batch_size) -> Tensor:
        """Create rhs of compact constraints: A_p x_p <= b_p"""
        # Get rhs = [observed_demand, LM_ub, LM_lb, VM_ub, VM_lb]
        # Stability constraints
        A = self.swap_signs_stability.view(-1, 1, 1,) * self.A.clone()
        rhs = utilization.view(*batch_size, -1) @ A.view(self.n_constraints, -1).T
        # Demand constraint
        rhs[:, :self.n_demand] = current_demand.view(-1, 1)
        return rhs
