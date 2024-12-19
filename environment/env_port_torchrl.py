import time
from typing import Optional, Iterable, List, Tuple, Dict

import torch
import torch as th
from torch import Tensor
from tensordict.tensordict import TensorDict
from torchrl.envs.common import EnvBase
from torchrl.data import (
    Bounded,
    Unbounded,
    Composite,
)

# Modules
from environment.env_torchrl import MasterPlanningEnv
from environment.generator import MPP_Generator
from environment.utils import *

class PortMasterPlanningEnv(MasterPlanningEnv):
    """Master Planning Problem environment."""
    name = "port_mpp"
    batch_locked = False

    def __init__(self, device="cuda", batch_size=[], td_gen=None, **kwargs):
        super().__init__(device=device, batch_size=batch_size)

        # Sets
        self.P = kwargs.get("ports") # Number of ports
        self.B = kwargs.get("bays")  # Number of bays
        self.D = kwargs.get("decks") # Number of decks
        self.T = int((self.P ** 2 - self.P) / 2) # Number of (POL,POD) transports
        self.CC = kwargs.get("customer_classes")  # Number of customer contracts
        self.K = kwargs.get("cargo_classes") * self.CC # Number of container classes
        self.W = kwargs.get("weight_classes")  # Number of weight classes

        # Seed
        self.seed = kwargs.get("seed")
        if self.seed is None:
            self.seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(self.seed)

        # Init fns
        # todo: perform big clean-up here!
        self.float_type = kwargs.get("float_type", th.float32)
        self.demand_uncertainty = kwargs.get("demand_uncertainty", False)
        self._compact_form_shapes()
        self.generator = MPP_Generator(**kwargs)
        if td_gen == None:
            self.td_gen = self.generator(batch_size=batch_size,)
        self._make_spec(self.td_gen)
        self.zero = th.tensor([0], device=self.generator.device, dtype=self.float_type)
        self.padding = th.tensor([self.P-1], device=self.generator.device, dtype=th.int32)

        # Parameters:
        # Transport and cargo characteristics
        self.ports = torch.arange(self.P, device=self.generator.device)
        self.transport_idx = get_transport_idx(self.P, device=self.generator.device)

        # Capacity in TEU per location (bay,deck)
        c = kwargs.get("capacity")
        self.capacity = th.full((self.B, self.D,), c[0], device=self.generator.device, dtype=self.float_type)
        self.total_capacity = th.sum(self.capacity)
        self.teus = th.arange(1, self.K // (self.CC * self.W) + 1, device=self.generator.device, dtype=self.float_type) \
            .repeat_interleave(self.W).repeat(self.CC)
        self.teus_episode = th.cat([self.teus.repeat(self.T)])
        self.weights = th.arange(1, self.W + 1, device=self.generator.device, dtype=self.float_type).repeat(self.K // self.W)
        self.duration = self.transport_idx[..., 1] - self.transport_idx[..., 0]

        # Revenue and costs
        self.revenues_matrix = self._precompute_revenues()
        self.ho_costs = kwargs.get("hatch_overstowage_costs") #* th.mean(self.revenues)
        self.lc_costs = kwargs.get("long_crane_costs") #* th.mean(self.revenues)
        self.ho_mask = kwargs.get("hatch_overstowage_mask")
        self.CI_target = kwargs.get("CI_target")

        # Transport sets
        self._precompute_transport_sets()
        # Step ordering: descending distance, longterm > spot; ascending TEU, ascending weight
        self.ordering = kwargs.get("episode_order")
        if self.ordering == "max_distance_then_priority":
            self.ordered_steps = self._precompute_order_max_distance_then_priority()
        elif self.ordering == "greedy_revenue":
            self.ordered_steps = self._precompute_order_greedy_revenue()
        elif self.ordering == "priority_then_greedy":
            raise NotImplementedError("Priority then greedy ordering is not implemented yet.")
        else:
            self.ordered_steps = self._precompute_order_standard()
        self.ordered_steps = th.cat([self.ordered_steps, self.padding])
        self.k, self.tau = get_k_tau_pair(self.ordered_steps, self.K)
        self.pol, self.pod = get_pol_pod_pair(self.tau, self.P)
        # self.pol, self.pod, self.k, self.tau = self._add_padding(self.pol, self.pod, self.k, self.tau)
        self.revenues = self.revenues_matrix[self.k, self.tau]
        self._precompute_transport_sets_episode()
        self.next_port_mask = self._precompute_next_port_mask()
        self.transform_tau_to_pol = get_pols_from_transport(self.transport_idx, self.P, dtype=self.float_type)
        self.transform_tau_to_pod = get_pods_from_transport(self.transport_idx, self.P, dtype=self.float_type)

        # Stability
        self.stab_delta = kwargs.get("stability_difference")
        self.LCG_target = kwargs.get("LCG_target")
        self.VCG_target = kwargs.get("VCG_target")
        self.longitudinal_position = th.arange(1/self.B, self.B * 2/self.B, 2/self.B,
                                               device=self.generator.device, dtype=self.float_type)
        self.vertical_position = th.arange(1/self.D, self.D * 2/self.D, 2/self.D,
                                           device=self.generator.device, dtype=self.float_type)
        self.lp_weight = th.einsum("d, b -> bd", self.weights, self.longitudinal_position).unsqueeze(0)
        self.vp_weight = th.einsum("d, c -> cd", self.weights, self.vertical_position).unsqueeze(0)
        self.stability_params_lhs = self._precompute_stability_parameters()

        # Constraints
        # todo clean-up
        ones_cons = th.ones(self.n_constraints, device=self.generator.device, dtype=self.float_type)
        self.constraint_signs = ones_cons.clone()
        indices_to_multiply = th.tensor([-3, -1], device=self.generator.device)
        self.constraint_signs[indices_to_multiply] *= -1
        self.swap_signs_stability = -ones_cons.clone() # swap signs for constraints
        self.swap_signs_stability[0] *= -1 # only demand constraint is positive
        self.A = self._create_constraint_matrix(shape=(self.n_constraints, self.n_action, self.T, self.K))

    def _make_spec(self, td:TensorDict = None) -> None:
        """Define the specs for observations, actions, rewards, and done flags."""
        batch_size = td.batch_size
        observation_spec = Unbounded(shape=(*batch_size,289), dtype=self.float_type) # 287, 307
        state_spec = Composite(
            utilization=Unbounded(shape=(*batch_size,self.B*self.D*self.T*self.K), dtype=self.float_type),
            target_long_crane=Unbounded(shape=(*batch_size,1), dtype=self.float_type),
            long_crane_moves_discharge=Unbounded(shape=(*batch_size,self.B-1), dtype=self.float_type),
            total_loaded=Unbounded(shape=(*batch_size,1), dtype=self.float_type),
            total_revenue=Unbounded(shape=(*batch_size,1), dtype=self.float_type),
            total_cost=Unbounded(shape=(*batch_size,1), dtype=self.float_type),
            total_rc=Unbounded(shape=(*batch_size,self.B*self.D), dtype=self.float_type),
            total_violation=Unbounded(shape=(*batch_size,self.n_constraints), dtype=self.float_type),
            current_demand=Unbounded(shape=(*batch_size, 1), dtype=torch.float32),
            observed_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=torch.float32),
            expected_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=torch.float32),
            std_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=torch.float32),
            realized_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=torch.float32),
            # Vessel
            lcg=Unbounded(shape=(*batch_size, 1), dtype=self.float_type),
            vcg=Unbounded(shape=(*batch_size, 1), dtype=self.float_type),
            residual_capacity=Unbounded(shape=(*batch_size, self.B * self.D),  dtype=self.float_type),
            residual_lc_capacity=Unbounded(shape=(*batch_size, self.B - 1), dtype=self.float_type),
            pol_location=Unbounded(shape=(*batch_size, self.B * self.D * self.P), dtype=self.float_type),
            pod_location=Unbounded(shape=(*batch_size, self.B * self.D * self.P), dtype=self.float_type),
            agg_pol_location=Unbounded(shape=(*batch_size, self.B * self.D), dtype=self.float_type),
            agg_pod_location=Unbounded(shape=(*batch_size, self.B * self.D), dtype=self.float_type),
            shape=batch_size,
        )
        self.observation_spec = Composite(
            # Obs and state
            observation=observation_spec,
            state=state_spec,

            # Action, batch_updates, timestep
            action=Unbounded(shape=(*batch_size, self.B * self.D), dtype=self.float_type),
            batch_updates=Unbounded(shape=(*batch_size,1), dtype=torch.float32),
            timestep=Unbounded(shape=(*batch_size,1), dtype=th.int64),

            # Performance
            profit=Unbounded(shape=(*batch_size,1), dtype=torch.float32),
            revenue=Unbounded(shape=(*batch_size,1), dtype=torch.float32),
            cost=Unbounded(shape=(*batch_size,1), dtype=torch.float32),

            # Demand
            realized_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=torch.float32),
            observed_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=torch.float32),
            expected_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=torch.float32),
            std_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=torch.float32),
            init_expected_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=torch.float32),

            # Constraints
            clip_min=Unbounded(shape=(*batch_size,self.B*self.D),dtype=self.float_type),
            clip_max=Unbounded(shape=(*batch_size,self.B*self.D),dtype=self.float_type),
            lhs_A=Unbounded(shape=(*batch_size,self.n_constraints,self.B*self.D),dtype=self.float_type),
            rhs=Unbounded(shape=(*batch_size,self.n_constraints),dtype=self.float_type),
            violation=Unbounded(shape=(*batch_size,self.n_constraints),dtype=self.float_type),
            shape=batch_size,
        )
        self.action_spec = Bounded(
            shape=(*batch_size, self.B*self.D),  # Define shape as needed
            low=0.0,
            high=50.0,  # Define high value as needed
            dtype=self.float_type,
        )
        self.reward_spec = Unbounded(shape=(*batch_size,1,))
        self.done_spec = Unbounded(shape=(*batch_size,1,), dtype=th.bool)

    def _check_done(self, t: Tensor) -> Tensor:
        """Determine if the episode is done based on the state."""
        return (t == (self.T*self.K) - 1).view(-1,1)

    def _step(self, td: TensorDict) -> TensorDict:
        """Perform a step in the environment and return the next state."""
        # todo: perform big clean-up here!
        ## Extraction and precompute
        batch_size = td.batch_size
        action, t, lhs_A, rhs = self._extract_from_td(td, batch_size)
        pol, pod, tau, k, rev = self._extract_cargo_parameters_for_step(t[0])
        utilization, target_long_crane, long_crane_moves_discharge, \
            demand_state, total_metrics = self._extract_from_state(td["state"], batch_size)

        ## Current state
        # Action clipping
        clip_max = td["clip_max"].clamp(max=demand_state["current_demand"]).view(*batch_size, self.B, self.D)
        action = action.clamp(min=td["clip_min"].view(*batch_size, self.B, self.D), max=clip_max)

        # Check done, update utilization, and compute violation
        done = self._check_done(t)
        utilization = self._update_state_loading(action, utilization, tau, k,)
        # todo: improve readability of this part
        if lhs_A.dim() == 2:
            violation = lhs_A @ action.view(*batch_size, -1) - rhs
        elif lhs_A.dim() == 3:
            violation = torch.bmm(lhs_A, action.view(*batch_size, -1, 1)) - rhs.unsqueeze(-1)
        else:
            raise ValueError("lhs_A has wrong dimensions.")
        violation = torch.clamp(violation, min=0).view(*batch_size, -1)
        total_metrics["total_violation"] += violation.clone()

        # Compute long crane moves
        long_crane_moves_load = self._compute_long_crane(utilization, pol)
        # Compute od-pairs
        pol_locations, pod_locations = self._compute_pol_pod_locations(utilization)
        agg_pol_location, agg_pod_location = self._aggregate_pol_pod_location(pol_locations, pod_locations)
        # Compute total loaded
        sum_action = action.sum(dim=(-2, -1)).unsqueeze(-1)
        total_metrics["total_loaded"] += sum_action

        ## Reward
        revenue = th.clamp(sum_action, min=th.zeros_like(sum_action), max=demand_state["current_demand"]) * self.revenues[t[0]]
        # revenue = sum_action * self.revenues[t[0]]
        profit = revenue.clone()
        if self.next_port_mask[t].any():
            # Compute aggregated: overstowage and long crane excess
            overstowage = self._compute_hatch_overstowage(utilization, pol)
            excess_crane_moves = th.clamp(long_crane_moves_load + long_crane_moves_discharge - target_long_crane.view(-1, 1), min=0)
            # Compute costs
            ho_costs = overstowage.sum(dim=-1, keepdim=True) * self.ho_costs
            lc_costs = excess_crane_moves.sum(dim=-1, keepdim=True) * self.lc_costs
            cost = ho_costs + lc_costs
            profit -= cost
        else:
            cost = th.zeros_like(profit)

        ## Transition to next step
        # Update next state
        t = t + 1
        next_state_dict = self._update_next_state(utilization, target_long_crane,
                                                  long_crane_moves_load, long_crane_moves_discharge,
                                                  demand_state, t, batch_size)
        if not done.any():
            # Update feasibility: lhs_A, rhs, clip_max based on next state
            lhs_A = self.create_lhs_A(t,)
            rhs = self.create_rhs(next_state_dict["utilization"], next_state_dict["current_demand"], batch_size)

            # Express residual capacity in teu
            residual_capacity = th.clamp(self.capacity - next_state_dict["utilization"].sum(dim=-2)
                                         @ self.teus, min=self.zero) / self.capacity
        else:
            residual_capacity = th.zeros_like(td["state"]["residual_capacity"], dtype=self.float_type).view(*batch_size, self.B, self.D, )

        # # Update action mask
        # # action_mask[t] = 0
        # mask_condition = (t[0] % self.K == 0)
        # min_pod = self._compute_min_pod(pol_locations, pod_locations )
        # new_mask = self._prevent_HO_mask(action_mask, pod, pod_locations, min_pod)
        # action_mask = th.where(mask_condition, new_mask, action_mask)

        # # Only for final port
        if t[0] == self.T*self.K:
            # Compute last port long crane excess
            lc_moves_last_port = self._compute_long_crane(utilization, pol)
            lc_excess_last_port = th.clamp(lc_moves_last_port - next_state_dict["target_long_crane"].view(-1,1), min=0)

            # Compute metrics
            excess_crane_moves += lc_excess_last_port
            lc_cost_ = lc_excess_last_port.sum(dim=-1, keepdim=True) * self.lc_costs
            profit -= lc_cost_
            cost += lc_cost_
            lc_costs += lc_cost_
            # set t back to fit clip max
            t -= 1

        # # Track metrics
        total_metrics["total_rc"] += residual_capacity.view(*batch_size, -1)
        total_metrics["total_revenue"] += revenue
        total_metrics["total_cost"] += cost
        # Normalize revenue \in [0,1]:
        # revenue_norm = rev_t / max(rev_t) * min(q_t, sum(a_t)) / q_t
        normalize_revenue = self.revenues.max() * demand_state["current_demand"]
        # Normalize accumulated cost \in [0, t_cost], where t_cost is the time at which we evaluate cost:
        # cost_norm = cost_{t_cost} / E[q_t]
        # todo: check proper cost normalization
        normalize_cost = demand_state["realized_demand"].view(*batch_size, -1)[...,:t[0]].sum(dim=-1, keepdims=True) / t[0]
        # Normalize reward: r_t = revenue_norm - cost_norm
        # We have spikes over delayed costs at specific time steps.
        reward = (revenue.clone() / normalize_revenue) #- (cost.clone() / normalize_cost)
        # reward = profit

        # Update td output
        obs = self._get_observation(next_state_dict, residual_capacity,  agg_pol_location, agg_pod_location, t, batch_size)

        clip_max = (residual_capacity * self.capacity.unsqueeze(0) / self.teus_episode[t].view(*batch_size, 1, 1))
        # todo: reduce number of outputs in td (way too much now)
        out =  TensorDict({
            "state":{
                # Vessel
                "utilization": next_state_dict["utilization"].view(*batch_size, self.B*self.D*self.T*self.K),
                "target_long_crane": next_state_dict["target_long_crane"],
                "long_crane_moves_discharge": next_state_dict["long_crane_moves_discharge"].view(*batch_size, self.B - 1),

                # # Performance
                "total_loaded": total_metrics["total_loaded"],
                "total_revenue": total_metrics["total_revenue"],
                "total_cost": total_metrics["total_cost"],
                "total_rc": total_metrics["total_rc"],
                "total_violation": total_metrics["total_violation"],
                # Demand
                "current_demand": next_state_dict["current_demand"].view(*batch_size, 1),
                "observed_demand": next_state_dict["observed_demand"].view(*batch_size, self.T * self.K),
                "expected_demand": next_state_dict["expected_demand"].view(*batch_size, self.T * self.K),
                "std_demand": next_state_dict["std_demand"].view(*batch_size, self.T * self.K),
                "realized_demand": next_state_dict["realized_demand"].view(*batch_size, self.T * self.K),
                # Vessel (in range [0, 1])
                "lcg": next_state_dict["lcg"].view(*batch_size, 1),
                "vcg": next_state_dict["vcg"].view(*batch_size, 1),
                "residual_capacity": residual_capacity.view(*batch_size, self.B * self.D),
                "residual_lc_capacity": next_state_dict["residual_lc_capacity"].view(*batch_size, self.B - 1),
                "pol_location": pol_locations.view(*batch_size, self.B * self.D * self.P).to(self.float_type),
                "pod_location": pod_locations.view(*batch_size, self.B * self.D * self.P).to(self.float_type),
                "agg_pol_location": agg_pol_location.view(*batch_size, self.B * self.D),
                "agg_pod_location": agg_pod_location.view(*batch_size, self.B * self.D),
            },
            "observation":obs,
            "realized_demand": td["realized_demand"].view(*batch_size, self.T * self.K),
            "observed_demand": td["observed_demand"].view(*batch_size, self.T*self.K),
            "expected_demand": td["expected_demand"].view(*batch_size, self.T*self.K),
            "std_demand": td["std_demand"].view(*batch_size, self.T*self.K),
            "init_expected_demand": td["init_expected_demand"].view(*batch_size, self.T * self.K),
            "batch_updates": td["batch_updates"] + 1,

            # # Feasibility and constraints
            "lhs_A": lhs_A.view(*batch_size, self.n_constraints, self.B*self.D),
            "rhs": rhs.view(*batch_size, self.n_constraints),
            "violation": violation.view(*batch_size, self.n_constraints),
            # in containers
            "clip_min": th.zeros_like(residual_capacity, dtype=self.float_type).view(*batch_size, self.B*self.D),
            "clip_max": clip_max.view(*batch_size, self.B*self.D),
            # Profit metrics
            "profit": profit,
            "revenue": revenue,
            "cost": cost,
            # Action, reward, done and step
            "action": action.view(*batch_size, self.B*self.D),
            "reward": reward,
            "done": done,
            "timestep":t,
        }, td.shape)
        return out

    def _reset(self,  td: Optional[TensorDict] = None, seed:Optional=None) -> TensorDict:
        """Reset the environment to the initial state."""
        # Extract batch_size from td if it exists
        batch_size = getattr(td, 'batch_size', self.batch_size)
        if td is None or td.is_empty():
            # Generate new demand
            td = self.generator(batch_size=batch_size, td=self.td_gen)
        else:
            #todo: implement non-iid demand generation
            pass

        # Reordering on demand
        realized_demand = td["realized_demand"].view(*batch_size, self.T, self.K).clone()
        observed_demand = td["observed_demand"].view(*batch_size, self.T, self.K).clone()
        expected_demand = td["expected_demand"].view(*batch_size, self.T, self.K).clone()
        std_demand = td["std_demand"].view(*batch_size, self.T, self.K).clone()

        # Initialize
        # Parameters
        device = td.device
        if batch_size == torch.Size([]):
            t = th.zeros(1, dtype=th.int64, device=device)
            residual_capacity = self.capacity / self.teus[t[0]]
        else:
            t = th.zeros(*batch_size, dtype=th.int64, device=device)
            residual_capacity = (self.capacity / self.teus[t[0]]).unsqueeze(0).repeat(*batch_size, 1, 1)

        pol, pod, tau, k, rev = self._extract_cargo_parameters_for_step(t[0])
        # Action mask
        action_mask = th.ones((*batch_size, self.B*self.D), dtype=th.bool, device=device)
        # Vessel state
        utilization = th.zeros((*batch_size, self.B, self.D, self.T, self.K), device=device, dtype=self.float_type)
        target_long_crane = self._compute_target_long_crane(
            realized_demand.to(self.float_type), self.moves_idx[t[0]]).view(*batch_size, 1)
        residual_lc_capacity = target_long_crane.repeat(1, self.B - 1)
        locations_utilization = th.zeros_like(action_mask, dtype=self.float_type)
        port_locations = th.zeros((*batch_size, self.B*self.D*self.P), dtype=self.float_type)
        # Demand
        current_demand = observed_demand[..., tau, k].view(*batch_size, 1).clone() # clone to prevent in-place!
        observed_demand[..., tau, k] = 0
        expected_demand[..., tau, k] = 0
        std_demand[..., tau, k] = 0
        # Constraints
        lhs_A = self.create_lhs_A(t,)
        rhs = self.create_rhs(utilization.to(self.float_type), current_demand, batch_size)

        # Init tds - state: internal state
        initial_state = TensorDict({
            # Vessel
            "utilization": utilization.view(*batch_size, self.B*self.D*self.T*self.K),
            "target_long_crane": target_long_crane,
            "long_crane_moves_discharge": th.zeros_like(residual_lc_capacity).view(*batch_size, self.B - 1),
            # Performance
            "total_loaded": th.zeros_like(current_demand, dtype=self.float_type),
            "total_revenue": th.zeros_like(current_demand, dtype=self.float_type),
            "total_cost": th.zeros_like(current_demand, dtype=self.float_type),
            "total_rc": th.zeros_like(locations_utilization),
            "total_violation": th.zeros_like(rhs, dtype=self.float_type),
            # Demand
            "current_demand": current_demand.view(*batch_size, 1),
            "observed_demand": observed_demand.view(*batch_size, self.T * self.K),
            "expected_demand": expected_demand.view(*batch_size, self.T * self.K),
            "std_demand": std_demand.view(*batch_size, self.T * self.K),
            "realized_demand": realized_demand.view(*batch_size, self.T * self.K),
            # Vessel
            "residual_capacity": th.ones_like(residual_capacity).view(*batch_size, self.B * self.D),
            "lcg": th.ones_like(t, dtype=self.float_type).view(*batch_size, 1),
            "vcg": th.ones_like(t, dtype=self.float_type).view(*batch_size, 1),
            "residual_lc_capacity": residual_lc_capacity.view(*batch_size, self.B - 1),
            "pol_location": th.zeros_like(port_locations, dtype=self.float_type),
            "pod_location": th.zeros_like(port_locations, dtype=self.float_type),
            "agg_pol_location": th.zeros_like(locations_utilization),
            "agg_pod_location": th.zeros_like(locations_utilization),
        }, batch_size=batch_size, device=device,)

        # Init tds - obs: observed by embeddings
        obs = self._get_observation(initial_state, residual_capacity, initial_state["agg_pol_location"], initial_state["agg_pod_location"], t, batch_size)

        # Init tds - full td
        out = TensorDict({
            # State + obs
            "state": initial_state,
            "observation": obs,
            # # Action mask
            "action": th.zeros_like(action_mask, dtype=self.float_type),
            # "action_mask": action_mask.view(*batch_size, -1),
            # # Constraints
            "clip_min": th.zeros_like(residual_capacity, dtype=self.float_type).view(*batch_size, self.B * self.D, ),
            "clip_max": residual_capacity.view(*batch_size, self.B * self.D),
            "lhs_A": lhs_A.view(*batch_size, self.n_constraints, self.B * self.D),
            "rhs":  rhs.view(*batch_size, self.n_constraints),
            "violation": th.zeros_like(rhs, dtype=self.float_type),
            # Performance
            "profit": th.zeros_like(t, dtype=self.float_type).view(*batch_size, 1),
            "revenue": th.zeros_like(t, dtype=self.float_type).view(*batch_size, 1),
            "cost": th.zeros_like(t, dtype=self.float_type).view(*batch_size, 1),
            # Reward, done and step
            "done": th.zeros_like(t, dtype=th.bool),
            "timestep": t,
            # from generator
            "realized_demand": td["realized_demand"].view(*batch_size, self.T * self.K),
            "observed_demand": td["observed_demand"].view(*batch_size, self.T * self.K),
            "expected_demand": td["expected_demand"].view(*batch_size, self.T * self.K),
            "std_demand": td["std_demand"].view(*batch_size, self.T * self.K),
            "init_expected_demand": td["init_expected_demand"].view(*batch_size, self.T * self.K),
            "batch_updates": td["batch_updates"] + 1,
        }, batch_size=batch_size, device=device,)
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng
