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
from environment.generator import MPP_Generator
from environment.utils import *

class MasterPlanningEnv(EnvBase):
    """Master Planning Problem environment."""
    name = "mpp"
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
        pol_locations, pod_locations = compute_pol_pod_locations(utilization,
                                                                 self.transform_tau_to_pol, self.transform_tau_to_pod)
        agg_pol_location, agg_pod_location = aggregate_pol_pod_location(pol_locations, pod_locations,
                                                                        self.ports, self.P, self.float_type)
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

    # Extraction functions
    def _extract_from_td(self, td, batch_size) -> Tuple:
        """Extract action, reward and step from the TensorDict."""
        # Must clone to avoid in-place operations!
        timestep = td["timestep"].view(-1)
        action = td["action"].view(*batch_size, self.B, self.D,)
        # action_mask = td["action_mask"].clone()
        lhs_A = td["lhs_A"].clone()
        rhs = td["rhs"].clone()
        return action, timestep, lhs_A, rhs

    def _extract_cargo_parameters_for_step(self, t) -> Tuple:
        """Extract cargo-related parameters"""
        pol = self.pol[t]
        pod = self.pod[t]
        k = self.k[t]
        tau = self.tau[t]
        rev_t = self.revenues[t]
        return pol, pod, tau, k, rev_t

    def _extract_from_state(self, state, batch_size) -> Tuple:
        """Extract and clone state variables from the state TensorDict."""
        # Vessel-related variables
        utilization = state["utilization"].view(*batch_size, self.B, self.D, self.T, self.K).clone()
        target_long_crane = state["target_long_crane"].view(*batch_size, 1).clone()
        long_crane_moves_discharge = state["long_crane_moves_discharge"].view(*batch_size, self.B-1).clone()
        # Demand-related variables
        demand = {
            # clones are needed to prevent in-place
            "current_demand": state["current_demand"].view(*batch_size, -1).clone(),
            "observed_demand": state["observed_demand"].view(*batch_size, self.T, self.K).clone(),
            "expected_demand": state["expected_demand"].view(*batch_size, self.T, self.K).clone(),
            "std_demand": state["std_demand"].view(*batch_size, self.T, self.K).clone(),
            "realized_demand": state["realized_demand"].view(*batch_size, self.T, self.K).clone(),
        }
        # Additional variables
        total_metrics = {
            "total_loaded": state["total_loaded"].view(*batch_size, 1).clone(),
            "total_revenue": state["total_revenue"].view(*batch_size, 1).clone(),
            "total_cost": state["total_cost"].view(*batch_size, 1).clone(),
            "total_rc": state["total_rc"].view(*batch_size, self.B * self.D).clone(),
            "total_violation": state["total_violation"].view(*batch_size, self.n_constraints).clone()
        }
        return utilization, target_long_crane, long_crane_moves_discharge, demand, total_metrics

    def _get_observation(self, next_state_dict, residual_capacity,
                         agg_pol_location, agg_pod_location, t, batch_size) -> Tensor:
        """Get observation from the TensorDict."""
        # normalize demand
        max_demand = next_state_dict["realized_demand"].max()
        # current_demand = next_state_dict["current_demand"].view(*batch_size, 1) / max_demand
        # observed_demand = next_state_dict["observed_demand"].view(*batch_size, self.T * self.K) / max_demand
        # expected_demand = next_state_dict["expected_demand"].view(*batch_size, self.T * self.K) / max_demand
        # std_demand = next_state_dict["std_demand"].view(*batch_size, self.T * self.K) / max_demand
        # print("-"*50)
        # print(f"mean_current_demand t:{t[0]}", current_demand.mean(dim=0))
        # print(f"mean_observed_demand t:{t[0]}", observed_demand.mean(dim=0))
        # print(f"mean_expected_demand t:{t[0]}", expected_demand.mean(dim=0))
        # print(f"mean_std_demand t:{t[0]}", std_demand.mean(dim=0))
        # lcg = next_state_dict["lcg"].view(*batch_size, 1)
        # vcg = next_state_dict["vcg"].view(*batch_size, 1)
        # print(f"mean_lcg t:{t[0]}", lcg.mean(dim=0))
        # print(f"mean_vcg t:{t[0]}", vcg.mean(dim=0))
        # residual_capacity = residual_capacity / self.capacity.unsqueeze(0)
        # residual_lc_capacity = (next_state_dict["residual_lc_capacity"]/next_state_dict["target_long_crane"].unsqueeze(0))
        # print(f"mean_residual_capacity t:{t[0]}", residual_capacity.mean(dim=0))
        # print(f"mean_residual_lc_capacity t:{t[0]}", residual_lc_capacity.mean(dim=0))
        # agg_pol_location = agg_pol_location / (self.P)
        # agg_pod_location = agg_pod_location / (self.P)
        # print(f"mean_agg_pol_location t:{t[0]}", agg_pol_location.mean(dim=0).T)
        # print(f"mean_agg_pod_location t:{t[0]}", agg_pod_location.mean(dim=0).T)

        return th.cat([
            t.view(*batch_size, 1) / (self.T * self.K),
            next_state_dict["current_demand"].view(*batch_size, 1) / max_demand,
            next_state_dict["observed_demand"].view(*batch_size, self.T * self.K) / max_demand,
            next_state_dict["expected_demand"].view(*batch_size, self.T * self.K) / max_demand,
            next_state_dict["std_demand"].view(*batch_size, self.T * self.K) / max_demand,
            # Vessel
            next_state_dict["lcg"].view(*batch_size, 1),
            next_state_dict["vcg"].view(*batch_size, 1),
            (residual_capacity/self.capacity.unsqueeze(0)).view(*batch_size, self.B * self.D),
            (next_state_dict["residual_lc_capacity"]/next_state_dict["target_long_crane"].unsqueeze(0)).view(*batch_size, self.B - 1),
            agg_pol_location.view(*batch_size, self.B * self.D) / (self.P),
            agg_pod_location.view(*batch_size, self.B * self.D) / (self.P),
        ], dim=-1)

    # Reward/costs functions
    def _get_reward(self, td, utilizations) -> Tensor:
        """Compute total profit for episode"""
        return (td["state"]["total_revenue"] - td["state"]["total_cost"])/1000

    def _compute_min_pod(self,pod_locations: Tensor) -> Tensor:
        """Compute min_pod based on utilization"""
        min_pod = th.argmax(pod_locations.to(self.float_type), dim=-1)
        min_pod[min_pod == 0] = self.P
        return min_pod

    def _compute_overstowage_loading_indicator(self, pod_locations: Tensor,) -> Tensor:
        """Indicate if overstowage is present due to discharge by: max(pod_d0) > min(pod_d1)
        Output: Positive integer tensor with shape [batch, bay, deck]"""
        # Get the number of pods
        pods = torch.arange(self.P, device=self.generator.device)
        # For above deck (d=0), compute the min POD index where value is 1 (non-zero elements)
        max_pod_d0 = torch.where(pod_locations[..., 0, :] > 0, pods, float('-inf')).max(dim=-1).values
        # For below deck (d=1), compute the max POD index where value is 1 (non-zero elements)
        min_pod_d1 = torch.where(pod_locations[..., 1, :] > 0, pods, float('inf')).min(dim=-1).values
        # Stack the results into the desired shape [batch, bay, 2]
        output = torch.stack((max_pod_d0, min_pod_d1), dim=-1)
        return output

    def _compute_hatch_overstowage(self, utilization: Tensor, single_pol: Tensor) -> Tensor:
        """Get hatch overstowage based on ac_transport and moves"""
        # Get indices
        ac_transport = self.remain_on_board_transport[single_pol]
        moves = self.moves_idx[single_pol]
        # Compute hatch overstowage
        hatch_open = utilization[..., 1:, moves, :].sum(dim=(-3, -2, -1)) > 0
        hatch_overstowage = utilization[..., :1, ac_transport, :].sum(dim=(-3, -2, -1)) * hatch_open
        return hatch_overstowage

    def _compute_target_long_crane(self, realized_demand: Tensor, moves: Tensor) -> Tensor:
        """Compute target crane moves per port:
        - Get total crane moves per port: load_moves + discharge_moves
        - Get optimal crane moves per adjacent bay by: 2 * total crane moves / B
        - Get adjacent capacity by: sum of capacity of adjacent bays
        - Get max capacity of adjacent bays by: max of adjacent capacity

        Return element-wise minimum of optimal crane moves and max capacity"""
        # Calculate optimal crane moves based per adjacent bay based on loading and discharging
        total_crane_moves = realized_demand[..., moves, :].sum(dim=(-1,-2))
        # Compute adjacent capacity and max capacity
        max_capacity = ((self.capacity[:-1] + self.capacity[1:]).sum(dim=-1)).max()
        # Compute element-wise minimum of crane moves and target long crane
        optimal_crane_moves_per_adj_bay = 2 * total_crane_moves / self.B
        return self.CI_target * th.minimum(optimal_crane_moves_per_adj_bay, max_capacity)

    def _compute_long_crane(self, utilization: Tensor, single_pol: Tensor) -> Tensor:
        """Compute long crane moves based on utilization"""
        moves_idx = self.moves_idx[single_pol].to(utilization.dtype).view(1, 1, 1, self.T, 1)
        moves_per_bay = (utilization * moves_idx).sum(dim=(2, 3, 4))
        return moves_per_bay[..., :-1] + moves_per_bay[..., 1:]

    def _prevent_HO_mask(self, mask:Tensor, pod: Tensor,pod_locations:Tensor, min_pod:Tensor) -> Tensor:
        """
        Mask action to prevent hatch overstowage. Deck indices: 0 is above-deck, 1 is below-deck.

        Variables:
            - Utilization: Current state of onboard cargo (bay,deck,cargo_class,transport)
            - POD_locations: Indicator to show PODs loaded in locations (bay,deck,P)
            - Min_pod: Minimum POD location based on POD_locations (bay,deck)

        Utilization is filled/emptied incrementally. Hence, we have certain circumstances to observe utilization:
            - Step after reset: Utilization is empty
            - Step of new POL:  Discharge utilization destined for new POL
            - Any other step:   Load utilization of current cargo_class and transport

        Two ways to prevent hatch overstowage:
        - If above-deck is empty, we can freely place below-deck. Otherwise, we need to restow above-deck directly.
            E.g.:
                    | 3 | 3 | o |
                    +---+---+---+
                    | x | x | o |   , where int is min_pod of location, x is blocked location, o is open location

        - Above-deck actions are allowed if current POD <= min_pod below-deck. Otherwise, we need to restow
            above-deck when below-deck will be discharged.
            E.g.:   POD = 2
                    | x | o | o |
                    +---+---+---+
                    | 1 | 2 | 3 |   , where int is min_pod of location, x is blocked location, o is open location
        """
        # Create mask:
        mask = mask.view(-1, self.B, self.D)
        # Action below-deck (d=1) allowed if above-deck (d=0) is empty
        mask[..., 1] = pod_locations[..., 0, :].sum(dim=-1) == 0
        # Action above-deck (d=0) allowed if POD <= min_pod below deck (d=1)
        mask[..., 0] = pod.unsqueeze(-1) <= min_pod[..., 1]
        return mask.view(-1, self.B*self.D)

    # Update state
    def _update_state_loading(self, action: Tensor, utilization: Tensor,
                              tau:Tensor, k:Tensor,) -> Tensor:
        """Transition to the next state based on the action."""
        new_utilization = utilization.clone()
        new_utilization[..., tau, k] = action
        return new_utilization

    def _update_next_state(self, utilization: Tensor, target_long_crane:Tensor,
                           long_crane_moves_load:Tensor, long_crane_moves_discharge:Tensor,
                           demand_state:Dict, t:Tensor, batch_size) -> Dict[str, Tensor]:
        """Update next state, following options:
        - Next step moves to new port POL+1
        - Next step moves to new transport (POL, POD-1)
        - Last step of episode; compute excess crane moves at last port
        """
        # Get cargo parameters
        pol, pod, tau, k, rev = self._extract_cargo_parameters_for_step(t[0])

        # Check next port with t - 1
        load_idx, disc_idx, moves_idx = self._precompute_for_step(pol)
        # Next port with discharging; Update utilization, observed demand and target long crane
        if self.next_port_mask[t-1].any():
            long_crane_moves_load = torch.zeros_like(long_crane_moves_load)
            long_crane_moves_discharge = self._compute_long_crane(utilization, pol)
            utilization = update_state_discharge(utilization, disc_idx)
            target_long_crane = self._compute_target_long_crane(
                demand_state["realized_demand"], moves_idx).view(*batch_size, 1)
            if self.demand_uncertainty:
                demand_state["observed_demand"][..., load_idx, :] = demand_state["realized_demand"][..., load_idx, :]

        # Update observed and expected demand by setting to 0
        demand_state["observed_demand"][..., tau, k] = 0
        demand_state["expected_demand"][..., tau, k] = 0
        demand_state["std_demand"][..., tau, k] = 0

        # Update residual lc capacity: target - actual load and discharge moves
        long_crane_moves = long_crane_moves_load + long_crane_moves_discharge
        residual_lc_capacity = (target_long_crane - long_crane_moves).clamp(min=0)

        # Compute stability
        location_weight = (utilization * self.weights.view(1,1,1,1,-1)).sum(dim=(-2,-1))
        total_weight = location_weight.sum(dim=(1,2))
        lcg = (location_weight * self.longitudinal_position.view(1, -1, 1)).sum(dim=(1,2)) / total_weight
        vcg = (location_weight * self.vertical_position.view(1, 1, -1)).sum(dim=(1,2)) / total_weight
        # Get output
        return {
            "current_demand": demand_state["realized_demand"][..., tau, k],
            "observed_demand": demand_state["observed_demand"],
            "expected_demand": demand_state["expected_demand"],
            "std_demand":demand_state["std_demand"],
            "realized_demand": demand_state["realized_demand"],
            "utilization": utilization,
            "location_utilization": (utilization * self.teus.view(1,1,1,1,-1)).sum(dim=(-2,-1)),
            "lcg": lcg,
            "vcg": vcg,
            "target_long_crane": target_long_crane,
            "residual_lc_capacity": residual_lc_capacity,
            "long_crane_moves_discharge": long_crane_moves_discharge,
        }

    # Compact formulation
    def _compact_form_shapes(self, ):
        """Define shapes for compact form"""
        self.n_demand = 1
        self.n_stability = 4
        # self.n_cranes = self.B - 1
        self.n_action = self.B * self.D
        self.n_locations = self.B * self.D
        self.n_constraints = self.n_demand + self.n_locations + self.n_stability #+ self.n_cranes

    def _create_constraint_matrix(self, shape: Tuple[int, int, int, int], ):
        """Create constraint matrix A for compact constraints Au <= b"""
        # [1, LM-TW, TW-LM, VM-TW, TW-VM]
        A = th.ones(shape, device=self.generator.device, dtype=self.float_type)
        diag = self.teus.view(1, 1, 1, -1) * th.eye(self.n_locations, device=self.generator.device, dtype=self.float_type).view(self.n_locations, self.B*self.D, 1, 1)
        A[self.n_demand:self.n_locations + self.n_demand,] *= diag
        A *= self.constraint_signs.view(-1, 1, 1, 1)
        A[self.n_locations + self.n_demand:self.n_locations + self.n_demand + self.n_stability] *= self.stability_params_lhs.view(self.n_stability, self.B*self.D, 1, self.K,)
        return A.view(self.n_constraints, self.B*self.D, -1)

    def create_lhs_A(self, t:Tensor) -> Tensor:
        """Get A_t based on ordered timestep"""
        order_t = self.ordered_steps[t]
        lhs_A = self.A[..., order_t].permute((2, 0, 1,)).contiguous()
        return lhs_A

    def create_rhs(self, utilization:Tensor, current_demand:Tensor, batch_size) -> Tensor:
        """Create b_t based on current utilization:
        - b_t = [current_demand, capacity, LM_ub, LM_lb, VM_ub, VM_lb]
        - demand -> stepwise current demand [#]
        - capacity -> residual capacity [TEUs]
        - stability -> lower and upper bounds for LCG, VCG
        """
        # Perform matmul to get initial rhs, including:
        # note: utilization, A, teus_episode have static shapes
        A = self.swap_signs_stability.view(-1, 1, 1,) * self.A.clone() # Swap signs for constraints
        rhs = utilization.view(*batch_size, -1) @ A.view(self.n_constraints, -1).T

        # Update rhs with current demand and add teu capacity to the rhs
        rhs[..., :self.n_demand] = current_demand.view(-1, 1)
        rhs[..., self.n_demand:self.n_locations + self.n_demand] = \
            torch.clamp(rhs[..., self.n_demand:self.n_locations + self.n_demand] + self.capacity.view(1, -1),
                        min=0, max=self.capacity[0,0].item())
        return rhs

    # Precomputes
    def _precompute_order_max_distance_then_priority(self):
        """Get ordered steps with transports in descending order of distance
        Suppose (k,tau) = (k, (POL,POD), then we have the following ordered set:
        { (0, (0,P-1)), (1, (0,P-1)), ..., (K-1, (0,P-1));
          (0, (0,P-2)), (1, (0,P-2)), ..., (K-1, (0,P-2));
          ...
          (0, (1,P-1)), (1, (1,P-1)), ..., (K-1, (1,P-1));
          (0, (1,P-2)), (1, (1,P-2)), ..., (K-1, (1,P-2));
          ...
          (0, (P-2,P-1)), (1, (P-2,P-1)), ..., (K-1, (P-2,P-1)) }
        """
        # Initialize steps and idx
        steps = th.zeros(self.T*self.K, dtype=th.int64, device=self.generator.device)
        idx = 0
        # We use loops for readability and simplicity, only because it is part of initialization
        for pol in range(self.P - 1):
            for pod in range(self.P - 1, pol, -1):
                # Get the transport index of (POL,POD)
                tau = get_transport_from_pol_pod(pol, pod, self.transport_idx)
                # Create a range of k values for this combination and store the result
                steps[idx:idx + self.K] = th.arange(self.K, device=self.generator.device, dtype=self.float_type) + tau * self.K
                idx += self.K
        return steps

    def _precompute_order_greedy_revenue(self):
        """Get ordered steps with transports in descending order of revenue per capacity usage:
        - Revenue per capacity: revenue / (teus * weights * duration)"""
        # Init
        steps = th.zeros(self.T*self.K, dtype=th.int64, device=self.generator.device)
        cap_usage = torch.einsum("j,i->ij", self.duration, (self.teus*self.weights))
        self.revenue_per_capacity = self.revenues_matrix / cap_usage
        idx = 0
        for pol in range(self.P - 1):
            # Create a mask for `revenue_per_capacity` where `loads` is True
            loads = self.load_transport[pol].expand(self.K, -1)
            masked_revenues = torch.where(loads, self.revenue_per_capacity, float("-inf"))  # Use -inf to exclude non-loads

            # Get argsort indices of the max values in descending order
            sorted_indices = torch.argsort(masked_revenues.T.flatten(), descending=True)
            # Filter out indices corresponding to -inf
            valid_indices = sorted_indices[masked_revenues.T.flatten()[sorted_indices] > float("-inf")]
            # Store the valid indices in the steps array
            steps[idx:idx + len(valid_indices)] = valid_indices
            idx += len(valid_indices)
        return steps

    def _precompute_order_standard(self):
        """Get standard order of steps;
        - POL, POD are in ascending order
        - K is in ascending order but based on priority"""
        return th.arange(self.T*self.K, device=self.generator.device, dtype=th.int64)


    def _add_padding(self, *inputs) -> List[Tensor]:
        """For any number of inputs, add a zero padding at the end of vector"""
        output = []
        for input_tensor in inputs:
            padded_tensor = th.cat([input_tensor, self.padding])
            output.append(padded_tensor)
        return output

    def _precompute_for_step(self, pol:Tensor) -> Tuple[Tensor,Tensor,Tensor]:
        """Precompute variables and index masks for the current step"""
        # Index masks
        load_idx = self.load_transport[pol]
        disc_idx = self.discharge_transport[pol]
        moves_idx = self.moves_idx[pol]
        return load_idx, disc_idx, moves_idx

    def _precompute_revenues(self, reduce_long_revenue=0.3) -> Tensor:
        """Precompute matrix of revenues with shape [K, T]"""
        # Initialize revenues and pod_grid
        revenues = th.zeros((self.K, self.P, self.P), device=self.generator.device, dtype=self.float_type) # Shape: [K, P, P]
        pol_grid, pod_grid = th.meshgrid(self.ports, self.ports, indexing='ij')  # Shapes: [P, P]
        duration_grid = (pod_grid-pol_grid).to(revenues.dtype) # Shape: [P, P]
        # Compute revenues
        mask = th.arange(self.K, device=self.generator.device, dtype=self.float_type) < self.K // 2 # Spot/long-term mask
        revenues[~mask] = duration_grid # Spot market contracts
        revenues[mask] = (duration_grid * (1 - reduce_long_revenue)) # Long-term contracts
        i, j = th.triu_indices(self.P, self.P, offset=1) # Get above-diagonal indices of revenues
        # Add 0.1 for variable revenue per container, regardless of (k,tau)
        return revenues[..., i, j] + 0.1 # Shape: [K, T], where T = P*(P-1)/2

    def _precompute_transport_sets(self):
        """Precompute transport sets based on POL with shape(s): [P, T]"""
        # Note: transport sets in the environment do not depend on batches for efficiency.
        # Hence, implementation only works for batches with the same episodic step (e.g., single-step MDP)

        # Get transport sets for demand
        p = th.arange(self.P, device=self.generator.device, dtype=self.float_type).view(-1,1)
        self.load_transport = get_load_transport(self.transport_idx, p)
        self.previous_load_transport = get_load_transport(self.transport_idx, p-1)
        self.discharge_transport = get_discharge_transport(self.transport_idx, p)
        self.not_on_board_transport = get_not_on_board_transport(self.transport_idx, p)
        self.remain_on_board_transport = get_remain_on_board_transport(self.transport_idx, p)
        self.moves_idx = self.load_transport + self.discharge_transport

    def _precompute_transport_sets_episode(self):
        """Precompute transport sets based on POL with shape(s): [Seq, T]"""
        # Get transport sets for demand
        pol_t = self.pol[:-1].view(-1,1)
        self.load_transport_episode = get_load_transport(self.transport_idx, pol_t)
        self.previous_load_transport_episode = get_load_transport(self.transport_idx, pol_t-1)
        self.discharge_transport_episode = get_discharge_transport(self.transport_idx, pol_t)
        self.not_on_board_transport_episode = get_not_on_board_transport(self.transport_idx, pol_t)
        self.remain_on_board_transport_episode = get_remain_on_board_transport(self.transport_idx, pol_t)
        self.moves_idx_episode = self.load_transport_episode + self.discharge_transport_episode

    def _precompute_next_port_mask(self):
        """Precompute next port based on POL with shape: [P, T]
        - Next port happens when POD = POL+1
        """
        # Initialize next_port
        next_port = th.zeros((self.T*self.K), dtype=th.bool, device=self.generator.device)
        pol_values = th.arange(self.P - 1, 0, -1,)  # POL values from P-1 to 1
        indices = th.cumsum(self.K * pol_values, dim=0) - 1
        next_port[indices] = True
        return next_port

    def _precompute_stability_parameters(self,):
        """Precompute lhs stability parameters for compact constraints. Get rhs by negating lhs."""
        lp_weight = self.lp_weight.unsqueeze(2).expand(-1,-1,self.D,-1)
        vp_weight = self.vp_weight.unsqueeze(1).expand(-1,self.B,-1,-1)
        p_weight = th.cat([lp_weight, lp_weight, vp_weight, vp_weight], dim=0)
        target = torch.tensor([self.LCG_target, self.LCG_target, self.VCG_target, self.VCG_target],
                              device=self.generator.device, dtype=self.float_type).view(-1,1,1,1)
        delta = torch.tensor([self.stab_delta, -self.stab_delta, self.stab_delta, -self.stab_delta],
                             device=self.generator.device, dtype=self.float_type).view(-1,1,1,1)
        output = p_weight - self.weights.view(1,1,1,self.K) * (target + delta)
        return output.view(-1, self.B*self.D, self.K,)