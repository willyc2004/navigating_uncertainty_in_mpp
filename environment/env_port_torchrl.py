import time
from typing import Optional, Iterable, List, Tuple, Dict

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
    """Port-wise Master Planning Problem environment."""
    name = "port_mpp"
    batch_locked = False

    def __init__(self, device="cuda", batch_size=[], td_gen=None, **kwargs):
        super().__init__(device=device, batch_size=batch_size, **kwargs)
        # Spec and td_gen
        if td_gen == None:
            self.td_gen = self.generator(batch_size=batch_size,)
        self._make_spec(self.td_gen)

        # Revenue
        self.static_revenue = self.revenues_matrix.T # Shape [T, K]

        ## Constraints # todo: clean-up here
        # Shapes
        self.n_demand = (self.K * (self.P - 1))
        self.n_locations = (self.B * self.D)
        self.n_stability = 4
        self.n_constraints = self.n_demand + self.n_locations + self.n_stability # (demand + capacity + stability)
        # Ranges
        self.demand_idx = th.arange(0, self.n_demand, device=device)
        self.location_idx = th.arange(self.n_demand, self.n_demand+self.n_locations, device=device)
        self.stability_idx = th.arange(self.n_demand+self.n_locations, self.n_constraints, device=device)
        # Add vectors
        ones = th.ones(self.n_constraints, device=self.generator.device, dtype=self.float_type)
        self.constraint_signs = ones.clone()
        indices_to_multiply = th.tensor([-3, -1], device=self.generator.device)
        self.constraint_signs[indices_to_multiply] *= -1
        self.swap_signs_stability = -ones.clone()
        self.swap_signs_stability[0] *= -1 # only demand constraint is positive

        # Get A
        self.lhs_A = self._create_port_constraint_matrix(shape=(self.n_constraints, self.B, self.D, self.P-1, self.K))
        self.rhs_A = self._create_port_util_constraint_matrix(shape=(self.n_constraints, self.B, self.D, self.T, self.K))

    def _make_spec(self, td:TensorDict = None) -> None:
        """Define the specs for observations, actions, rewards, and done flags."""
        batch_size = td.batch_size
        observation_spec = Unbounded(shape=(*batch_size,288), dtype=self.float_type)
        state_spec = Composite(
            utilization=Unbounded(shape=(*batch_size,self.B*self.D*self.T*self.K), dtype=self.float_type),
            target_long_crane=Unbounded(shape=(*batch_size,1), dtype=self.float_type),
            long_crane_moves=Unbounded(shape=(*batch_size,self.B-1), dtype=self.float_type),
            observed_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=th.float32),
            expected_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=th.float32),
            std_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=th.float32),
            realized_demand=Unbounded(shape=(*batch_size, self.T * self.K), dtype=th.float32),
            shape=batch_size,
        )
        self.observation_spec = Composite(
            # Obs and state
            observation=observation_spec,
            state=state_spec,
            # Action, timestep
            action=Unbounded(shape=(*batch_size, self.B * self.D * (self.P-1) * self.K), dtype=self.float_type),
            timestep=Unbounded(shape=(*batch_size,1), dtype=th.int64),
            # Demand
            realized_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=th.float32),
            observed_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=th.float32),
            expected_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=th.float32),
            std_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=th.float32),
            init_expected_demand=Unbounded(shape=(*batch_size,self.T*self.K),dtype=th.float32),
            batch_updates=Unbounded(shape=(*batch_size, 1), dtype=th.float32),

            # Constraints
            # lhs_A=Unbounded(shape=(*batch_size,self.n_constraints,self.B*self.D),dtype=self.float_type),
            # rhs=Unbounded(shape=(*batch_size,self.n_constraints),dtype=self.float_type),
            shape=batch_size,
        )
        self.action_spec = Bounded(
            shape=(*batch_size, self.B*self.D*(self.P-1)*self.K),  # Define shape as needed
            low=0.0,
            high=50.0,  # Define high value as needed
            dtype=self.float_type,
        )
        self.reward_spec = Unbounded(shape=(*batch_size,1,))
        self.done_spec = Unbounded(shape=(*batch_size,1,), dtype=th.bool)

    def _check_done(self, t: Tensor) -> Tensor:
        """Determine if the episode is done based on the state."""
        return (t == (self.P - 1)).view(-1,1)

    def _step(self, td: TensorDict) -> TensorDict:
        """Perform a step in the environment and return the next state."""
        # Extraction
        batch_size = td.batch_size
        action, t = self._extract_from_td(td, batch_size)
        load_idx, disc_idx, moves_idx = self._precompute_for_step(t[0]) # inherited!
        ac_transport = self.remain_on_board_transport[t[0]]  # inherited!
        utilization, demand_state = self._extract_from_state(td["state"], batch_size)

        # Check done, update utilization and long cranes
        # todo: add masking of action on irrelevant actions (when pod < pol)
        done = self._check_done(t)
        utilization = self._update_state_loading(action, utilization, load_idx, t[0])
        target_long_crane = compute_target_long_crane(demand_state["realized_demand"], moves_idx, self.capacity, self.B, self.CI_target)
        long_crane_moves = compute_long_crane(utilization, moves_idx, self.T)

        ## Reward
        # Precompute
        sum_action = action.sum(dim=(-4, -3))[...,t:,:]
        load_demand = demand_state["observed_demand"][..., load_idx, :]
        load_revenues = self.static_revenue[load_idx, :]
        # Revenue
        revenue_matrix = (th.clamp(sum_action, min=th.zeros_like(sum_action),max=load_demand) * load_revenues)
        revenue = revenue_matrix.sum(dim=(-2,-1))
        # Costs
        overstowage = compute_hatch_overstowage(utilization, moves_idx, ac_transport)
        excess_crane_moves = th.clamp(long_crane_moves - target_long_crane.view(-1, 1), min=0)
        ho_costs = overstowage.sum(dim=-1, keepdim=True) * self.ho_costs
        lc_costs = excess_crane_moves.sum(dim=-1, keepdim=True) * self.lc_costs
        cost = ho_costs + lc_costs
        # Reward/profit
        profit = revenue - cost
        reward = profit.clone()
        # todo: normalization of reward
        # normalize_revenue = load_revenues * load_demand
        # reward = (revenue_matrix.clone() / normalize_revenue).mean(dim=(-2,-1)) - (cost / normalize_revenue.sum()).clamp(max=10.0)

        ## Transition to next step
        # Update next state
        t = t + 1
        next_state_dict = self._update_next_state(utilization, demand_state, t, batch_size)
        if not done.any():
            rhs = self.create_port_rhs(next_state_dict["utilization"], next_state_dict["rhs_demand"], batch_size)

        # Only for final port
        if done:
            # Compute last port long crane excess
            _, _, moves_idx = self._precompute_for_step(t[0])  # inherited!
            lc_moves_last_port = compute_long_crane(utilization, moves_idx, self.T)
            lc_excess_last_port = th.clamp(lc_moves_last_port - next_state_dict["target_long_crane"].view(-1,1), min=0)

            # Compute metrics
            excess_crane_moves += lc_excess_last_port
            lc_cost_ = lc_excess_last_port.sum(dim=-1, keepdim=True) * self.lc_costs
            profit -= lc_cost_
            cost += lc_cost_
            lc_costs += lc_cost_

        # Update td output
        obs = self._get_observation(next_state_dict, t, batch_size)
        out =  TensorDict({
            "state":{
                # Vessel
                "utilization": next_state_dict["utilization"].view(*batch_size, self.B*self.D*self.T*self.K),
                "target_long_crane": next_state_dict["target_long_crane"].view(*batch_size, 1),
                "long_crane_moves": next_state_dict["long_crane_moves"].view(*batch_size, self.B - 1),
                # Demand
                "observed_demand": next_state_dict["observed_demand"].view(*batch_size, self.T * self.K),
                "expected_demand": next_state_dict["expected_demand"].view(*batch_size, self.T * self.K),
                "std_demand": next_state_dict["std_demand"].view(*batch_size, self.T * self.K),
                "realized_demand": next_state_dict["realized_demand"].view(*batch_size, self.T * self.K),
            },
            "observation":obs,
            # todo: fix redundancy demand?
            "realized_demand": td["realized_demand"].view(*batch_size, self.T * self.K),
            "observed_demand": td["observed_demand"].view(*batch_size, self.T*self.K),
            "expected_demand": td["expected_demand"].view(*batch_size, self.T*self.K),
            "std_demand": td["std_demand"].view(*batch_size, self.T*self.K),
            "init_expected_demand": td["init_expected_demand"].view(*batch_size, self.T * self.K),
            "batch_updates": td["batch_updates"] + 1,

            # Action, reward, done and step
            "action": action.view(*batch_size, -1),
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

        # Initialize
        device = td.device
        t = th.zeros((1,) if batch_size == th.Size([]) else (*batch_size,), dtype=th.int64, device=device)
        load_idx, _, moves_idx = self._precompute_for_step(t[0]) # inherited!
        action_mask = th.ones(self.action_spec.shape, dtype=th.bool, device=device)
        utilization = th.zeros((*batch_size, self.B, self.D, self.T, self.K), device=device, dtype=self.float_type)
        target_long_crane = compute_target_long_crane(td["realized_demand"].view(*batch_size, self.T, self.K),
                                                            moves_idx, self.capacity, self.B, self.CI_target)
        long_crane_moves = compute_long_crane(utilization, moves_idx, self.T)
        rhs_demand = th.zeros((*batch_size, self.P-1, self.K), device=device, dtype=self.float_type)
        rhs_demand[...,t:,:] = td["observed_demand"].view(*batch_size, self.T, self.K)[...,load_idx,:]
        rhs = self.create_port_rhs(utilization, rhs_demand, batch_size)
        initial_state = TensorDict({
            "utilization": utilization.view(*batch_size, self.B*self.D*self.T*self.K),
            "target_long_crane": target_long_crane.view(*batch_size, 1),
            "long_crane_moves": long_crane_moves.view(*batch_size, self.B-1),
            "observed_demand": td["observed_demand"].view(*batch_size, self.T * self.K).clone(),
            "expected_demand": td["expected_demand"].view(*batch_size, self.T * self.K).clone(),
            "std_demand": td["std_demand"].view(*batch_size, self.T * self.K).clone(),
            "realized_demand": td["realized_demand"].view(*batch_size, self.T * self.K).clone(),
        }, batch_size=batch_size, device=device,)
        obs = self._get_observation(initial_state, t, batch_size)

        # Initialize td
        out = TensorDict({
            # State + obs
            "state": initial_state,
            "observation": obs,
            # Action, mask, rhs
            "action": th.zeros_like(action_mask, dtype=self.float_type),
            # "action_mask": action_mask.view(*batch_size, -1),
            "rhs":rhs.view(*batch_size,-1),
            # Reward, done and step
            "done": th.zeros_like(t, dtype=th.bool),
            "timestep": t,
            # Demand generator
            "realized_demand": td["realized_demand"].view(*batch_size, self.T * self.K),
            "observed_demand": td["observed_demand"].view(*batch_size, self.T * self.K),
            "expected_demand": td["expected_demand"].view(*batch_size, self.T * self.K),
            "std_demand": td["std_demand"].view(*batch_size, self.T * self.K),
            "init_expected_demand": td["init_expected_demand"].view(*batch_size, self.T * self.K),
            "batch_updates": td["batch_updates"] + 1,
        }, batch_size=batch_size, device=device,)
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = th.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

    # Unpacking
    def _extract_from_td(self, td, batch_size) -> Tuple:
        action = td["action"].view(*batch_size, self.B, self.D, self.P-1, self.K,)
        t = td["timestep"].view(-1)
        return action, t

    def _extract_from_state(self, state, batch_size) -> Tuple:
        # Vessel-related variables
        utilization = state["utilization"].view(*batch_size, self.B, self.D, self.T, self.K).clone()
        # Demand-related variables
        demand = {
            # clones are needed to prevent in-place
            "observed_demand": state["observed_demand"].view(*batch_size, self.T, self.K).clone(),
            "expected_demand": state["expected_demand"].view(*batch_size, self.T, self.K).clone(),
            "std_demand": state["std_demand"].view(*batch_size, self.T, self.K).clone(),
            "realized_demand": state["realized_demand"].view(*batch_size, self.T, self.K).clone(),
        }
        return utilization, demand

    def _get_observation(self, next_state_dict, t, batch_size, eps=1e-5, **kwargs) -> Tensor:
        ## Demand
        # todo: evaluate demand normalization
        print("-------")
        print("t", t[0])
        demand_norm = max(next_state_dict["realized_demand"].max().item(), next_state_dict["expected_demand"].max().item())
        observed_demand = next_state_dict["observed_demand"] / demand_norm
        expected_demand = next_state_dict["expected_demand"] / demand_norm
        std_demand = next_state_dict["std_demand"] / demand_norm
        # print("obs", observed_demand)
        # print("e[x]", expected_demand)
        # print("std[x]", std_demand)

        ## Vessel
        # Capacity
        utilization = next_state_dict["utilization"].view(*batch_size, self.B, self.D, self.T, self.K)
        residual_capacity = th.clamp(self.capacity - utilization.sum(dim=-2) @ self.teus, min=self.zero) / self.capacity
        residual_lc_capacity = (next_state_dict["target_long_crane"] - next_state_dict["long_crane_moves"]).clamp(min=0) \
                               / next_state_dict["target_long_crane"]
        print("residual_capacity", residual_capacity.T)
        print("residual_lc_capacity", residual_lc_capacity)

        # Stability
        location_weight = (utilization * self.weights.view(1,1,1,1,-1)).sum(dim=(-2,-1))
        total_weight = location_weight.sum(dim=(1,2))
        lcg = (location_weight * self.longitudinal_position.view(1, -1, 1)).sum(dim=(1,2)) / (total_weight+eps)
        vcg = (location_weight * self.vertical_position.view(1, 1, -1)).sum(dim=(1,2)) / (total_weight+eps)

        # Origin-destination pairs
        pol_locations, pod_locations = compute_pol_pod_locations(utilization, self.transform_tau_to_pol, self.transform_tau_to_pod)
        # todo: pol_locations is 1.0 initially, which is annoying - try to rewrite.
        # 0 has value of port 0, while pol 1.0 does not exist. However, this does not make sense for model.
        # 0 needs to mean 0, 1/P, 2/P etc.
        agg_pol_location, agg_pod_location = aggregate_pol_pod_location(pol_locations, pod_locations,
                                                                        self.ports, self.P, self.float_type)

        return th.cat([
            t.view(*batch_size, 1) / (self.T * self.K),
            observed_demand.view(*batch_size, self.T * self.K),
            expected_demand.view(*batch_size, self.T * self.K),
            std_demand.view(*batch_size, self.T * self.K),
            lcg.view(*batch_size, 1),
            vcg.view(*batch_size, 1),
            residual_capacity.view(*batch_size, self.B * self.D),
            residual_lc_capacity.view(*batch_size, self.B - 1),
            agg_pol_location.view(*batch_size, self.B * self.D) / self.P,
            agg_pod_location.view(*batch_size, self.B * self.D) / self.P,
        ], dim=-1)

    # Update state
    def _update_state_loading(self, action: Tensor, utilization: Tensor, load_idx: Tensor, t) -> Tensor:
        """Transition to load action to utilization."""
        new_utilization = utilization.clone()
        new_utilization[..., load_idx, :] = action[...,t:,:]
        return new_utilization

    def _update_next_state(self, utilization:Tensor, demand_state:Dict, t:Tensor, batch_size:th.Size) -> Dict[str, Tensor]:
        """Update next state, following options:
        """
        # Check next port with t - 1
        load_idx, disc_idx, moves_idx = self._precompute_for_step(t[0])
        # Next port with discharging; Update utilization, observed demand and target long crane
        long_crane_moves = compute_long_crane(utilization, moves_idx, self.T)
        utilization = update_state_discharge(utilization, disc_idx)
        target_long_crane = compute_target_long_crane(demand_state["realized_demand"], moves_idx,
                                                      self.capacity, self.B, self.CI_target).view(*batch_size, 1)
        if self.demand_uncertainty:
            demand_state["observed_demand"][..., load_idx, :] = demand_state["realized_demand"][..., load_idx, :]

        # Update residual lc capacity: target - actual load and discharge moves
        residual_lc_capacity = (target_long_crane - long_crane_moves).clamp(min=0)

        # rhs demand # todo: initialize rhs_demand constantly not efficient
        rhs_demand = th.zeros((*batch_size, self.P - 1, self.K), device=self.device, dtype=self.float_type)
        rhs_demand[..., t:, :] = demand_state["observed_demand"].view(*batch_size, self.T, self.K)[..., load_idx, :]

        # Get output
        return {
            "observed_demand": demand_state["observed_demand"],
            "expected_demand": demand_state["expected_demand"],
            "std_demand":demand_state["std_demand"],
            "realized_demand": demand_state["realized_demand"],
            "rhs_demand":rhs_demand,
            "utilization": utilization,
            "location_utilization": (utilization * self.teus.view(1,1,1,1,-1)).sum(dim=(-2,-1)),
            "target_long_crane": target_long_crane,
            "long_crane_moves":long_crane_moves,
            "residual_lc_capacity": residual_lc_capacity,
        }

    # Constraints
    def _create_port_constraint_matrix(self, shape: Tuple):
        """Create lhs matrix A in Ax <= b_t.
            Shape [M, N]:
            - M: [demand (P-1)*K, capacity B*D, LM-TW, TW-LM, VM-TW, TW-VM]
            - N: action B*D*(P-1)*K
        """
        A = th.ones(shape, device=self.generator.device, dtype=self.float_type)
        A[self.demand_idx] = th.eye(self.n_demand, device=self.generator.device, dtype=self.float_type).view(self.n_demand, 1, 1, self.P-1, self.K)
        A[self.location_idx] = self.teus.view(1, 1, 1, -1) * th.eye(self.n_locations, device=self.generator.device, dtype=self.float_type).view(self.n_locations, self.B, self.D, 1, 1)
        A[self.stability_idx] *= self.stability_params_lhs.view(self.n_stability, self.B, self.D, 1, self.K,)
        return A.view(self.n_constraints, -1) * self.constraint_signs.unsqueeze(-1)

    def _create_port_util_constraint_matrix(self, shape: Tuple):
        """Create constraint matrix A' as part of rhs vector: b_t = b - A'u_t.
        Shape [M, N]:
        - M: [demand (P-1)*K, capacity B*D, LM-TW, TW-LM, VM-TW, TW-VM]
        - N: utilization B*D*T*K

        Note this is only relevant for capacity and stability constraints, as it depends on the current utilization.
        """
        A = th.ones(shape, device=self.generator.device, dtype=self.float_type)
        A[self.demand_idx] = 0 # set to 0, as irrelevant for demand
        A[self.location_idx] = self.teus.view(1, 1, 1, -1) * th.eye(self.n_locations, device=self.generator.device, dtype=self.float_type).view(self.n_locations, self.B, self.D, 1, 1)
        A[self.stability_idx] *= self.stability_params_lhs.view(self.n_stability, self.B, self.D, 1, self.K,)
        return A.view(self.n_constraints, -1) * self.constraint_signs.unsqueeze(-1)

    def create_port_rhs(self, utilization, observed_demand, batch_size):
        """Create rhs vector b_t in Ax <= b_t.
            Shape [M]:
            - M: [demand (P-1)*K, capacity B*D, LM-TW, TW-LM, VM-TW, TW-VM]

        Note: b_t = b - A'u_t, where b is some vector, A'u_t is matmul of rhs constraint matrix and utilization u_t.
        """
        # Compute -A'u_t
        A = self.swap_signs_stability.view(-1, 1) * self.rhs_A.clone() # Obtain -A'
        b_t = utilization.view(*batch_size, -1) @ A.view(self.n_constraints, -1).T # Matmul -A'u_t
        # Compute b_t = b - A'u_t
        b_t[..., self.demand_idx] = observed_demand.view(*batch_size, -1)
        capacity = self.capacity.view(*batch_size, -1)
        b_t[..., self.location_idx] = th.clamp(b_t[..., self.location_idx] + capacity,
                                               min=th.zeros_like(capacity, device=self.device), max=capacity)
        return b_t