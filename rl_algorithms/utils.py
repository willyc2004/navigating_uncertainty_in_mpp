from typing import Optional, Dict
from dotmap import DotMap
from tensordict import TensorDict
import torch
from environment.env_torchrl import MasterPlanningEnv
from environment.utils import compute_violation

def make_env(env_kwargs:DotMap, batch_size:Optional[list] = [], device: torch.device = torch.device("cuda")):
    """Setup and transform the Pendulum environment."""
    return MasterPlanningEnv(batch_size=batch_size, **env_kwargs).to(device)

def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def compute_loss_feasibility(td, action, feasibility_coef, aggregate_feasibility="sum"):
    # todo: redudant - also in loss_module of PPO_feas
    """Compute feasibility loss based on the action, lhs_A, and rhs tensors."""
    lhs_A = td.get("lhs_A")
    rhs = td.get("rhs")
    violation = compute_violation(action, lhs_A, rhs)

    # Get aggregation dimensions
    if aggregate_feasibility == "sum":
        sum_dims = [-x for x in range(1, violation.dim())]
        return feasibility_coef * violation.sum(dim=sum_dims).mean(), violation
    elif aggregate_feasibility == "mean":
        return feasibility_coef * violation.mean(), violation

def recursive_check_for_nans(td, parent_key=""):
    """Recursive check for NaNs and Infs in e.g. TensorDicts and raise an error if found."""
    for key, value in td.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, torch.Tensor):
            check_for_nans(value, full_key)
        elif isinstance(value, TensorDict) or  isinstance(value, Dict):
            # Recursively check nested TensorDicts
            recursive_check_for_nans(value, full_key)

def check_for_nans(tensor, name):
    """Check for NaNs and Infs in a tensor and raise an error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")