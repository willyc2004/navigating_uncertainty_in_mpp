import os
import time
import yaml
import numpy as np
import torch
import tqdm

from collections import defaultdict
from typing import Optional
from dotmap import DotMap
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

# TorchRL
from torchrl.envs import (
    ParallelEnv,
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms import Transform, ObservationNorm, StepCounter
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.data import Bounded, Composite, Unbounded

# Custom Environment
from environment.env_example import PendulumEnv
from environment.env_ import MasterPlanningEnv

def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config


def make_env(env_kwargs:DotMap, device: torch.device = torch.device("cuda")):
    """Setup and transform the Pendulum environment."""
    env = MasterPlanningEnv(**env_kwargs)  # Custom environment
    # # Apply necessary transforms
    # env = PendulumEnv()
    # env = TransformedEnv(
    #     env,
    #     # ``Unsqueeze`` the observations that we will concatenate
    #     UnsqueezeTransform(
    #         dim=-1,
    #         in_keys=["th", "thdot"],
    #         in_keys_inv=["th", "thdot"],
    #     ),
    # )
    #
    # t_sin = SinTransform(in_keys=["th"], out_keys=["sin"])
    # t_cos = CosTransform(in_keys=["th"], out_keys=["cos"])
    # env.append_transform(t_sin)
    # env.append_transform(t_cos)
    #
    # cat_transform = CatTensors(
    #     in_keys=["sin", "cos", "thdot"], dim=-1, out_key="observation", del_keys=False
    # )
    # env.append_transform(cat_transform)

    # Move the environment to the specified device
    transformed_env = env.to(device)
    return transformed_env

class SinTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class CosTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.cos()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )

def main(config: Optional[DotMap] = None):
    # Initialize torch and cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 64  # or some higher value

    # Set random seed and device
    torch.set_num_threads(1)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
        torch.backends.cudnn.deterministic = True

    ## Environment initialization
    emb_dim = 128
    env_kwargs = config.env

    # todo: add our custom environment
    env = make_env(env_kwargs)
    # todo: fix parallel env
    # env = ParallelEnv(4, make_env)
    check_env_specs(env)  # this must pass for ParallelEnv to work

    # Simple training
    torch.manual_seed(0)
    env.set_seed(0)

    # todo: add more complex model
    net = nn.Sequential(
        nn.LazyLinear(1000),
        nn.ReLU(),
        nn.LazyLinear(1000),
        nn.ReLU(),
        nn.LazyLinear(1000),
        nn.ReLU(),
        nn.LazyLinear(env.action_spec.shape[0]),
    ).to(device).to(env.float_type)

    policy = TensorDictModule(
        net,
        in_keys=["observation", ],
        out_keys=["action"],
    )

    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)

    batch_size = 32
    pbar = tqdm.tqdm(range(20_000 // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
    logs = defaultdict(list)

    for _ in pbar:
        init_td = env.reset(env.generator(batch_size=batch_size))
        rollout = env.rollout(72, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()

        # # Debugging checks
        # if torch.isnan(init_td["observation"]).any():
        #     raise ValueError("NaN detected in initial observations!")
        # # Check rewards after rollout
        # if torch.isnan(rollout["next", "reward"]).any():
        #     raise ValueError("NaN detected in rewards after rollout!")
        # # Check model output (actions) after policy call
        # actions = policy(init_td)["action"]
        # if torch.isnan(actions).any():
        #     raise ValueError("NaN detected in policy actions!")
        #
        # print(f"Policy actions: {actions.mean(), actions.shape}")
        # if torch.isnan(traj_return):
        #     raise ValueError("NaN detected in trajectory return!")
        #
        # print(f"Trajectory return: {traj_return}")

        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
            f"last total_revenue: {rollout[..., -1]['next', 'state', 'total_revenue'].mean(): 4.4f}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

    def plot():
        import matplotlib
        from matplotlib import pyplot as plt

        with plt.ion():
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(logs["return"])
            plt.title("returns")
            plt.xlabel("iteration")
            plt.subplot(1, 2, 2)
            plt.plot(logs["last_reward"])
            plt.title("last reward")
            plt.xlabel("iteration")
            plt.show()

    plot()


if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()
    with open(f'{file_path}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    main(config)