from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class PendulumEnv(EnvBase):
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cuda"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def _step(tensordict):
        th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

        g_force = tensordict["params", "g"]
        mass = tensordict["params", "m"]
        length = tensordict["params", "l"]
        dt = tensordict["params", "dt"]
        u = tensordict["action"].squeeze(-1)
        u = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"])
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        new_thdot = (
                thdot
                + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length ** 2) * u) * dt
        )
        new_thdot = new_thdot.clamp(
            -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
        )
        new_th = th + new_thdot * dt
        reward = -costs.view(*tensordict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        out = TensorDict(
            {
                "th": new_th,
                "thdot": new_thdot,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        high_th = torch.tensor(DEFAULT_X, device=self.device)
        high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
        low_th = -high_th
        low_thdot = -high_thdot

        # for non batch-locked environments, the input ``tensordict`` shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        th = (
                torch.rand(tensordict.shape, generator=self.rng, device=self.device)
                * (high_th - low_th)
                + low_th
        )
        thdot = (
                torch.rand(tensordict.shape, generator=self.rng, device=self.device)
                * (high_thdot - low_thdot)
                + low_thdot
        )
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            th=Bounded(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
            ),
            thdot=Bounded(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = Bounded(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(1,),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(*td_params.shape, 1))

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            if self.device.type == "cuda":
                # Set the seed for CUDA
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            else:
                # Set the seed for CPU
                torch.manual_seed(seed)

            # Ensure deterministic behavior if needed
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Store the seed and device RNG state
            self.rng = torch.Generator(device=self.device).manual_seed(seed)

    @staticmethod
    def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_speed": 8,
                        "max_torque": 2.0,
                        "dt": 0.05,
                        "g": g,
                        "m": 1.0,
                        "l": 1.0,
                    },
                    [],
                )
            },
            [],
        device="cuda",)
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td


# Support functions
def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi