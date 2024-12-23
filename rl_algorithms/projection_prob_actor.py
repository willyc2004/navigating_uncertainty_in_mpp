# Datatypes
from typing import Optional, Tuple, Dict, Union, Sequence
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# TorchRL
from torchrl.modules import ProbabilisticActor
from torchrl.data.tensor_specs import Composite, TensorSpec

class ProjectionProbabilisticActor(ProbabilisticActor):
    def __init__(self,
                 module: TensorDictModule,
                 in_keys: Union[NestedKey, Sequence[NestedKey]],
                 out_keys: Optional[Sequence[NestedKey]] = None,
                 *,
                 spec: Optional[TensorSpec] = None,
                 projection_layer: Optional[nn.Module] = None,
                 action_rescale_min: Optional[float] = None,
                 action_rescale_max: Optional[float] = None,
                 **kwargs):
        super().__init__(module, in_keys, out_keys, spec=spec, **kwargs)

        # Rescale actions from [-upscale, upscale] to (min, max)
        self.upscale = kwargs.get("distribution_kwargs", {}).get("upscale", 1.0)
        if action_rescale_min is not None and action_rescale_max is not None:
            self.rescale_action = lambda x: x * (action_rescale_max - action_rescale_min) + action_rescale_min
        # Initialize projection layer
        self.projection_layer = projection_layer

    @staticmethod
    def conditional_softmax(sample, upper_bound):
        if sample.dim() - upper_bound.dim() == 1:
            upper_bound = upper_bound.unsqueeze(-1)
        elif sample.dim() - upper_bound.dim() < 0 or sample.dim() - upper_bound.dim() > 1:
            raise ValueError(f"Sample dim {sample.dim()} and upper_bound dim {upper_bound.dim()} not compatible.")

        # If the sum exceeds the upper_bound, apply softmax scaling
        condition = sample.sum(dim=-1, keepdim=True) > upper_bound
        scaled_sample = torch.where(
            condition,
            F.softmax(sample, dim=-1) * upper_bound,
            sample
        )
        return scaled_sample

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        ub = out["realized_demand"][...,out["state", "timestep"][0]] if out["realized_demand"].dim() == 2 \
            else out["realized_demand"][..., out["state", "timestep"][0,0],:]
        out["action"] = self.conditional_softmax(out["action"], upper_bound=ub).clone()
        if not self.training:
            out["action"] = out["loc"]
        #
        #     if self.rescale_action is not None:
        #         out["action"] = self.rescale_action((out["loc"] + self.upscale) / 2)
        # else:
        #     if self.rescale_action is not None:
        #         out["action"] = self.rescale_action((out["action"] + self.upscale) / 2)  # Rescale from [-x, x] to (min, max)
        if self.projection_layer is not None:
            action = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
            out["action"] = action
        return out
