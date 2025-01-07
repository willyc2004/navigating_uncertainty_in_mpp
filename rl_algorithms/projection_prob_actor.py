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

# Custom
from environment.utils import compute_violation

class ProjectionProbabilisticActor(ProbabilisticActor):
    def __init__(self,
                 module: TensorDictModule,
                 in_keys: Union[NestedKey, Sequence[NestedKey]],
                 out_keys: Optional[Sequence[NestedKey]] = None,
                 *,
                 spec: Optional[TensorSpec] = None,
                 projection_layer: Optional[nn.Module] = None,
                 projection_type: Optional[str] = None,
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
        self.projection_type = projection_type

    @staticmethod
    def conditional_softmax(sample, upper_bound):
        if sample.dim() - upper_bound.dim() == 1:
            upper_bound = upper_bound.unsqueeze(-1)
        if sample.dim() - upper_bound.dim() == 2:
            upper_bound = upper_bound.view(-1, 1, 1)
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

    @staticmethod
    def conditional_direct_scaling(sample, ub, epsilon=1e-8):
        sum_sample = sample.sum(dim=-1, keepdim=True)
        upper_bound = ub.unsqueeze(-1)
        scaling_factor = upper_bound / (sum_sample + epsilon)  # Avoid division by zero
        scaled_sample = torch.where(
            sum_sample > upper_bound,
            sample * scaling_factor,
            sample
        )
        return scaled_sample

    def jacobian_adaptation(self, logprob, jacobian) -> Tensor:
        """Perform logprob adaptation for invertible and differentiable projection functions with non-singular Jacobians."""
        # log pi'(x|s) = log pi(x|s) - log|det(J_g(x))|
        if jacobian is None:
            return logprob
        sign, log_abs_det = torch.linalg.slogdet(jacobian)
        # print("logprob", logprob.mean(),"log_abs_det", log_abs_det.mean())
        return logprob - log_abs_det

    def jacobian_direct_scaling(self, x, y, epsilon=1e-8):
        """Compute the Jacobian of the direct scaling projection:
            J_g(x) = y / sum(x)^2 * (diag(sum(x)) - x * 1^T)"""
        batch, n = x.shape
        sum_x = x.sum(dim=-1, keepdim=True)
        scaling_factor = y.unsqueeze(-1) / (sum_x)**2 # Shape: (batch_size, n)
        kronecker_delta = torch.eye(n).unsqueeze(0).expand(batch, n, n).to(x.device)  # Shape: (batch_size, n, n)
        x_product = x.unsqueeze(-1) * torch.ones(x.size(-1), device=x.device)  # Shape: (batch_size, n, n)
        jacobian = scaling_factor.unsqueeze(-1) * (kronecker_delta * sum_x.unsqueeze(-1)) - x_product # Shape: (batch_size, n, n)
        return jacobian

    def jacobian_violation(self, x, A, b, alpha,):
        """Compute the Jacobian of the violation projection:
            J_g(x) = I + alpha * A^T D A, where D = diag((Ax-b) > 0)"""
        print("shape", x.shape, A.shape, b.shape)
        if A.dim() == 3:
            batch, m, n = A.shape
        else:
            batch, n = A.shape
        # todo: SAC issue with shapes
        I = torch.eye(n, device=x.device).expand(batch, n, n)  # Identity matrix for each batch
        violation = compute_violation(x, A, b)
        D = torch.diag_embed(violation > 0).float()
        jacobian = I + alpha * torch.bmm(A.transpose(1, 2), D).bmm(A)
        return jacobian

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        # if self.rescale_action is not None:
        #     out["action"] = self.rescale_action((out["loc"] + self.upscale) / 2) # Rescale from [-x, x] to (min, max)

        # Apply projection and logprob adaptation with Jacobian
        if self.projection_type == "direct_scaling":
            ub = out["state", "realized_demand"][..., out["state", "timestep"][0]] if out["state", "realized_demand"].dim() == 2 \
                else out["state", "realized_demand"][..., out["state", "timestep"][0, 0], :]
            out["action"] = self.conditional_direct_scaling(out["action"], ub=ub).clone()
            jacobian = self.jacobian_direct_scaling(out["action"], ub)
        # elif self.projection_type == "softmax":
        #     out["action"] = self.conditional_softmax(out["action"], upper_bound=ub).clone()
        elif self.projection_type == "linear_violation":
                out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
                jacobian = self.jacobian_violation(out["action"], out["lhs_A"],  out["rhs"], self.projection_layer.alpha)
        elif self.projection_type in ["convex_program", "worst_case_violation", "linear_program"]:
            raise ValueError(f"Jacobian adaptation of log probs for projection type \'{self.projection_type}\' not supported.")
        else:
            jacobian = None

        out["sample_log_prob"] = self.jacobian_adaptation(out["sample_log_prob"], jacobian=jacobian).clone()
        return out
