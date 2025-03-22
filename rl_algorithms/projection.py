import torch
import torch.nn as nn
import qpth
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import warnings

class EmptyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x, **kwargs):
        return x

class LinearViolationAdaption(nn.Module):
    """Convex violation layer to enforce soft feasibility by projecting solutions towards feasible region."""

    def __init__(self, **kwargs):
        super(LinearViolationAdaption, self).__init__()
        self.alpha = kwargs.get('alpha', 0.005)
        self.scale = kwargs.get('scale', 0.001)
        self.delta = kwargs.get('delta', 0.1)
        self.max_iter = kwargs.get('max_iter', 100)
        self.use_early_stopping = kwargs.get('use_early_stopping', True)

    def forward(self, x, A, b, **kwargs):
        # Raise error is dimensions are invalid
        if b.dim() not in [2, 3] or A.dim() not in [3, 4]:
            raise ValueError("Invalid dimensions: 'b' must have dim 2 or 3 and 'A' must have dim 3 or 4.")

        # Shapes
        batch_size = b.shape[0]
        m = b.shape[-1]
        n_step = 1 if b.dim() == 2 else b.shape[-2] if b.dim() == 3 else None

        # Tensors shapes
        x_ = x.clone()
        b = b.unsqueeze(1) if b.dim() == 2 else b
        A = A.unsqueeze(1) if A.dim() == 3 else A
        x_ = x_.unsqueeze(1) if x_.dim() == 2 else x_
        # Initialize tensors
        active_mask = torch.ones(batch_size, n_step, dtype=torch.bool, device=x.device)  # Start with all batches active

        # Start loop with early exit in case of nans
        if torch.isnan(x_).any():
            return x_.squeeze(1)
        count = 0
        while torch.any(active_mask):
            # Compute current violation for each batch and step
            violation_new = torch.clamp(torch.matmul(x_.unsqueeze(2), A.transpose(-2, -1)).squeeze(2) - b, min=0)
            # Shape: [batch_size, n_step, m]
            total_violation = torch.sum(violation_new, dim=-1)  # Sum violations in [batch_size, n_step]

            # Define batch-wise stopping conditions
            no_violation = total_violation < self.delta

            # Update active mask: only keep batches and steps that are not within tolerance
            active_mask = ~(no_violation)

            # Break if no batches/steps are left active
            if self.use_early_stopping and not torch.any(active_mask):
                break

            # Calculate penalty gradient for adjustment
            penalty_gradient = torch.matmul(violation_new.unsqueeze(2), A).squeeze(2)  # Shape: [32, 1, 20]

            # Apply penalty gradient update only for active batches/steps
            # scale = 1 / (torch.std(penalty_gradient, dim=0, keepdim=True) + 1e-6)
            lr = self.alpha / (1 + self.scale * penalty_gradient)
            x_ = torch.where(active_mask.unsqueeze(2), x_ - lr * penalty_gradient, x_)
            x_ = torch.clamp(x_, min=0) # Ensure non-negativity

            count += 1
            if count > self.max_iter:
                break
        # Return the adjusted x_, reshaped to remove n_step dimension if it was initially 2D
        return x_.squeeze(1) if n_step == 1 else x_


import torch
import torch.nn as nn
import qpth
import warnings

class QPProjectionWithSlack(nn.Module):
    def __init__(self, slack_penalty=1.0, box_bounds=None, **kwargs):
        super().__init__()
        self.slack_penalty = slack_penalty
        self.box_bounds = box_bounds  # (lower, upper) or None

    def forward(self, x_raw, A, b):
        """
        Supports input of shape [batch, n] or [batch, seq, n]
        A: [batch, seq, m, n] or [batch, m, n]
        b: [batch, seq, m] or [batch, m]
        """

        orig_shape = x_raw.shape
        is_seq = x_raw.dim() == 3  # [batch, seq, n]

        if is_seq:
            batch_size, seq_len, n = x_raw.shape
            x_raw = x_raw.reshape(-1, n)            # [batch*seq, n]
            A = A.reshape(-1, *A.shape[-2:])        # [batch*seq, m, n]
            b = b.reshape(-1, b.shape[-1])          # [batch*seq, m]
        else:
            batch_size, n = x_raw.shape

        m = b.shape[-1]
        device = x_raw.device
        total = x_raw.shape[0]  # batch * seq OR batch

        # Build Q: block diagonal [I, 0; 0, λI]
        Q = torch.cat([
            torch.cat([torch.eye(n), torch.zeros(n, m)], dim=1),
            torch.cat([torch.zeros(m, n), self.slack_penalty * torch.eye(m)], dim=1)
        ], dim=0).to(device)
        Q = Q.unsqueeze(0).repeat(total, 1, 1)  # [total, n+m, n+m]

        # Linear term: [-x_raw, 0]
        p = torch.cat([-x_raw, torch.zeros(total, m, device=device)], dim=1)  # [total, n+m]

        # Constraints
        G_list = []
        h_list = []

        # A x - s ≤ b  → [A, -I]
        G1 = torch.cat([A, -torch.eye(m).unsqueeze(0).repeat(total, 1, 1).to(device)], dim=2)
        G_list.append(G1)
        h_list.append(b)

        # x ≤ upper  and  -x ≤ -lower
        if self.box_bounds is not None:
            lower, upper = self.box_bounds
            if lower.dim() == 1:
                lower = lower.unsqueeze(0).expand(total, -1)
            if upper.dim() == 1:
                # todo: what is upper here?
                upper = upper.unsqueeze(0).expand(total, -1)

            I_n = torch.eye(n).unsqueeze(0).repeat(total, 1, 1).to(device)
            zero_ns = torch.zeros(total, n, m, device=device)

            G_upper = torch.cat([I_n, zero_ns], dim=2)   # [I, 0]
            h_upper = upper

            G_lower = torch.cat([-I_n, zero_ns], dim=2)  # [-I, 0]
            h_lower = -lower

            G_list += [G_upper, G_lower]
            h_list += [h_upper, h_lower]

        # s ≥ 0  → [0, -I]
        zero_mn = torch.zeros(total, m, n, device=device)
        minus_I_m = -torch.eye(m).unsqueeze(0).repeat(total, 1, 1).to(device)
        G_slack = torch.cat([zero_mn, minus_I_m], dim=2)
        h_slack = torch.zeros(total, m, device=device)

        G_list.append(G_slack)
        h_list.append(h_slack)

        G = torch.cat(G_list, dim=1)  # [total, ?, n+m]
        h = torch.cat(h_list, dim=1)  # [total, ?]

        dummy_A = torch.empty(total, 0, n + m, device=device)
        dummy_b = torch.empty(total, 0, device=device)

        # Solve QP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_s_slack = qpth.qp.QPFunction(eps=1e-2, verbose=False, maxIter=10)(
                Q, p, G, h, dummy_A, dummy_b
            )  # [total, n + m]

        x_proj = x_s_slack[:, :n]  # [total, n]

        if is_seq:
            return x_proj.view(batch_size, seq_len, n)
        else:
            return x_proj  # [batch, n]


class CvxpyProjectionLayer(nn.Module):
    def __init__(self, n_action=80, n_constraints=85, slack_penalty=1, **kwargs):
        """
        n: number of decision variables
        m: number of linear inequality constraints
        slack_penalty: how much to penalize constraint violation (higher = stricter)
        """
        super().__init__()
        self.n = n_action
        self.m = n_constraints
        self.slack_penalty = slack_penalty

        # Define CVXPY variables and parameters
        x = cp.Variable(n_action)
        s = cp.Variable(n_constraints)

        x_raw_param = cp.Parameter(n_action)
        A_param = cp.Parameter((n_constraints, n_action))
        b_param = cp.Parameter(n_constraints)
        lower_param = cp.Parameter(n_action)
        upper_param = cp.Parameter(n_action)

        # Objective: projection + slack penalty
        objective = cp.Minimize(
            0.5 * cp.sum_squares(x - x_raw_param) +
            slack_penalty * cp.sum_squares(s)
        )

        constraints = [
            A_param @ x <= b_param + s,
            s >= 0,
            x >= lower_param,
            x <= upper_param
        ]

        problem = cp.Problem(objective, constraints)

        # Wrap in differentiable layer
        self.cvxpy_layer = CvxpyLayer(
            problem,
            parameters=[x_raw_param, A_param, b_param, lower_param, upper_param],
            variables=[x]
        )

    def forward(self, x_raw, A, b, lower=None, upper=None):
        """
        x_raw: [batch, n]
        A: [batch, m, n]
        b: [batch, m]
        lower, upper: [n] or [batch, n] (optional)
        Returns: projected x: [batch, n]
        """
        batch_size = x_raw.shape[0]
        device = x_raw.device

        # Default bounds
        if lower is None:
            lower = torch.zeros_like(x_raw)
        if upper is None:
            upper = torch.ones_like(x_raw) * 100

        # Handle broadcasting if bounds are 1D
        if lower.dim() == 1:
            lower = lower.unsqueeze(0).expand(batch_size, -1)
        if upper.dim() == 1:
            upper = upper.unsqueeze(0).expand(batch_size, -1)

        # Solve per batch item
        x_proj = []
        for i in range(batch_size):
            x_i, = self.cvxpy_layer(
                x_raw[i], A[i], b[i], lower[i], upper[i]
            )
            x_proj.append(x_i)
        return torch.stack(x_proj, dim=0)



class ProjectionFactory:
    _class_map = {
        'linear_violation':LinearViolationAdaption,
        'quadratic_program':QPProjectionWithSlack,
        'convex_program':CvxpyProjectionLayer,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict):
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
