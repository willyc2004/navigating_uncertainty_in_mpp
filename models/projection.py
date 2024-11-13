from typing import Any, Tuple, Union
import torch as th
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from models.lp_solver import stepwise_lp

class EmptyLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x, A, b):
        return x

class ConvexProgramLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(ConvexProgramLayer, self).__init__()
        # Input
        n_action = kwargs['n_action']
        n_constraints = kwargs['n_constraints']

        # Create convex program
        x_ = cp.Variable(n_action)
        slack_minus = cp.Variable(n_action)
        x = cp.Parameter(n_action)
        A = cp.Parameter((n_action, n_constraints))
        b = cp.Parameter(n_constraints)
        constraints = [slack_minus >= 0, x_ >= 0,
                       x_ == (x - slack_minus),
                       A.T @ x_ <= b]
        objective = cp.Minimize(cp.sum(slack_minus))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        # Create CVXPY layer
        self.cvxpylayer = CvxpyLayer(problem, parameters=[A, b, x], variables=[slack_minus, x_])
        self.solver_options = {
            'solve_method': "ECOS",  # "SCS", "ECOS"
            'max_iters': 1000, # creates feasible solutions; but convex layer takes long
        }
        # check code: https://github.com/INFERLab/PROF/blob/main/agents/nn_policy.py

    def forward(self, x, A, b, t):
        # Get the dimensions of the input
        batch_size = A.shape[0]
        x_out = []

        # Loop over each batch
        for i in range(batch_size):
            A_i = A[i].T
            b_i = b[i]
            x_i = x[i]
            Ax = A_i.T @ x_i
            print("-"*50)
            print(f"Batch {i}: \n A_i: {A_i},\n b_i: {b_i},\n x_i: {x_i}, \n A_i x_i: {Ax} \n t: {t[0]}")
            try:
                # Solve the convex problem for each batch element
                _, x_ = self.cvxpylayer(A_i, b_i, x_i, solver_args=self.solver_options)
                x_out.append(x_)
            except cp.error.SolverError as e:
                print(f"Solver failed for batch {i}: {e}")
                x_out.append(th.zeros_like(x_i))  # Handle failure case

        # Stack the results back into a single tensor
        x_out = th.stack(x_out)
        return x_out  # Scale back to original values

class LinearViolationAdaption(th.nn.Module):
    def __init__(self, **kwargs):
        super(LinearViolationAdaption, self).__init__()

    def forward(self, x, A, b, alpha=0.005, delta=0.05, tolerance=0.05):
        # alpha => 0.04 diverges,
        # validation settings: alpha=0.025, delta=0.05, tolerance=0.05
        # Determine the shape based on dimensionality of b
        x_ = x.clone()
        if b.dim() == 2:
            batch_size, m = b.shape
            n_step = 1
            A = A.unsqueeze(1)  # Expand to [batch_size, 1, m, F] for consistency
            b = b.unsqueeze(1)  # Expand to [batch_size, 1, m] for consistency
            x_ = x_.unsqueeze(1)  # Expand to [batch_size, 1, F] for consistency
        elif b.dim() == 3:
            batch_size, n_step, m = b.shape
        else:
            raise ValueError("Invalid shape of 'b'. Expected dimensions 2 or 3.")
        violation_old = th.zeros(batch_size, n_step, m, dtype=x.dtype, device=x.device)
        active_mask = th.ones(batch_size, n_step, dtype=th.bool, device=x.device)  # Start with all batches active
        if th.isnan(x_).any():
            return x_

        count = 0
        while th.any(active_mask):
            # Compute current violation for each batch and step
            violation_new = th.clamp(th.matmul(x_.unsqueeze(2), A.transpose(-2, -1)).squeeze(2) - b, min=0)
            # Shape: [batch_size, n_step, m]
            total_violation = th.sum(violation_new, dim=-1)  # Sum violations in [batch_size, n_step]

            # Define batch-wise stopping conditions
            no_violation = total_violation < tolerance
            # print("count", count, "total_violation", total_violation.mean(),
            #       "diff", th.abs(total_violation - th.sum(violation_old, dim=-1)).mean())
            stalling_check = th.abs(total_violation - th.sum(violation_old, dim=-1)) < delta

            # Update active mask: only keep batches and steps that are neither within tolerance nor stalled
            active_mask = ~(no_violation | stalling_check)

            # Break if no batches/steps are left active
            if not th.any(active_mask):
                break

            # Calculate penalty gradient for adjustment
            penalty_gradient = th.matmul(violation_new.unsqueeze(2), A).squeeze(2)  # Shape: [32, 1, 20]

            # Apply penalty gradient update only for active batches/steps
            x_ = th.where(active_mask.unsqueeze(2), x_ - alpha * penalty_gradient, x_)

            # Update violation_old for the next iteration
            violation_old = violation_new.clone()
            count += 1
        print("tot_count", count)
        # Return the adjusted x_, reshaped to remove n_step dimension if it was initially 2D
        if n_step == 1:
            return x_.squeeze(1)
        else:
            return x_

class LinearProgramLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(LinearProgramLayer, self).__init__()

    def forward(self, x, A, b,):
        x_ = th.zeros_like(x)
        for i in range(x.size(0)):
            # Solve the stepwise problem for each instance in the batch
            solution, _, _, _ = stepwise_lp(x[i], A[i], b[i], verbose=True)
            x_[i, :] = th.tensor(solution, device=x.device)
        return x_

class WorstcaseViolationLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(WorstcaseViolationLayer, self).__init__()

    def forward(self, x, A, b, violation=None, eps=1e-3):
        """Scale x by worst-case constraint violation based on exponential similarity"""
        eta = th.clamp(self.exponential_similarity(violation), min=eps)
        eta_min = th.min(eta, dim=1).values
        return x * eta_min.unsqueeze(-1), eta_min

    def exponential_similarity(self, x, y=None, k=0.1):
        return th.exp(-k * x) # Compute the exponential similarity

class WorstcaseViolationLayerIter(WorstcaseViolationLayer):
    def __init__(self, **kwargs):
        """Run the worst-case violation layer for X iterations."""
        super(WorstcaseViolationLayer, self).__init__()
        self.iters = kwargs.get('iters', None)

    def forward(self, x, A, b, **kwargs):
        for _ in range(self.iters):
            violation = th.clamp(th.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b, 0)
            x, _ = super().forward(x, A, b, violation)
        return x

class WorstcaseViolationLayerWhile(WorstcaseViolationLayer):
    def __init__(self, **kwargs):
        """Run the worst-case violation layer until all constraints are satisfied."""
        super(WorstcaseViolationLayer, self).__init__()

    def forward(self, x, A, b, delta=0.01, **kwargs):
        # Initialize max_violation and active mask
        eta_min_old = th.zeros(x.size(0), dtype=th.float16, device=x.device)
        active_mask = th.ones(x.size(0), dtype=th.bool, device=x.device)  # All batches are initially active
        # count = 0
        while th.any(active_mask):
            # Compute new violations and x
            violation = th.clamp(th.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b, 0)
            x_new, eta_min_new = super().forward(x, A, b, violation=violation)

            # Update max_violation only for active batches where violation is improving
            improving_mask = eta_min_new - eta_min_old > delta
            active_mask = active_mask & improving_mask
            # Update for improving batches; unchanged otherwise
            x = th.where(improving_mask.unsqueeze(1), x_new, x)
            # Update eta_min_old
            eta_min_old = eta_min_new
        #     count += 1
        # # print("eta_min,", eta_min_old.mean())
        # # print("count", count)
        # print("-"*50)
        return x

class ProjectionFactory:
    """
    Projection layers to project the input tensor to the feasible set.
    - Convex programming:   Use cvxpylayers to solve a convex optimization problem.
    - Linear programming:   Call CPU solver to solve a linear programming problem.
    - Differentiable lp:    Use differentiable linear programming to solve a linear programming problem.
    - Maximum violation:    Scale action with respect to maximum constraint violation.
    - None:                 Empty layer.
    """
    _class_map = {
        'worst_case_violation': WorstcaseViolationLayerWhile,
        'worst_case_violation_iter': WorstcaseViolationLayerIter,
        'convex_program': ConvexProgramLayer,
        'linear_program': LinearProgramLayer,
        'linear_violation':LinearViolationAdaption,
        "linear_violation_sample": LinearViolationAdaption,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict):
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
