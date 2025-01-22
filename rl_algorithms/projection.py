import torch as th
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from rl_algorithms.lp_solver import stepwise_lp

class EmptyLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x, **kwargs):
        return x

class LinearViolationAdaption(th.nn.Module):
    """Convex violation layer to enforce soft feasibility by projecting solutions towards feasible region."""

    def __init__(self, **kwargs):
        super(LinearViolationAdaption, self).__init__()
        self.alpha = kwargs.get('alpha', 0.005)
        self.delta = kwargs.get('delta', 0.01)
        self.tolerance = kwargs.get('tolerance', 0.01)
        self.max_iter = kwargs.get('max_iter', 100)

    def forward(self, x, A, b, **kwargs):
        """
        - alpha => 0.04 diverges, 0.025 might also cause nans
        - delta, tolerance = 0.05 causes overshooting demand

        Good settings (project per t): alpha=0.005, delta=0.01, tolerance=0.01
        """
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
        violation_old = th.zeros(batch_size, n_step, m, dtype=x.dtype, device=x.device)
        active_mask = th.ones(batch_size, n_step, dtype=th.bool, device=x.device)  # Start with all batches active

        # Start loop with early exit in case of nans
        if th.isnan(x_).any():
            return x_
        count = 0
        while th.any(active_mask):
            # Compute current violation for each batch and step
            violation_new = th.clamp(th.matmul(x_.unsqueeze(2), A.transpose(-2, -1)).squeeze(2) - b, min=0)
            # Shape: [batch_size, n_step, m]
            total_violation = th.sum(violation_new, dim=-1)  # Sum violations in [batch_size, n_step]

            # Define batch-wise stopping conditions
            no_violation = total_violation < self.tolerance
            stalling_check = th.abs(total_violation - th.sum(violation_old, dim=-1)) < self.delta

            # Update active mask: only keep batches and steps that are neither within tolerance nor stalled
            active_mask = ~(no_violation | stalling_check)

            # Break if no batches/steps are left active
            if not th.any(active_mask):
                break

            # Calculate penalty gradient for adjustment
            penalty_gradient = th.matmul(violation_new.unsqueeze(2), A).squeeze(2)  # Shape: [32, 1, 20]

            # Apply penalty gradient update only for active batches/steps
            x_ = th.where(active_mask.unsqueeze(2), x_ - self.alpha * penalty_gradient, x_)
            x_ = th.clamp(x_, min=0) # Ensure non-negativity

            # Update violation_old for the next iteration
            violation_old = violation_new.clone()
            count += 1
            if count > self.max_iter:
                break
        # Return the adjusted x_, reshaped to remove n_step dimension if it was initially 2D
        return x_.squeeze(1) if n_step == 1 else x_

class ConvexProgramLayer(th.nn.Module):
    """Convex programming layer to enforce strict feasibility by ensuring constraints are satisfied."""

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

    def forward(self, x, A, b, **kwargs):
        # Get the dimensions of the input
        batch_size = A.shape[0]
        x_out = []

        # Loop over each batch
        for i in range(batch_size):
            A_i = A[i].T
            b_i = b[i]
            x_i = x[i]
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

class LinearProgramLayer(th.nn.Module):
    """Linear programming layer to enforce strict feasibility by ensuring constraints are satisfied."""
    def __init__(self, **kwargs):
        super(LinearProgramLayer, self).__init__()

    def forward(self, x, A, b, **kwargs):
        x_ = th.zeros_like(x)
        for i in range(x.size(0)):
            # Solve the stepwise problem for each instance in the batch
            solution, _, _, _ = stepwise_lp(x[i], A[i], b[i], verbose=False)
            x_[i, :] = th.tensor(solution, device=x.device)
        return x_

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
        'convex_program': ConvexProgramLayer,
        'linear_program': LinearProgramLayer,
        'linear_violation':LinearViolationAdaption,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict):
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
