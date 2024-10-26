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
        # # Solve the convex program, with A_, b, x as inputs
        # A_ = A[0].T
        # _, x_, = self.cvxpylayer(A_, b, x, solver_args=self.solver_options)
        # return x_

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


class LinearProgramLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(LinearProgramLayer, self).__init__()

    def forward(self, x, A, b,):
        x_ = th.zeros_like(x)
        for i in range(x.size(0)):
            # Solve the stepwise problem for each instance in the batch
            solution, _, _, _ = stepwise_lp(x[i], A[i], b[i], verbose=False)
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
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict):
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
