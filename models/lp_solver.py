# Description: This file contains the function to solve the LP problem using CPLEX
# Author: Jaike van Twiller

# Import the required libraries
import torch as th
import numpy as np
from docplex.mp.model import Model
import sys
sys.path.append('\\wsl.localhost\\Ubuntu-22.04\\home\\jaiv\\.pyenv\\versions\\3.9.18\\envs\\master_planning\\lib\\python3.9\\site-packages\\cplex')

def to_numpy(*args):
    """Convert each argument to a NumPy array if it's a PyTorch tensor."""
    result = []
    for arg in args:
        try:
            # Check if the input is a PyTorch tensor
            if isinstance(arg, th.Tensor):
                # Detach, move to CPU, and convert to NumPy array
                result.append(arg.detach().cpu().numpy())
            else:
                # If it's not a tensor, keep it as it is
                result.append(np.array(arg))
        except:
            # If something goes wrong, leave it unchanged
            result.append(arg)
    return result

def stepwise_lp(action, A, b, verbose=True,):
    """Stepwise LP to find feasible action"""
    # Create environment and MIP model
    mdl = Model(name='stepwise_lp')

    # Get parameters, detach and reshape
    action, A, b = to_numpy(action, A, b)
    num_vars = action.shape[0]
    num_constraints = b.shape[0]

    # Add variables to the model
    s_min = mdl.continuous_var_list(num_vars, lb=0, name="s-")
    # s_plus = mdl.continuous_var_list(num_vars, lb=0, name="s+")
    x_ = mdl.continuous_var_list(num_vars, lb=0, name="x_")

    # Set the objective function
    mdl.minimize(mdl.sum(s_min[j] for j in range(num_vars)))

    # Add constraints to the model
    for j in range(num_vars):
        mdl.add_constraint(x_[j] == (action[j] - s_min[j]))

    for i in range(num_constraints):
        mdl.add_constraint(mdl.sum(A[i][j] * x_[j] for j in range(num_vars)) <= b[i])

    # Solve the problem
    mdl.set_time_limit(100) #3600
    mdl.parameters.mip.tolerances.mipgap = 0.0001  # 0.01%
    solution = mdl.solve(log_output=verbose)
    # print("Solution status: ", mdl.get_solve_status())

    # Extract solution, objective value, optimality gap and time
    x_sol = np.zeros((num_vars,))
    try:
        # Extract solution
        for j in range(num_vars):
            x_sol[j] = x_[j].solution_value
        # Compute objective value, optimality gap and time
        # print("action", action)
        # print("x_sol", x_sol)
        obj = solution.get_objective_value()
        opt_gap = mdl.solve_details.mip_relative_gap
        time = mdl.solve_details.time

    except:
        obj = np.nan
        opt_gap = np.nan
        time = np.nan
        x_sol = action

        # Find violations
        print("-"*50)
        print("Infeasible MIP")
        print("-"*50)
        print("A", A)
        print("b", b)
        print("action", action)
        breakpoint()
        # Ax = A.T @ action.reshape(-1)
        # diff = b - Ax
        # print("Ax", Ax)
        # print("diff", diff)


        pass
    return x_sol, obj, opt_gap, time