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
    z = mdl.continuous_var_list(num_constraints, lb=0, name="z")

    # Add constraints to the model
    for j in range(num_vars):
        mdl.add_constraint(x_[j] == (action[j] - s_min[j]))

    for i in range(num_constraints):
        mdl.add_constraint(mdl.sum(A[i][j] * x_[j] for j in range(num_vars)) <= b[i])

    # Set the objective function
    mdl.minimize(mdl.sum(s_min[j] for j in range(num_vars)))

    # # Add constraints to the model
    # for j in range(num_vars):
    #     mdl.add_constraint(x_[j] == (action[j] - s_min[j]))
    #
    # for i in range(num_constraints):
    #     if i != 0:
    #         mdl.add_constraint(mdl.sum(A[i][j] * x_[j] for j in range(num_vars)) - b[i] <= z[i])
    #
    # mdl.minimize(param * mdl.sum(z[i] for i in range(num_constraints)) + mdl.sum(s_min[j] for j in range(num_vars)))

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

def polwise_lp(util, demand, env, verbose=True,):
    """POL-wise LP to find feasible action"""
    # Create environment and MIP model
    mdl = Model(name='pol_wise_lp')

    # add mdl variables
    P = env.P
    B, D, T, K = env.B, env.D, env.T, env.K

    # detach
    vertical_position = env.vertical_position.detach().cpu().numpy()
    longitudinal_position = env.longitudinal_position.detach().cpu().numpy()
    weights = env.weights.detach().cpu().numpy()
    stab_delta = env.stab_delta
    LCG_target = env.LCG_target
    VCG_target = env.VCG_target

    # Util of shape [Sequence, num_actions]
    util = util.reshape(env.B, env.D, env.T, env.K)  # Reshape util to [B, D, T, K]
    demand = demand.reshape(env.T, env.K)  # Reshape demand to [P, P, K]

    # create continuous variables
    LM = {}  # Longitudinal moment
    VM = {}  # Vertical moment
    TW = {}  # Total weight

    # Stability:
    for pol in range(P):
        LM[pol] = mdl.continuous_var(name=f'LM_{pol}')
        VM[pol] = mdl.continuous_var(name=f'VM_{pol}')
        TW[pol] = mdl.continuous_var(name=f'TW_{pol}')

    # Utilization
    s = {} # Slack variable
    for pol in range(P):
        for b in range(B):
            for d in range(D):
                for tau in range(T):
                    for k in range(env.K):
                        s[b, d, tau, k] = mdl.continuous_var(name=f's_{pol}_{b}_{d}_{tau}_{k}')

    # Add constraints to the model
    for pol in range(P):
        # get sets
        transport_indices = [(i, j) for i in range(P) for j in range(P) if i < j]
        on_board = [transport_indices.index((i, j)) for i in range(P) for j in range(P) if i <= pol and j > pol]

        for tau in on_board:
            for k in range(env.K):
                # Demand satisfaction
                mdl.add_constraint(
                    mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for b in range(B) for d in range(D)) <= demand[tau, k]
                )

        # Stability
        mdl.add_constraint(
            TW[pol] == mdl.sum(weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k]
                                               for tau in on_board for d in range(D))
                          for k in range(K) for b in range(B))
        )

        # LCG
        mdl.add_constraint(
            LM[pol] == mdl.sum(longitudinal_position[b] * mdl.sum(weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k]
                                                                                  for tau in on_board for d in range(D))
                                                             for k in range(K)) for b in range(B)))
        mdl.add_constraint(
            stab_delta * mdl.sum(
                weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for d in range(D))
                for k in range(K) for b in range(B)) >= mdl.sum(longitudinal_position[b] * mdl.sum(
                weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for d in range(D)) for
                k in range(K)) for b in range(B)) - LCG_target * mdl.sum(
                weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for d in range(D)) for
                k in range(K) for b in range(B)))
        mdl.add_constraint(stab_delta * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for d in range(D)) for k in
            range(K) for b in range(B)) >= - mdl.sum(longitudinal_position[b] * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for d in range(D)) for k in
            range(K)) for b in range(B)) + LCG_target * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for d in range(D)) for k in
            range(K) for b in range(B)))

        # VCG
        mdl.add_constraint(VM[pol] == mdl.sum(vertical_position[d] * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K)) for d in range(D)))
        mdl.add_constraint(stab_delta * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)) >= mdl.sum(vertical_position[d] * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K)) for d in range(D)) - VCG_target * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)))
        mdl.add_constraint(stab_delta * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)) >= - mdl.sum(vertical_position[d] * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K)) for d in range(D)) + VCG_target * mdl.sum(
            weights[k] * mdl.sum(util[b, d, tau, k] - s[b, d, tau, k] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)))

    # Set the objective function
    mdl.minimize(mdl.sum(s[b, d, tau, k] for b in range(B) for d in range(D)
                         for tau in range(T) for k in range(K)))

    # Solve the problem
    mdl.set_time_limit(100) #3600
    mdl.parameters.mip.tolerances.mipgap = 0.0001  # 0.01%
    solution = mdl.solve(log_output=verbose)

    # Extract solution, objective value, optimality gap and time
    s_sol = np.zeros((B, D, T, K))
    try:
        # Extract solution
        for b in range(B):
            for d in range(D):
                for tau in range(T):
                    for k in range(K):
                        for pol in range(P):
                            s_sol[b, d, tau, k] = s[b, d, tau, k].solution_value
        # Compute objective value, optimality gap and time
        obj = solution.get_objective_value()
        opt_gap = mdl.solve_details.mip_relative_gap
        time = mdl.solve_details.time

        # Get utilization
        util_output = util - s_sol

    except:
        obj = np.nan
        opt_gap = np.nan
        time = np.nan
        util_output = util

        # Find violations
        print("-"*50)
        print("Infeasible MIP")
        print("-"*50)
        print("util", util)
        print("demand", demand)
        print("weights", weights)
        print("vertical_position", vertical_position)
        print("longitudinal_position", longitudinal_position)
        print("stab_delta", stab_delta)
        print("LCG_target", LCG_target)
        print("VCG_target", VCG_target)
        breakpoint()

    return util_output, obj, opt_gap, time