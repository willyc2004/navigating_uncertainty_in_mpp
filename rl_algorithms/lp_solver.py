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
    # Create environment and MIP model
    model = Model(name='stepwise_lp')

    # Get parameters, detach and reshape
    action, A, b = to_numpy(action, A, b)
    n = action.shape[0]
    m = b.shape[0]

    # Decision variables
    s = model.continuous_var_list(n, name="s", lb=0)  # s >= 0
    x = model.continuous_var_list(n, name="x", lb=0)  # x >= 0
    slack = model.continuous_var_list(m, name="slack", lb=0)  # slack >= 0
    slack_param = 10

    # Constraints
    model.add_constraints(A[j, :] @ x <= b[j] for j in range(m))  # A x <= b (optional: + slack[j] )
    model.add_constraints(x[i] == action[i] - s[i] for i in range(n))  # x = action - s
    model.add_constraints(s[i] <= action[i] for i in range(n))  # s <= action

    # Objective function: Minimize sum of s
    model.set_time_limit(100)
    model.parameters.mip.tolerances.mipgap = 0.0001  # 0.01%
    model.minimize(model.sum(s)) # + slack_param * model.sum(slack))

    # Solve the model
    solution = model.solve(log_output=verbose)


    # Check if the solution exists
    if solution:
        obj = solution.get_objective_value()
        opt_gap = model.solve_details.mip_relative_gap
        time = model.solve_details.time
        s_opt = np.array([solution[s[i]] for i in range(n)])
        x_opt = np.array([solution[x[i]] for i in range(n)])
        if verbose:
            print("-"*50)
            print("Optimal s:", s_opt.mean())
            print("Optimal x:", x_opt.mean())
        return x_opt, obj, opt_gap, time
    else:
        raise Exception("No solution found.")


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

    # Reshape
    util = util.reshape(env.B, env.D, env.K, env.T)
    demand = demand.reshape(env.K, env.T)

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
    x = {} # Utilization variable
    for pol in range(P):
        for b in range(B):
            for d in range(D):
                for tau in range(T):
                    for k in range(env.K):
                        s[b, d, k, tau] = mdl.continuous_var(name=f's_{pol}_{b}_{d}_{k}_{tau}')
                        x[b, d, k, tau] = mdl.continuous_var(name=f'x_{pol}_{b}_{d}_{k}_{tau}')

    # Add constraints to the model
    for pol in range(P):
        # get sets
        transport_indices = [(i, j) for i in range(P) for j in range(P) if i < j]
        on_board = [transport_indices.index((i, j)) for i in range(P) for j in range(P) if i <= pol and j > pol]

        for tau in on_board:
            for k in range(env.K):
                for b in range(B):
                    for d in range(D):
                        # Define utilization x
                        mdl.add_constraint(
                            x[b, d, k, tau] == util[b, d, k, tau] - s[b, d, k, tau]
                        )

                # Demand satisfaction
                mdl.add_constraint(
                    mdl.sum(x[b, d, k, tau] for b in range(B) for d in range(D)) <= demand[k, tau]
                )

        # Stability
        mdl.add_constraint(
            TW[pol] == mdl.sum(weights[k] * mdl.sum(x[b, d, k, tau]
                                               for tau in on_board for d in range(D))
                          for k in range(K) for b in range(B))
        )

        # LCG
        mdl.add_constraint(
            LM[pol] == mdl.sum(longitudinal_position[b] * mdl.sum(weights[k] * mdl.sum(x[b, d, k, tau]
                                                                                  for tau in on_board for d in range(D))
                                                             for k in range(K)) for b in range(B)))
        mdl.add_constraint(
            stab_delta * mdl.sum(
                weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for d in range(D))
                for k in range(K) for b in range(B)) >= mdl.sum(longitudinal_position[b] * mdl.sum(
                weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for d in range(D)) for
                k in range(K)) for b in range(B)) - LCG_target * mdl.sum(
                weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for d in range(D)) for
                k in range(K) for b in range(B)))
        mdl.add_constraint(stab_delta * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for d in range(D)) for k in
            range(K) for b in range(B)) >= - mdl.sum(longitudinal_position[b] * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for d in range(D)) for k in
            range(K)) for b in range(B)) + LCG_target * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for d in range(D)) for k in
            range(K) for b in range(B)))

        # VCG
        mdl.add_constraint(VM[pol] == mdl.sum(vertical_position[d] * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K)) for d in range(D)))
        mdl.add_constraint(stab_delta * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)) >= mdl.sum(vertical_position[d] * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K)) for d in range(D)) - VCG_target * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)))
        mdl.add_constraint(stab_delta * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)) >= - mdl.sum(vertical_position[d] * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K)) for d in range(D)) + VCG_target * mdl.sum(
            weights[k] * mdl.sum(x[b, d, k, tau] for tau in on_board for b in range(B)) for k in
            range(K) for d in range(D)))

    # Set the objective function
    mdl.minimize(mdl.sum(s[b, d, k, tau] for b in range(B) for d in range(D)
                         for k in range(K) for tau in range(T)))

    # Solve the problem
    mdl.set_time_limit(100) #3600
    mdl.parameters.mip.tolerances.mipgap = 0.0001  # 0.01%
    solution = mdl.solve(log_output=verbose)

    # Extract solution, objective value, optimality gap and time
    s_sol = np.zeros((B, D, K, T))
    x_sol = np.zeros((B, D, K, T))
    try:
        # Extract solution
        for b in range(B):
            for d in range(D):
                for tau in range(T):
                    for k in range(K):
                        for pol in range(P):
                            s_sol[b, d, k, tau] = s[b, d, k, tau].solution_value
                            x_sol[b, d, k, tau] = x[b, d, k, tau].solution_value

        # Compute objective value, optimality gap and time
        obj = solution.get_objective_value()
        opt_gap = mdl.solve_details.mip_relative_gap
        time = mdl.solve_details.time

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

    return x_sol, obj, opt_gap, time