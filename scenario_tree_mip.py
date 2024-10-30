# Imports
import numpy as np
import torch as th
import random
import yaml
from dotmap import DotMap
from docplex.mp.model import Model
import sys
import os
import json
sys.path.append('/home/jaiv/ILOG/CPLEX_Studio2211/cplex/python/3.9/x86-64_linux')

# Module imports
path_to_main = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_to_main)
from main import adapt_env_kwargs, make_env
from environment.utils import get_pol_pod_pair

def main(config, scenarios_per_stage=32, seed=42, perfect_information=False, deterministic=False):
    # Create the environment on cpu
    env_kwargs = config.env
    env_kwargs.seed = seed
    env = make_env(env_kwargs, device='cpu')

    # Problem parameters
    P = env.P
    B = env.B
    D = env.D
    K = env.K
    T = env.T
    stab_delta = env.stab_delta
    LCG_target = env.LCG_target
    VCG_target = env.VCG_target
    CI_target_parameter = env.CI_target
    teus = env.teus.detach().cpu().numpy()
    weights = env.weights.detach().cpu().numpy()
    revenues = env.revenues.detach().cpu().numpy()
    capacity = env.capacity.detach().cpu().numpy()
    longitudinal_position = env.longitudinal_position.detach().cpu().numpy()
    vertical_position = env.vertical_position.detach().cpu().numpy()

    # Scenario tree parameters
    M = 10 ** 3 # Big M
    stages = P - 1  # Number of load ports (P-1)
    max_paths = scenarios_per_stage ** (stages-1) + 1
    num_nodes_per_stage = [1*scenarios_per_stage**stage for stage in range(stages)]

    # Precompute functions
    def precompute_node_list(stages, scenarios_per_stage):
        """Precompute the list of nodes and their coordinates in the scenario tree"""
        node_list = []  # List to store the coordinates of all nodes
        # Loop over each stage, starting from stage 1 (root is stage 1)
        for stage in range(stages):
            nodes_in_current_stage = scenarios_per_stage ** (stage)  # Number of nodes at this stage

            # For each node in the current stage
            for node_id in range(nodes_in_current_stage):
                node_list.append((stage, node_id))

        return node_list

    def precompute_demand(node_list):
        """Precompute the demand scenarios for each node in the scenario tree"""
        td = env.reset(batch_size=[max_paths])
        pregen_demand = td["realized_demand"].detach().cpu().numpy()

        # Preallocate demand array for transport demands
        demand_ = np.zeros((max_paths, K, P, P))
        # Precompute transport demands for all paths
        for transport in range(T):
            pol, pod = get_pol_pod_pair(th.tensor(transport), P)
            demand_[:, :, pol, pod] = pregen_demand[:, :, transport]

        demand_ = demand_.transpose(2, 0, 1, 3)

        # Populate demand scenarios
        demand_scenarios = {}
        for (stage, node_id) in node_list:
            demand_scenarios[stage, node_id] = demand_[stage, node_id, :, :]

        # Real demand
        real_demand = {}
        for stage in range(stages):
            real_demand[stage, max_paths, 0] = demand_[stage, -1, :, :]

        # Allow for perfect information
        if deterministic:
            for (stage, node_id) in node_list:
                demand_scenarios[stage, node_id] = real_demand[stage, max_paths, 0]

        return demand_scenarios, real_demand

    def get_demand_history(stage, demand):
        """Get the demand history up to the given stage for the given scenario"""
        if stage > 0:
            demand_history = []
            for s in range(stage):
                for node_id in range(num_nodes_per_stage[s]):
                    # Concatenate predicted demand history for the current scenario up to the given stage
                    demand_history.append(demand[s, node_id,].flatten())
            return np.array(demand_history)
        else:
            # If there's no history (stage 0), return an empty array or some other initialization
            return np.array([])  # Or use np.zeros((shape,))


    # Create a CPLEX model
    mdl = Model(name="multistage_mpp")

    # Decision variable dictionaries
    x = {} # Cargo allocation
    HO = {} # Hatch overstowage
    HM = {} # Hatch move
    CI = {} # Crane intensity
    CI_target = {} # Crane intensity target
    spread_moves_bay = {} # Spread moves bay
    LM = {} # Longitudinal moment
    VM = {} # Vertical moment
    TW = {} # Total weight

    # Sets of different stages
    on_boards = []
    all_port_moves = []
    all_load_moves = []
    transport_indices = [(i, j) for i in range(P) for j in range(P) if i < j]


    # Function to compute relevant sets for each stage
    def onboard_groups(ports:int, pol:int, transport_indices:list) -> np.array:
        load_index = np.array([transport_indices.index((pol, i)) for i in range(ports) if i > pol])  # List of cargo groups to load
        load = np.array([transport_indices[idx] for idx in load_index]).reshape((-1,2))
        discharge_index = np.array([transport_indices.index((i, pol)) for i in range(ports) if i < pol])  # List of cargo groups to discharge
        discharge = np.array([transport_indices[idx] for idx in discharge_index]).reshape((-1,2))
        port_moves = np.vstack([load, discharge]).astype(int)
        on_board = [(i, j) for i in range(ports) for j in range(ports) if i <= pol and j > pol]  # List of cargo groups to load
        return np.array(on_board), port_moves, load

    def build_tree(stages, demand):
        """Function to build the scenario tree; with decisions and constraints for each node"""
        for stage in range(stages):
            for node_id in range(num_nodes_per_stage[stage]):
                # Crane intensity:
                CI[stage, node_id] = mdl.continuous_var(name=f'CI_{stage}_{node_id}')
                CI_target[stage, node_id] = mdl.continuous_var(name=f'CI_target_{stage}_{node_id}')
                spread_moves_bay[stage, node_id] = mdl.continuous_var(name=f'spread_moves_bay_{stage}_{node_id}')

                # Stability:
                LM[stage, node_id] = mdl.continuous_var(name=f'LM_{stage}_{node_id}')
                VM[stage, node_id] = mdl.continuous_var(name=f'VM_{stage}_{node_id}')
                TW[stage, node_id] = mdl.continuous_var(name=f'TW_{stage}_{node_id}')

                for bay in range(B):
                    # Hatch overstowage:
                    HO[stage, node_id, bay] = mdl.continuous_var(name=f'HO_{stage}_{node_id}_{bay}')
                    HM[stage, node_id, bay] = mdl.binary_var(name=f'HM_{stage}_{node_id}_{bay}')

                    for deck in range(D):
                        for cargo_class in range(K):
                            for pol in range(stage + 1):
                                for pod in range(pol + 1, P):
                                    # Cargo allocation:
                                    x[stage, node_id, bay, deck, cargo_class, pol, pod] = \
                                            mdl.continuous_var(name=f'x_{stage}_{node_id}_{bay}_{deck}_{cargo_class}_{pol}_{pod}')

            # Define sets
            # Current port
            on_board, port_moves, load_moves = onboard_groups(P, stage, transport_indices)
            on_boards.append(on_board)
            all_port_moves.append(port_moves)
            all_load_moves.append(load_moves)

            # constraints for the current stage and node
            for node_id in range(num_nodes_per_stage[stage]):
                print(f"Stage {stage}, Node {node_id}")
                for (i, j) in on_board:
                    for k in range(K):
                        # Demand satisfaction
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, d, k, i, j] for b in range(B) for d in range(D))
                            <= demand[stage, node_id][k, j]
                        )

                for b in range(B):
                    for d in range(D):
                        # TEU capacity
                        mdl.add_constraint(
                            mdl.sum(teus[k] * mdl.sum(x[stage, node_id, b, d, k, i, j]
                                                      for (i, j) in on_board) for k in range(K))
                            <= capacity[b, d]
                        )

                        if not perfect_information:
                            demand_history1 = get_demand_history(stage, demand)
                            for node_id2 in range(node_id + 1, num_nodes_per_stage[stage]):
                                demand_history2 = get_demand_history(stage, demand)
                                if np.allclose(demand_history1, demand_history2, atol=1e-5):  # Use a tolerance if floats
                                    for k in range(K):
                                        for (i, j) in load_moves:
                                            # Non-anticipation at stage, provided demand history is similar
                                            mdl.add_constraint(
                                                x[stage, node_id, b, d, k, i, j] == x[stage, node_id2, b, d, k, i, j]
                                            )

                    # Open hatch (d=1 is below deck)
                    mdl.add_constraint(
                        mdl.sum(x[stage, node_id, b, 1, k, i, j] for (i, j) in all_port_moves[stage] for k in range(K))
                        <= M * HM[stage, node_id, b]
                    )

                    # Hatch overstows (d=0 is on deck)
                    # Overstowage is arrival condition of previous port: on_boards[stage - 1]
                    # Vessel is empty before stage 0, hence no overstows
                    if stage > 0:
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, 0, k, i, j]for (i, j) in on_boards[stage - 1]
                                    for k in range(K) if j > stage) - M * (1 - HM[stage, node_id, b] )
                            <= HO[stage, node_id, b]
                        )

                # Stability
                mdl.add_constraint(
                    TW[stage, node_id] == mdl.sum(weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j]
                                                                               for (i, j) in on_board for d in range(D))
                                                          for k in range(K) for b in range(B))
                )

                # LCG
                mdl.add_constraint(
                    LM[stage, node_id] == mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D))
                    for k in range(K)) for b in range(B)))
                mdl.add_constraint(
                    stab_delta * mdl.sum(weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D))
                    for k in range(K) for b in range(B)) >= mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K)) for b in range(B)) - LCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K) for b in range(B)))
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K) for b in range(B)) >= - mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K)) for b in range(B)) + LCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K) for b in range(B)))

                # VCG
                mdl.add_constraint(VM[stage, node_id] == mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K)) for d in range(D)))
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)) >= mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K)) for d in range(D)) - VCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)))
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)) >= - mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K)) for d in range(D))  + VCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)))

                # Compute lower bound on long crane
                for adj_bay in range(B - 1):
                    mdl.add_constraint(
                        mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in all_port_moves[stage] for k in range(K)
                                for d in range(D) for b in [adj_bay, adj_bay + 1])
                        <= CI[stage, node_id]
                    )
                # Ensure CI[stage] is bounded by CI_target
                mdl.add_constraint(
                    CI_target[stage, node_id] ==
                    CI_target_parameter * 2 / B * mdl.sum(demand[stage, node_id][k, j]
                                                          for (i, j) in all_port_moves[stage] for k in range(K))
                )
                mdl.add_constraint(
                    spread_moves_bay[stage, node_id] ==
                    2 / B * mdl.sum(demand[stage, node_id][k, j]
                                    for (i, j) in all_port_moves[stage] for k in range(K))
                )
                mdl.add_constraint(CI[stage, node_id] <= CI_target[stage, node_id])

    # Precompute nodelist and demand
    node_list = precompute_node_list(stages, scenarios_per_stage)
    demand, real_demand = precompute_demand(node_list)
    # Build the scenario tree
    build_tree(stages, demand)

    # Define the objective function
    # Objective: Maximize the total expected revenue over all scenarios
    probabilities = {}
    for (stage, node_id) in node_list:
        probabilities[stage, node_id] = 1 / num_nodes_per_stage[stage]

    objective = mdl.sum(
        probabilities[stage, node_id] * (
            mdl.sum(
                revenues[transport_indices.index((stage, j))] * x[stage, node_id, b, d, k, stage, j]
                for j in range(stage + 1, P) # Loop over discharge ports
                for b in range(B)  # Loop over bays
                for d in range(D)  # Loop over decks
                for k in range(K)  # Loop over cargo classes
            )
            - mdl.sum(env.ho_costs * HO[stage, node_id, b] for b in range(B))
        )
        for stage in range(stages)  # Iterate over all stages
        for node_id in range(num_nodes_per_stage[stage])  # Iterate over nodes at each stage
    )
    mdl.maximize(objective)
    mdl.context.cplex_parameters.read.datacheck = 2
    mdl.parameters.mip.strategy.file = 3
    mdl.parameters.emphasis.memory = 1  # Prioritize memory savings over speed
    mdl.parameters.threads = 1  # Use only 1 thread to reduce memory usage
    mdl.parameters.mip.tolerances.mipgap = 0.001  # 0.1%

    # Solve the model
    solution = mdl.solve(log_output=True)

    # Print the solution
    if solution:
        print(f"Objective value: {solution.objective_value}")

        # Analyze the solution
        x_ = np.zeros((stages, max_paths, B, D, K, P, P))
        HO_ = np.zeros((stages, max_paths, B,))
        HM_ = np.zeros((stages, max_paths, B,))
        CI_ = np.zeros((stages, max_paths,))
        CI_target_ = np.zeros((stages, max_paths,))
        spread_moves_bay_ = np.zeros((stages, max_paths,))
        LM_ = np.zeros((stages, max_paths,))
        VM_ = np.zeros((stages, max_paths,))
        TW_ = np.zeros((stages, max_paths,))

        for stage in range(stages):
            for node_id in range(num_nodes_per_stage[stage]):
                for bay in range(B):
                    for deck in range(D):
                        for cargo_class in range(K):
                            for pol in range(stage + 1):
                                for pod in range(pol + 1, P):
                                    x_[stage, node_id, bay, deck, cargo_class, pol, pod] = x[stage, node_id, bay, deck, cargo_class, pol, pod].solution_value

                    HO_[stage, node_id, bay] = HO[stage, node_id, bay].solution_value
                    HM_[stage, node_id, bay] = HM[stage, node_id, bay].solution_value

                CI_[stage, node_id] = CI[stage, node_id].solution_value
                CI_target_[stage, node_id] = CI_target[stage, node_id].solution_value
                spread_moves_bay_[stage, node_id] = spread_moves_bay[stage, node_id].solution_value

                LM_[stage, node_id] = LM[stage, node_id].solution_value
                VM_[stage, node_id] = VM[stage, node_id].solution_value
                TW_[stage, node_id] = TW[stage, node_id].solution_value

        results = {
            "seed":seed,
            "ports":P,
            "scenarios":scenarios_per_stage,
            "obj":solution.objective_value,
            "time":mdl.solve_details.time,
            "gap":mdl.solve_details.mip_relative_gap,
        }
        vars = {
            "seed": seed,
            "ports": P,
            "scenarios": scenarios_per_stage,
            "x": x_.tolist(),
            "HO_": HO_.tolist(),
            "HM_": HM_.tolist(),
            "CI_": CI_.tolist(),
            "CI_target_": CI_target_.tolist(),
            "spread_moves_bay_": spread_moves_bay_.tolist(),
            "LM_": LM_.tolist(),
            "VM_": VM_.tolist(),
            "TW_": TW_.tolist(),
        }

        return results, vars
    else:
        # Print the error
        print("No solution found")

def set_unique_seed(batch_index, base_seed=42):
    """Set a unique seed per batch."""
    seed = base_seed + batch_index
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    # Load the configuration file
    with open(f'{path_to_main}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    # Run main for different seeds and number of scenarios
    perfect_information = False
    deterministic = False

    num_seed = 20
    for scen in [4,8,12,16,20]:#,24,32]:
        results = []
        vars = []
        for x in range(num_seed):
            seed = config.env.seed + x
            set_unique_seed(seed)
            result, var = main(config, scen, seed, perfect_information, deterministic)
            results.append(result)
            vars.append(var)

        # Save results to a JSON file
        with open(f"results_scenario_tree_s{scen}_pi{perfect_information}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
        with open(f"variables_scenario_tree_s{scen}_pi{perfect_information}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)