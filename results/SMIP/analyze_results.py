import json
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Scenario Tree Evaluation Parameters")

parser.add_argument('--s', type=int, default=24, help='scenarios')
parser.add_argument('--p', type=int, default=3, help='Planning horizon or number of periods (p)')
parser.add_argument('--episodes', type=int, default=30, help='Number of episodes to evaluate')
parser.add_argument('--perfect_information', type=bool, default=True)
parser.add_argument('--gen', type=bool, default=False)
parser.add_argument('--cv', type=float, default=0.5, help='Coefficient of variation (cv)')
parser.add_argument('--teu', type=int, default=20000, help='TEU value')
args = parser.parse_args()

# Access arguments like variables
s = args.s
p = args.p
episodes = args.episodes
perfect_information = args.perfect_information
gen = args.gen
teu = args.teu
cv = args.cv

# folder = 'testing' if not gen else 'generalization'
# path = f'{folder}/cv={cv}'
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "scenario_tree", "block_mpp",)

# Data containers
obj = []
time = []
total_load = []
total_demand = []
total_revenue = []
total_cost = []

# Load data
for e in range(episodes):
    with open(f'{path}/results_scenario_tree_teu{teu}_p{p}_e{e}_s{s}_pi{perfect_information}_gen{gen}.json', 'r') as file:
        x = json.load(file)
        obj.append(x["obj"])
        time.append(x["time"])
        total_load.append(np.sum(x["mean_load_per_port"], axis=-1))
        total_demand.append(np.sum(x["mean_demand"], axis=-1))
        total_revenue.append(np.sum(x["mean_revenue"], axis=-1))
        total_cost.append(np.sum(x["mean_cost"], axis=-1))

# Statistics container
stats_summary = {}

# Helper function to collect and print statistics
def collect_stats(data, name=""):
    """Collects and prints summary statistics of the data"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    z = 1.96
    LB = mean - z * std / np.sqrt(n)
    UB = mean + z * std / np.sqrt(n)

    # Print
    print(f"\n{name}")
    print(f"n: {n}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"95% CI: ({LB}, {UB})")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")

    # Save in dictionary
    stats_summary[name] = {
        "n": n,
        "mean": mean,
        "std": std,
        "95% CI": [LB, UB],
        "min": min_val,
        "max": max_val
    }

# Collect and print stats
collect_stats(obj, "Objective")
collect_stats(time, "Time")
collect_stats(total_load, "Load")
collect_stats(total_demand, "Demand")
collect_stats(total_revenue, "Revenue")
collect_stats(total_cost, "Cost")

# Save to JSON file
path = os.path.join(base_dir, "AI2STOW", "teu20k", f"p{p}", "pi")
output_path = f"{path}/summary_stats_teu{teu}_p{p}_s{s}_pi{perfect_information}_gen{gen}.json"
with open(output_path, 'w') as f:
    json.dump(stats_summary, f, indent=4)

print(f"\nSummary statistics saved to {output_path}")
