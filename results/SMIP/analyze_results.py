import json
import numpy as np
import os

# Parameters
s = 4
p = 4 - 1
episodes = 30
perfect_information = False
gen = False
cv = 0.5
teu = 20000
# folder = 'testing' if not gen else 'generalization'
# path = f'{folder}/cv={cv}'
# path = "scenario_tree/block_mpp"
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "AI2STOW", "teu20k", f"p{p}", "na")

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
output_path = f"{path}/summary_stats_teu{teu}_p{p}_s{s}_pi{perfect_information}_gen{gen}.json"
with open(output_path, 'w') as f:
    json.dump(stats_summary, f, indent=4)

print(f"\nSummary statistics saved to {output_path}")
