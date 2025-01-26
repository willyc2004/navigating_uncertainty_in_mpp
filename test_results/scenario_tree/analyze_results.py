import json
import numpy as np

# Load JSON file
s = 20
episodes = 30
perfect_information = False
gen = False

obj = []
time = []
total_load = []
total_demand = []
total_revenue = []
total_cost = []
for e in range(episodes):
    with open(f'results_scenario_tree_e{e}_s{s}_pi{perfect_information}_gen{gen}.json', 'r') as file:
        x = json.load(file)
        obj.append(x["obj"])
        time.append(x["time"])
        total_load.append(np.sum(x["mean_load_per_port"], axis=-1))
        total_demand.append(np.sum(x["mean_demand"], axis=-1))
        total_revenue.append(np.sum(x["mean_revenue"], axis=-1))
        total_cost.append(np.sum(x["mean_cost"], axis=-1))


# folder = 'generalization'
# with open(f'{folder}/results_scenario_tree_s{s}_pi{perfect_information}.json', 'r') as file:
#     data = json.load(file)

# obj = []
# time = []
# total_load = []
# total_demand = []
# total_revenue = []
# total_cost = []
# for x in data:
#     obj.append(x["obj"])
#     time.append(x["time"])
#     total_load.append(np.sum(x["mean_load_per_port"], axis=-1))
#     total_demand.append(np.sum(x["mean_demand"], axis=-1))
#     total_revenue.append(np.sum(x["mean_revenue"], axis=-1))
#     total_cost.append(np.sum(x["mean_cost"], axis=-1))

# Helper function to print statistics
def print_stats(data, name=""):
    """Summary statistics of the data"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    max = np.max(data)
    min = np.min(data)
    z = 1.96
    LB = mean - z * std / np.sqrt(n)
    UB = mean + z * std / np.sqrt(n)
    print(f"\n{name}")
    print(f"n: {n}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"95% CI: ({LB}, {UB})")
    print(f"Min: {min}")
    print(f"Max: {max}")

print_stats(obj, "Objective")
print_stats(time, "Time")
print_stats(total_load, "Load")
print_stats(total_demand, "Demand")
print_stats(total_revenue, "Revenue")
print_stats(total_cost, "Cost")