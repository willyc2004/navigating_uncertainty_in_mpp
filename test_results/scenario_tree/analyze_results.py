import json
import numpy as np

# Load JSON file
s = 32
with open(f'results_scenario_tree_debug.json', 'r') as file:
    data = json.load(file)

obj = []
time = []
total_load = []
total_demand = []
for x in data:
    obj.append(x["obj"])
    time.append(x["time"])
    total_load.append(np.sum(x["mean_load_per_port"], axis=-1))
    total_demand.append(np.sum(x["mean_demand"], axis=-1))

# Helper function to print statistics
def print_stats(data, name=""):
    print(f"\n{name}")
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Min: {np.min(data)}")

print_stats(obj, "Objective")
print_stats(time, "Time")
print_stats(total_load, "Load")
print_stats(total_demand, "Demand")