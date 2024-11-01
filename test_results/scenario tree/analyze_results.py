import json
import numpy as np

# Load JSON file
s = 24
with open(f'results_scenario_tree_s{s}_piTrue.json', 'r') as file:
    data = json.load(file)

obj = []
time = []
for x in data:
    obj.append(x["obj"])
    time.append(x["time"])

print(np.mean(obj))
print(np.std(obj))

print(np.mean(time))
print(np.std(time))