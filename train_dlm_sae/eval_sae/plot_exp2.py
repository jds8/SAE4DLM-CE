import json
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

# Gets the exact folder where plot_exp2.py is located and looks for the "exp2_results" folder
# (Change "exp2_results" to "data" if you moved the files back to a folder named "data")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "exp2_results") 

# The 6 k-values you found
trainer_to_k = {
    0: 50,  
    1: 80,  
    2: 160, 
    3: 320, 
    4: 520, 
    5: 820  
}

# The 6 layers evaluated in Experiment 2
layer_order = [1, 5, 10, 14, 23, 27] 

def get_file_index(filename):
    """
    Parses the index from files named like: 
    'delta_lm_loss_random(mask).json' -> returns 0
    'delta_lm_loss_random(mask) (1).json' -> returns 1
    """
    match = re.search(r"\((\d+)\)\.json$", filename)
    if match:
        return int(match.group(1))
    return 0

# 1. Read files and filter for ONLY the "(mask)" files
files = [
    f for f in os.listdir(DATA_FOLDER) 
    if f.endswith(".json") and "(mask)" in f
]

# 2. Sort files numerically using the custom regex index
files = sorted(files, key=get_file_index)

layer_points = defaultdict(list)

for i, file_name in enumerate(files):
    path = os.path.join(DATA_FOLDER, file_name)

    with open(path, "r") as f:
        data = json.load(f)

    delta = data["delta_lm_loss(mask)"]

    # 3. Math for 6 files per layer
    layer_idx = i // 6
    trainer = i % 6

    if layer_idx >= len(layer_order):
        print(f"Skipping extra file: {file_name}")
        continue

    layer = layer_order[layer_idx]
    k = trainer_to_k[trainer]

    print(f"{file_name} -> layer {layer}, trainer {trainer}, k={k}, delta={delta}")

    layer_points[layer].append((k, delta))

# 4. Plotting
plt.figure(figsize=(10, 6))

for layer in layer_order:
    if layer in layer_points and len(layer_points[layer]) > 0:
        # Sort points by k value to ensure the line connects properly left-to-right
        points = sorted(layer_points[layer], key=lambda x: x[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        plt.plot(x_vals, y_vals, marker="o", label=f"L{layer}")

plt.xlabel("Sparsity (k)")
plt.ylabel("Cross Entropy Delta")
plt.title("Random Weights: Delta vs Sparsity by Layer (Exp 2)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()