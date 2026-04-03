import json
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

DATA_FOLDER = "data"

trainer_to_k = {
    0: 50,
    1: 80,
    2: 180,
    3: 320,
    4: 520
}

layer_order = [1, 10, 23]

files = sorted(
    [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")],
    key=lambda name: int(re.search(r"\.(\d+)\.json$", name).group(1))
)

layer_points = defaultdict(list)

for i, file_name in enumerate(files):
    path = os.path.join(DATA_FOLDER, file_name)

    with open(path, "r") as f:
        data = json.load(f)

    delta = data["delta_lm_loss(mask)"]

    layer_idx = i // 5
    trainer = i % 5

    if layer_idx >= len(layer_order):
        print(f"Skipping extra file: {file_name}")
        continue

    layer = layer_order[layer_idx]
    k = trainer_to_k[trainer]

    print(f"{file_name} -> layer {layer}, trainer {trainer}, k={k}, delta={delta}")

    layer_points[layer].append((k, delta))

for layer in layer_order:
    points = sorted(layer_points[layer], key=lambda x: x[0])
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    plt.plot(x_vals, y_vals, marker="o", label=f"L{layer}")

plt.xlabel("Sparsity (k)")
plt.ylabel("Cross Entropy Delta")
plt.title("Delta vs Sparsity by Layer")
plt.legend()
plt.show()