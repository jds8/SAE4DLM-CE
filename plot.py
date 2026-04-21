import json
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
import glob

# DATA_FOLDER = "train_dlm_sae/saes_mask_Dream-org_Dream-v0-Base-7B_top_k"
DATA_FOLDER = "qwen-topk-saes"

is_dlm = 'mask' in DATA_FOLDER

trainer_to_k = {
    0: 50,
    1: 80,
    2: 180,
    3: 320,
    4: 520,
    5: 820,
}

layer_order = [1, 5, 10, 14, 23, 27]

files = sorted(
    [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")],
    key=lambda name: int(re.search(r"\.(\d+)\.json$", name).group(1))
)

layer_points = defaultdict(list)

# ── Load eval results from directory structure ───────────────────────────
if is_dlm:
    result_paths = glob.glob(f"{DATA_FOLDER}/**/trainer_*/og_delta*(mask*.json", recursive=True)
else:
    result_paths = glob.glob(f"{DATA_FOLDER}/**/trainer_*/og_delta*.json", recursive=True)

for path in result_paths:

    with open(path) as f:
        data = json.load(f)

    if is_dlm:
        delta = data["delta_lm_loss(mask)"]
    else:
        delta = data["delta_lm_loss"]

    parts = path.split(os.sep)

    # Expect: ... / resid_post_layer_X / trainer_Y / delta_eval_*.json
    try:
        layer_str = parts[-3].split('_')[-1]
        trainer_str = parts[-2].split('_')[-1]
    except IndexError:
        continue

    layer = int(layer_str)
    trainer = int(trainer_str)

    k = trainer_to_k[trainer]

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
# plt.savefig('dream_smoltalk_og_denoising.pdf', format='pdf')
plt.savefig('qwen_pile_og_denoising.pdf', format='pdf')
# plt.show()
