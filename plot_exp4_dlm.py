import json
import glob
import os
import re

import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────
# sae_root = "dream_saes/saes_mask_Dream-org_Dream-v0-Base-7B_top_k"
sae_root = "train_dlm_sae/saes_mask_Dream-org_Dream-v0-Base-7B_top_k"
layers_of_interest = [5, 14, 27]

# ── Helpers ──────────────────────────────────────────────────────────────
def extract_layer(s):
    m = re.search(r"layer_(\d+)", s)
    return int(m.group(1)) if m else None

def extract_trainer(s):
    m = re.search(r"trainer_(\d+)", s)
    return int(m.group(1)) if m else None


# ── Load sparsity (k) from config.json ───────────────────────────────────
sae_info = {}

config_paths = glob.glob(f"{sae_root}/**/trainer_*/config.json", recursive=True)

for path in config_paths:
    with open(path) as f:
        cfg = json.load(f)

    layer = cfg["trainer"]["layer"]
    k = cfg["trainer"]["k"]

    trainer_idx = extract_trainer(path)

    sae_info[(layer, trainer_idx)] = k


# ── Load eval results from directory structure ───────────────────────────
result_paths = glob.glob(f"{sae_root}/**/trainer_*/delta*.json", recursive=True)

mask_rows = []
unmask_rows = []

for path in result_paths:
    parts = path.split(os.sep)

    # Expect: ... / resid_post_layer_X / trainer_Y / delta_eval_*.json
    try:
        layer_str = parts[-3]
        trainer_str = parts[-2]
    except IndexError:
        continue

    layer = extract_layer(layer_str)
    trainer = extract_trainer(trainer_str)

    if layer not in layers_of_interest:
        continue

    with open(path) as f:
        data = json.load(f)

    k = sae_info.get((layer, trainer))
    if k is None:
        continue

    try:
        mask_rows.append({
            "layer": layer,
            "trainer": trainer,
            "k": k,

            "delta_no_sae_noise": data["delta_no_sae_noise(mask)"],
            "delta_yes_sae_noise": data["delta_yes_sae_noise(mask)"],

            # derived metric (your key hypothesis)
            "denoising_effect": data["delta_no_sae_noise(mask)"] - data["delta_yes_sae_noise(mask)"],
        })
    except:
        unmask_rows.append({
            "layer": layer,
            "trainer": trainer,
            "k": k,

            "delta_no_sae_noise": data["delta_no_sae_noise(unmask)"],
            "delta_yes_sae_noise": data["delta_yes_sae_noise(unmask)"],

            # derived metric (your key hypothesis)
            "denoising_effect": data["delta_no_sae_noise(unmask)"] - data["delta_yes_sae_noise(unmask)"],
        })


# ── Organize per-layer ───────────────────────────────────────────────────
metrics = [
    ("delta_no_sae_noise", "Noise only"),
    ("delta_yes_sae_noise", "SAE + Noise"),
    ("denoising_effect", "Denoising Effect (NoSAE - SAE)"),
]

data = {layer: {m[0]: [] for m in metrics} for layer in layers_of_interest}

for r in mask_rows:
    layer = r["layer"]
    for key, _ in metrics:
        data[layer][key].append((r["k"], r[key]))

# sort by k
for layer in layers_of_interest:
    for key, _ in metrics:
        data[layer][key].sort()


# ── Plot ─────────────────────────────────────────────────────────────────
colors  = {5: "#E41A1C", 14: "#377EB8", 27: "#4DAF4A"}
markers = {5: "o",       14: "s",       27: "^"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

titles = [m[1] for m in metrics]

ymaxes = []
for col, (metric_key, title) in enumerate(metrics):
    ax = axes[col]

    for layer in layers_of_interest:
        pairs = data[layer][metric_key]
    
        if not pairs:
            continue
    
        k_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
    
        ax.plot(
            k_vals,
            y_vals,
            color=colors[layer],
            marker=markers[layer],
            linewidth=2,
            markersize=7,
            label=f"Layer {layer}",
        )
    
    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("L₀ (active features per token)")
    ax.set_ylabel("ΔCE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _, ymax = ax.get_ylim()
    ymaxes.append(ymax)

ymax = max(ymaxes) * 1.1
for ax in axes:
    lwr, _ = ax.get_ylim()
    ax.set_ylim((lwr, ymax))

plt.suptitle(
    "Noise Robustness vs Sparsity — Dream-7B",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

save_path = "dream_smoltalk_noise_denoising.pdf"
plt.savefig(save_path, dpi=150, bbox_inches="tight", format='pdf')
print('finished')
# plt.show()
