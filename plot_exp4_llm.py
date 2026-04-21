import os
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
ROOT = "qwen-topk-saes"  # top-level directory
SAVE_DIR = "llm_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def extract_layer(layer_str):
    """
    Extract integer layer index from strings like:
    'resid_post_layer_1'
    """
    m = re.search(r"layer_(\d+)", layer_str)
    return int(m.group(1)) if m else None


def extract_trainer(trainer_str):
    """
    Extract trainer index from 'trainer_0'
    """
    m = re.search(r"trainer_(\d+)", trainer_str)
    return int(m.group(1)) if m else None


# ----------------------------
# Load all results
# ----------------------------
rows = []

for path in Path(ROOT).rglob("delta_eval_ar.json"):
    parts = path.parts

    # Expect structure:
    # ... / resid_post_layer_X / trainer_Y / delta_eval_ar.json
    try:
        layer_str = parts[-3]
        trainer_str = parts[-2]
    except IndexError:
        continue

    layer = extract_layer(layer_str)
    trainer = extract_trainer(trainer_str)

    with open(path, "r") as f:
        data = json.load(f)

    rows.append({
        "layer": layer,
        "trainer": trainer,

        "clean_no_sae": data["clean_no_sae"],
        "clean_yes_sae": data["clean_yes_sae"],

        "noise_no_sae": data["noise_no_sae"],
        "noise_yes_sae": data["noise_yes_sae"],

        "delta_noise_no_sae": data["delta_noise_no_sae"],
        "delta_noise_yes_sae": data["delta_noise_yes_sae"],
    })

df = pd.DataFrame(rows)
df = df.sort_values(["layer", "trainer"])

print(df.head())

# Plot 1: Noise sensitivity v layer (main result)
plt.figure()

for trainer in sorted(df["trainer"].unique()):
    sub = df[df["trainer"] == trainer]

    plt.plot(
        sub["layer"],
        sub["delta_noise_no_sae"],
        linestyle="--",
        marker="o",
        label=f"no_sae (trainer {trainer})"
    )

    plt.plot(
        sub["layer"],
        sub["delta_noise_yes_sae"],
        linestyle="-",
        marker="o",
        label=f"yes_sae (trainer {trainer})"
    )

plt.xlabel("Layer")
plt.ylabel("Δ CE Loss (noise - clean)")
plt.title("Noise Sensitivity With vs Without SAE")
plt.legend()
plt.grid()

plt.savefig(os.path.join(SAVE_DIR, "noise_sensitivity_vs_layer.pdf"), format='pdf')

# Plo2 2: Direct comparison (gap = denoising effect)
df["denoising_effect"] = df["delta_noise_no_sae"] - df["delta_noise_yes_sae"]

plt.figure()

for trainer in sorted(df["trainer"].unique()):
    sub = df[df["trainer"] == trainer]

    plt.plot(
        sub["layer"],
        sub["denoising_effect"],
        marker="o",
        label=f"trainer {trainer}"
    )

plt.axhline(0)
plt.xlabel("Layer")
plt.ylabel("Denoising Effect (positive = SAE helps)")
plt.title("Does SAE Reduce Noise Sensitivity?")
plt.legend()
plt.grid()

plt.savefig(os.path.join(SAVE_DIR, "denoising_effect.pdf"), format='pdf')

# Plot 3: Absolute losses (sanity check)
plt.figure()

sub = df[df["trainer"] == 0]  # start with trainer 0

plt.plot(sub["layer"], sub["clean_no_sae"], label="clean no_sae", linestyle="--")
plt.plot(sub["layer"], sub["clean_yes_sae"], label="clean yes_sae")

plt.plot(sub["layer"], sub["noise_no_sae"], label="noise no_sae", linestyle="--")
plt.plot(sub["layer"], sub["noise_yes_sae"], label="noise yes_sae")

plt.xlabel("Layer")
plt.ylabel("Cross Entropy Loss")
plt.title("Absolute Losses (Trainer 0)")
plt.legend()
plt.grid()

plt.savefig(os.path.join(SAVE_DIR, "absolute_losses_trainer0.pdf"), format='pdf')

# Plot 4: Heatmap ove (layer, trainer)
pivot = df.pivot(index="trainer", columns="layer", values="denoising_effect")

plt.figure()
plt.imshow(pivot, aspect="auto")
plt.colorbar(label="Denoising Effect")

plt.xlabel("Layer")
plt.ylabel("Trainer (sparsity level)")
plt.title("Denoising Effect Heatmap")

plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)

plt.savefig(os.path.join(SAVE_DIR, "heatmap.pdf"), format='pdf')

# Plot 5: CE delta vs sparsity
## Processing
import json, glob, re

sae_root = "qwen-topk-saes/saes__Qwen_Qwen2.5-7B_top_k"

def extract_layer(s):
    m = re.search(r"layer_(\d+)", s)
    return int(m.group(1)) if m else None

def extract_trainer(s):
    m = re.search(r"trainer_(\d+)", s)
    return int(m.group(1)) if m else None

# build (layer, trainer) -> k
sae_info = {}
for path in glob.glob(f"{sae_root}/**/trainer_*/config.json", recursive=True):
    with open(path) as f:
        cfg = json.load(f)
    layer = cfg["trainer"]["layer"]
    k = cfg["trainer"]["k"]
    trainer = extract_trainer(path)
    sae_info[(layer, trainer)] = k

# attach k to df
df["k"] = df.apply(lambda r: sae_info.get((r["layer"], r["trainer"])), axis=1)
df = df.dropna(subset=["k"])

## Plotting
import itertools

layers_of_interest = sorted(df["layer"].unique())

# compute denoising effect
df["denoising_effect"] = df["delta_noise_no_sae"] - df["delta_noise_yes_sae"]

metrics = [
    ("delta_noise_no_sae", "Noise only"),
    ("delta_noise_yes_sae", "SAE + Noise"),
    ("denoising_effect", "Denoising Effect (NoSAE - SAE)"),
]

cmap = plt.get_cmap("tab10")  # supports up to 10 distinct colors
colors = {layer: cmap(i % 10) for i, layer in enumerate(layers_of_interest)}

marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]
markers = {layer: marker_list[i % len(marker_list)] for i, layer in enumerate(layers_of_interest)}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# lock y-range explicitly
ymin = df[["delta_noise_no_sae", "delta_noise_yes_sae", "denoising_effect"]].min().min()
ymax = df[["delta_noise_no_sae", "delta_noise_yes_sae", "denoising_effect"]].max().max()
buffered_ymax = ymax * 1.1

for ax in axes:
    ax.set_ylim(ymin, buffered_ymax)
#

for col, (metric_key, title) in enumerate(metrics):
    ax = axes[col]

    for layer in layers_of_interest:
        sub = df[df["layer"] == layer].sort_values("k")

        if len(sub) == 0:
            continue

        ax.plot(
            sub["k"],
            sub[metric_key],
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

plt.suptitle(
    "LLM Noise Robustness vs Sparsity",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

save_path = "plot5_llm_sparsity.pdf"
plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=150, bbox_inches='tight', format='pdf')

print(f"Saved to: {SAVE_DIR}")
