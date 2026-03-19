"""
visualize_activations.py

Loads the [L x 3 x F x T] activation tensor produced by collect_activations.py
and generates two sets of plots:

  1. Heatmaps (3 x L):  features (sorted by peak time) vs diffusion timestep
  2. Histograms (3 x L): distribution of peak times across all features

Usage:
    python visualize_activations.py --bundle outputs_exp1/layer_stats_bundle.pt
    python visualize_activations.py --tensor outputs_exp1/layer_stats_Lx3xFxT.pt --layers 5 14 27
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


STAT_NAMES = [
    "All-token Average",
    "Top-M Token Average",
    "Activation Frequency",
]

STAT_SHORT = ["avg_all", "avg_topm", "freq"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAE activation statistics.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bundle",
        type=str,
        help="Path to layer_stats_bundle.pt (contains tensor + layer list).",
    )
    group.add_argument(
        "--tensor",
        type=str,
        help="Path to layer_stats_Lx3xFxT.pt (raw tensor only).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices (required when --tensor is used; ignored for --bundle).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=None,
        help="Show only the top-N most active features in heatmaps (default: all).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for heatmaps.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args: argparse.Namespace):
    """
    Returns:
        tensor: [L, 3, F, T]  float32
        layers: list of int, length L
    """
    if args.bundle:
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
        tensor = bundle["tensor"]           # [L, 3, F, T]
        layers = list(bundle["layers"])
    else:
        tensor = torch.load(args.tensor, map_location="cpu", weights_only=False)
        if args.layers is None:
            raise ValueError("--layers is required when loading a raw tensor with --tensor.")
        layers = list(args.layers)

    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape [L, 3, F, T], got {tuple(tensor.shape)}")
    if tensor.shape[0] != len(layers):
        raise ValueError(
            f"Tensor has {tensor.shape[0]} layers but {len(layers)} layer indices were provided."
        )

    return tensor.float(), layers


# ---------------------------------------------------------------------------
# Peak time computation
# ---------------------------------------------------------------------------

def compute_peak_times(activation: np.ndarray) -> np.ndarray:
    """
    Args:
        activation: [F, T]
    Returns:
        peak_times: [F]  argmax over T for each feature
    """
    return np.argmax(activation, axis=1)


def sort_features_by_peak_time(activation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort features by their peak diffusion timestep (ascending).

    Returns:
        sorted_activation: [F, T]
        sort_order:        [F]  indices that sort the original features
    """
    peak_times = compute_peak_times(activation)
    sort_order = np.argsort(peak_times, kind="stable")
    return activation[sort_order], sort_order


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _timestep_ticks(T: int, max_ticks: int = 8) -> List[int]:
    """Return a compact list of tick positions for the T axis."""
    step = max(1, T // max_ticks)
    ticks = list(range(0, T, step))
    if ticks[-1] != T - 1:
        ticks.append(T - 1)
    return ticks


def plot_heatmap(
    activation: np.ndarray,
    layer: int,
    stat_idx: int,
    out_dir: str,
    top_features: Optional[int],
    dpi: int,
    colormap: str,
) -> None:
    """
    Plot a single heatmap of features (sorted by peak time) vs diffusion timestep.

    Args:
        activation: [F, T]
    """
    F, T = activation.shape

    # Optionally restrict to the top-N features by mean activation
    if top_features is not None and top_features < F:
        mean_act = activation.mean(axis=1)          # [F]
        top_idx = np.argsort(mean_act)[-top_features:]
        activation = activation[top_idx]
        F = top_features

    sorted_act, _ = sort_features_by_peak_time(activation)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        sorted_act,
        aspect="auto",
        origin="lower",
        cmap=colormap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label=STAT_NAMES[stat_idx])

    ax.set_title(
        f"Layer {layer} — {STAT_NAMES[stat_idx]}\n"
        f"Features sorted by peak diffusion time  ({F} features shown)",
        fontsize=11,
    )
    ax.set_xlabel("Diffusion Timestep")
    ax.set_ylabel("Feature (sorted by peak time ↑)")

    ticks = _timestep_ticks(T)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))

    # Annotate y-axis with approximate peak-time thirds
    thirds = [0, F // 3, 2 * F // 3, F - 1]
    ax.set_yticks(thirds)
    ax.set_yticklabels([f"{v}" for v in thirds])

    plt.tight_layout()
    fname = os.path.join(out_dir, f"heatmap_L{layer}_{STAT_SHORT[stat_idx]}.png")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_peak_time_histogram(
    activation: np.ndarray,
    layer: int,
    stat_idx: int,
    out_dir: str,
    dpi: int,
) -> None:
    """
    Plot a histogram of peak times across all features.

    Args:
        activation: [F, T]
    """
    F, T = activation.shape
    peak_times = compute_peak_times(activation)   # [F]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(peak_times, bins=T, range=(-0.5, T - 0.5), color="steelblue", edgecolor="white", linewidth=0.4)

    ax.set_title(
        f"Layer {layer} — {STAT_NAMES[stat_idx]}\n"
        f"Peak Time Distribution ({F} features)",
        fontsize=11,
    )
    ax.set_xlabel("Peak Diffusion Timestep  (argmax over T)")
    ax.set_ylabel("Number of Features")

    ticks = _timestep_ticks(T)
    ax.set_xticks(ticks)

    # Draw vertical lines at the diffusion stage boundaries (thirds)
    for boundary in [T // 3, 2 * T // 3]:
        ax.axvline(boundary, color="tomato", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.text(T // 6,       ax.get_ylim()[1] * 0.92, "Early",  ha="center", color="tomato", fontsize=9)
    ax.text(T // 2,       ax.get_ylim()[1] * 0.92, "Middle", ha="center", color="tomato", fontsize=9)
    ax.text(5 * T // 6,   ax.get_ylim()[1] * 0.92, "Late",   ha="center", color="tomato", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"hist_peak_L{layer}_{STAT_SHORT[stat_idx]}.png")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading activation tensor...")
    tensor, layers = load_data(args)
    L, num_stats, F, T = tensor.shape
    print(f"Tensor shape: {tuple(tensor.shape)}  (L={L}, stats=3, F={F}, T={T})")
    print(f"Layers: {layers}")

    # ------------------------------------------------------------------
    # 1. Heatmaps  (3 x L)
    # ------------------------------------------------------------------
    print("\nGenerating heatmaps...")
    for i, layer in enumerate(layers):
        for s in range(num_stats):
            activation = tensor[i, s].numpy()   # [F, T]
            plot_heatmap(
                activation=activation,
                layer=layer,
                stat_idx=s,
                out_dir=args.out_dir,
                top_features=args.top_features,
                dpi=args.dpi,
                colormap=args.colormap,
            )

    # ------------------------------------------------------------------
    # 2. Peak time histograms  (3 x L)
    # ------------------------------------------------------------------
    print("\nGenerating peak time histograms...")
    for i, layer in enumerate(layers):
        for s in range(num_stats):
            activation = tensor[i, s].numpy()   # [F, T]
            plot_peak_time_histogram(
                activation=activation,
                layer=layer,
                stat_idx=s,
                out_dir=args.out_dir,
                dpi=args.dpi,
            )

    total = 2 * L * num_stats
    print(f"\nDone. {total} plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()