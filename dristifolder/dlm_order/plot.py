#!/usr/bin/env python3
"""
Plot DLM order analysis results (Step1 / Step2 / Step3).

Auto-detect available algorithms by scanning under run_dir/metrics:

Step1 CSVs:
  step1_mask_stability_<alg>.csv

Step2 CSVs:
  step2_post_drift_<alg>.csv

Step3 JSONs:
  step3_order_sensitive_<ref>_vs_<other>.json

Outputs (to out_dir):
- <prefix>_step1_mask_stability.(png|pdf)
- <prefix>_step2_post_drift.(png|pdf)
- <prefix>_step3_order_sensitive_<ref>_vs_<other>.(png|pdf)  (one per JSON)

Important in this version:
- No figure-level suptitle.
- No revise_count footer text (avoids overlap).
- Heatmaps use per-algorithm x-range (so different pos scales do NOT create empty columns).
- Mean curves use a shared x-axis based on normalized generation progress in [0, 1]
  (binned into N bins), so different pos_rel_gen scales remain comparable.
- Step1 removes the extra "Top-1 value delta" line plot.
- Titles / axis labels are paper-friendly (no underscores; Title Case for subplot titles/labels).
- Legend algorithm names are Title Case and underscores removed.
- Specific line colors for Origin / Entropy / Topk Margin in Step1 & Step2 mean curves.

Paper readability upgrades in this version:
- Step1 and Step2 figure aspect ratio forced to 2:1 (width:height).
- Larger fonts for subplot titles, axis labels, and tick labels (paper-friendly).
- All subplot titles are bold.
- Axis labels are bold and larger.
- Tick density is automatically reduced when needed to prevent overlap/clutter.
- Colorbar tick labels are enlarged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Naming / Styling
# ----------------------------

ALG_LINE_COLORS = {
    "origin": "#25307A",
    "entropy": "#FF7F0D",
    "topk_margin": "#2FA12F",
}
ALG_LINEWIDTHS = {
    "origin": 3.0,
    "entropy": 3.0,
    "topk_margin": 3.0,
}
DEFAULT_LINEWIDTH = 2.6

TIME_STAMP_RE = re.compile(r"_\d{8}_\d{6}")


# ----------------------------
# Paper-friendly global style
# ----------------------------

PAPER_STYLE = {
    # Base fonts
    "base_font": 14,
    "title_size": 16,
    "label_size": 16,
    "tick_size": 13,
    "legend_size": 13,
    # Layout and lines
    "grid_alpha": 0.28,
    "axes_linewidth": 1.2,
    "tick_major_width": 1.1,
    "tick_major_size": 4.0,
    "tick_pad": 3,
    # Saving
    "save_dpi": 300,
    # Heatmap tick density
    "heatmap_max_xticks": 10,
    "heatmap_max_yticks": 10,
}


def _apply_paper_style() -> None:
    """Apply paper-friendly matplotlib rcParams."""
    plt.rcParams.update(
        {
            "font.size": PAPER_STYLE["base_font"],
            "axes.titlesize": PAPER_STYLE["title_size"],
            "axes.titleweight": "bold",
            "axes.labelsize": PAPER_STYLE["label_size"],
            "axes.labelweight": "bold",
            "xtick.labelsize": PAPER_STYLE["tick_size"],
            "ytick.labelsize": PAPER_STYLE["tick_size"],
            "legend.fontsize": PAPER_STYLE["legend_size"],
            "axes.linewidth": PAPER_STYLE["axes_linewidth"],
            "xtick.major.width": PAPER_STYLE["tick_major_width"],
            "ytick.major.width": PAPER_STYLE["tick_major_width"],
            "xtick.major.size": PAPER_STYLE["tick_major_size"],
            "ytick.major.size": PAPER_STYLE["tick_major_size"],
        }
    )


def _strip_timestamp(s: str) -> str:
    # Remove patterns like "_20260120_231249" anywhere in the string
    return TIME_STAMP_RE.sub("", s)


def _title_case_words(s: str) -> str:
    # Title Case with underscore removal; keep hyphenated tokens as-is except first char.
    s = s.replace("_", " ").strip()
    if not s:
        return s
    out = []
    for tok in s.split():
        if "-" in tok:
            parts = tok.split("-")
            parts = [p[:1].upper() + p[1:] if p else p for p in parts]
            out.append("-".join(parts))
        else:
            out.append(tok[:1].upper() + tok[1:])
    return " ".join(out)


def _display_alg_name(alg: str) -> str:
    return _title_case_words(alg)


# ----------------------------
# Utilities
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # Convert numeric-like columns
    for c in df.columns:
        if c == "alg":
            continue
        try:
            converted = pd.to_numeric(df[c], errors="coerce")
            nan_ratio = float(converted.isna().mean()) if len(converted) > 0 else 1.0
            if nan_ratio < 0.95:
                df[c] = converted
        except Exception:
            pass

    return df


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_pos_col(df: pd.DataFrame) -> str:
    if "pos_rel_gen" in df.columns:
        return "pos_rel_gen"
    if "pos_abs" in df.columns:
        return "pos_abs"
    for c in df.columns:
        if "pos" in c.lower():
            return c
    raise ValueError("No position column found in dataframe.")


def _infer_layer_col(df: pd.DataFrame) -> str:
    if "layer" in df.columns:
        return "layer"
    for c in df.columns:
        if "layer" in c.lower():
            return c
    raise ValueError("No layer column found in dataframe.")


def _coerce_int_sorted(values: List[Any]) -> List[int]:
    out = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            pass
    return sorted(list(set(out)))


def _nice_alg_order(algs: List[str]) -> List[str]:
    # Put common baseline first, then others
    pri = {"origin": 0, "entropy": 1, "maskgit_plus": 2, "topk_margin": 3}
    return sorted(algs, key=lambda a: (pri.get(a, 999), a))


def _restrict_pos_range_single(
    df: pd.DataFrame,
    pos_col: str,
    metric_col: str,
    min_nonzero_points: int = 5,
) -> Tuple[int, int]:
    """
    Restrict x-range for a single algorithm to positions where metric is informative
    (non-zero, non-NaN). If too few informative points, fall back to full min/max of pos.
    """
    if df.empty or pos_col not in df.columns:
        return 0, 0

    if metric_col in df.columns:
        m = pd.to_numeric(df[metric_col], errors="coerce")
        p = pd.to_numeric(df[pos_col], errors="coerce")
        mask = (~m.isna()) & (~p.isna()) & (m != 0)
        p_valid = p[mask].to_numpy()
    else:
        p_valid = np.array([], dtype=float)

    if p_valid.size >= min_nonzero_points:
        return int(np.nanmin(p_valid)), int(np.nanmax(p_valid))

    # Fallback to full min/max of pos
    all_pos = pd.to_numeric(df[pos_col], errors="coerce").to_numpy()
    if all_pos.size == 0:
        return 0, 0
    return int(np.nanmin(all_pos)), int(np.nanmax(all_pos))


def _collect_layers_union(dfs: Dict[str, pd.DataFrame], layer_col: str) -> List[int]:
    layers = set()
    for _, df in dfs.items():
        if layer_col in df.columns:
            vals = df[layer_col].dropna().unique().tolist()
            for v in vals:
                try:
                    layers.add(int(v))
                except Exception:
                    pass
    return sorted(layers)


def _pivot_matrix_fixed(
    df: pd.DataFrame,
    layer_col: str,
    pos_col: str,
    value_col: str,
    layers: List[int],
    positions: List[int],
) -> np.ndarray:
    mat = np.full((len(layers), len(positions)), np.nan, dtype=float)
    if df.empty or value_col not in df.columns:
        return mat

    sub = df[[layer_col, pos_col, value_col]].copy()
    sub[layer_col] = pd.to_numeric(sub[layer_col], errors="coerce")
    sub[pos_col] = pd.to_numeric(sub[pos_col], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return mat

    grp = sub.groupby([layer_col, pos_col], as_index=False)[value_col].mean()

    layer_to_i = {l: i for i, l in enumerate(layers)}
    pos_to_j = {p: j for j, p in enumerate(positions)}

    for _, r in grp.iterrows():
        li = layer_to_i.get(int(r[layer_col]))
        pj = pos_to_j.get(int(r[pos_col]))
        if li is None or pj is None:
            continue
        mat[li, pj] = float(r[value_col])
    return mat


def _choose_tick_indices(n: int, max_ticks: int) -> List[int]:
    """Choose tick indices with an upper bound to avoid overlap."""
    if n <= 0:
        return []
    if n <= max_ticks:
        return list(range(n))
    step = int(np.ceil(n / float(max_ticks)))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def _imshow_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    layers: List[int],
    positions: List[int],
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.AxesImage:
    im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=6)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=6)

    # X ticks (reduce density for large fonts)
    x_idx = _choose_tick_indices(len(positions), PAPER_STYLE["heatmap_max_xticks"])
    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(positions[i]) for i in x_idx], rotation=0)

    # Y ticks (reduce density for large fonts)
    y_idx = _choose_tick_indices(len(layers), PAPER_STYLE["heatmap_max_yticks"])
    ax.set_yticks(y_idx)
    ax.set_yticklabels([str(layers[i]) for i in y_idx])

    ax.tick_params(axis="both", which="major", pad=PAPER_STYLE["tick_pad"])
    return im


def _line_style_for_index(i: int) -> str:
    linestyles = ["-", "--", "-.", ":"]
    return linestyles[i % len(linestyles)]


def _plot_mean_curve_multi_progress(
    ax: plt.Axes,
    dfs: Dict[str, pd.DataFrame],
    pos_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    n_bins: int = 60,
    min_points_per_bin: int = 1,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Plot mean(value_col) vs normalized progress in [0, 1], averaged over layers,
    for ALL algorithms on a shared x-axis.

    For each algorithm:
      progress = (pos - min_pos) / (max_pos - min_pos)
      then bin progress into n_bins and plot mean(value) per bin.
    """
    algs = _nice_alg_order(list(dfs.keys()))
    any_plotted = False

    for i, alg in enumerate(algs):
        df = dfs[alg]
        if pos_col not in df.columns or value_col not in df.columns:
            continue

        d = df[[pos_col, value_col]].copy()
        d[pos_col] = pd.to_numeric(d[pos_col], errors="coerce")
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        d = d.dropna()
        if d.empty:
            continue

        p = d[pos_col].to_numpy(dtype=float)
        v = d[value_col].to_numpy(dtype=float)

        p_min = float(np.nanmin(p))
        p_max = float(np.nanmax(p))
        denom = (p_max - p_min)

        if not np.isfinite(p_min) or not np.isfinite(p_max):
            continue

        if denom <= 0:
            prog = np.zeros_like(p, dtype=float)
        else:
            prog = (p - p_min) / denom
            prog = np.clip(prog, 0.0, 1.0)

        bins = np.floor(prog * n_bins).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)

        g = (
            pd.DataFrame({"bin": bins, "val": v})
            .groupby("bin", as_index=False)["val"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "mean", "count": "count"})
        )

        g = g[g["count"] >= int(min_points_per_bin)]
        if g.empty:
            continue

        x = (g["bin"].to_numpy(dtype=float) + 0.5) / float(n_bins)
        y = g["mean"].to_numpy(dtype=float)

        ls = _line_style_for_index(i)
        label = _display_alg_name(alg)

        color = None
        if color_map is not None:
            color = color_map.get(alg, None)

        lw = ALG_LINEWIDTHS.get(alg, DEFAULT_LINEWIDTH)

        ax.plot(
            x,
            y,
            label=label,
            linestyle=ls,
            linewidth=lw,
            alpha=0.98,
            color=color,
        )
        any_plotted = True

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Generation Progress (0–1)", fontweight="bold", labelpad=6)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=6)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=PAPER_STYLE["grid_alpha"])
    ax.tick_params(axis="both", which="major", pad=PAPER_STYLE["tick_pad"])

    if any_plotted:
        ncol = min(3, max(1, len(algs)))
        ax.legend(frameon=False, ncol=ncol, loc="best")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=int(PAPER_STYLE["save_dpi"]), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")


def _figure_size_2_to_1(n_alg: int, min_height: float = 10.0, per_alg_width: float = 6.0) -> Tuple[float, float]:
    """
    Choose (width, height) such that width:height = 2:1, while keeping enough width per algorithm.
    """
    width = max(2.0 * float(min_height), float(per_alg_width) * max(1, int(n_alg)))
    height = width / 2.0
    return width, height


# ----------------------------
# Step 1
# ----------------------------

def plot_step1(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
    prefix: str,
    progress_bins: int = 60,
    progress_min_points_per_bin: int = 1,
) -> None:
    """
    Step1 plots:
    - Row1: Pre-mask stability heatmaps per algorithm (per-algorithm x-range)
    - Row2: Top-1 feature lock rate heatmaps per algorithm (per-algorithm x-range)
    - Row3: Mean pre-mask similarity during generation (shared normalized progress axis)
    """
    if not dfs:
        raise ValueError("Step1: no CSV dataframes provided.")

    _apply_paper_style()

    any_df = next(iter(dfs.values()))
    layer_col = _infer_layer_col(any_df)
    pos_col = _infer_pos_col(any_df)

    sim_col = _pick_col(any_df, ["pre_sim_mean", "pre_sim", "sim_mean", "sim"])
    lock_col = _pick_col(any_df, ["pre_top1_id_lock_ratio", "top1_id_lock_ratio", "id_lock_ratio"])

    if sim_col is None:
        raise ValueError("Step1: cannot find a similarity column (e.g., pre_sim_mean).")
    if lock_col is None:
        raise ValueError("Step1: cannot find a lock-ratio column (e.g., pre_top1_id_lock_ratio).")

    algs = _nice_alg_order(list(dfs.keys()))
    n_alg = len(algs)

    layers = _collect_layers_union(dfs, layer_col)

    # Per-algorithm positions to avoid empty columns due to different pos ranges
    positions_by_alg: Dict[str, List[int]] = {}
    for alg in algs:
        df = dfs[alg]
        pos_min, pos_max = _restrict_pos_range_single(df, pos_col, sim_col)
        positions_by_alg[alg] = list(range(pos_min, pos_max + 1))

    mats_sim: Dict[str, np.ndarray] = {}
    mats_lock: Dict[str, np.ndarray] = {}

    for alg in algs:
        df = dfs[alg]
        positions = positions_by_alg[alg]
        mats_sim[alg] = _pivot_matrix_fixed(df, layer_col, pos_col, sim_col, layers, positions)
        mats_lock[alg] = _pivot_matrix_fixed(df, layer_col, pos_col, lock_col, layers, positions)

    all_sim = np.concatenate([m.flatten() for m in mats_sim.values()]) if mats_sim else np.array([0.0])
    if np.isfinite(np.nanmin(all_sim)) and np.isfinite(np.nanmax(all_sim)):
        vmin_sim = float(np.nanmin(all_sim))
        vmax_sim = float(np.nanmax(all_sim))
    else:
        vmin_sim, vmax_sim = 0.0, 1.0

    fig_w, fig_h = _figure_size_2_to_1(n_alg=n_alg, min_height=10.0, per_alg_width=6.2)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.25, wspace=0.02, hspace=0.02)

    gs = fig.add_gridspec(3, n_alg, height_ratios=[1.12, 1.12, 0.95])

    # Row 1: stability heatmaps
    ax_row1 = []
    for i, alg in enumerate(algs):
        ax = fig.add_subplot(gs[0, i])
        positions = positions_by_alg[alg]
        title = f"{_display_alg_name(alg)}: Pre-mask Stability"
        im = _imshow_heatmap(
            ax,
            mats_sim[alg],
            layers,
            positions,
            title=title,
            xlabel="Generation Steps",
            ylabel="Layer" if i == 0 else "",
            vmin=vmin_sim,
            vmax=vmax_sim,
        )
        ax_row1.append((ax, im))

    cbar1 = fig.colorbar(ax_row1[-1][1], ax=[a for a, _ in ax_row1], fraction=0.045, pad=0.02)
    cbar1.ax.tick_params(labelsize=PAPER_STYLE["tick_size"])

    # Row 2: lock rate heatmaps
    ax_row2 = []
    for i, alg in enumerate(algs):
        ax = fig.add_subplot(gs[1, i])
        positions = positions_by_alg[alg]
        title = f"{_display_alg_name(alg)}: Top-1 Feature Lock Rate"
        im = _imshow_heatmap(
            ax,
            mats_lock[alg],
            layers,
            positions,
            title=title,
            xlabel="Generation Steps",
            ylabel="Layer" if i == 0 else "",
            vmin=0.0,
            vmax=1.0,
        )
        ax_row2.append((ax, im))

    cbar2 = fig.colorbar(ax_row2[-1][1], ax=[a for a, _ in ax_row2], fraction=0.045, pad=0.02)
    cbar2.ax.tick_params(labelsize=PAPER_STYLE["tick_size"])

    # Row 3: mean similarity curve
    ax3 = fig.add_subplot(gs[2, :])
    _plot_mean_curve_multi_progress(
        ax3,
        dfs,
        pos_col,
        sim_col,
        title="Mean Pre-mask Similarity During Generation",
        ylabel="Similarity",
        n_bins=int(progress_bins),
        min_points_per_bin=int(progress_min_points_per_bin),
        color_map=ALG_LINE_COLORS,
    )

    _save_figure(fig, out_dir, f"{prefix}_step1_mask_stability")
    plt.close(fig)


# ----------------------------
# Step 2
# ----------------------------

def plot_step2(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
    prefix: str,
    progress_bins: int = 60,
    progress_min_points_per_bin: int = 1,
) -> None:
    """
    Step2 visualizes ONLY:
    - Top-K drift (drift_mean / post_drift_mean / 1 - post_sim_mean)
    - Top-1 flip count (post_top1_flip_count)
    - Mean post-decode Top-K drift during generation (shared normalized progress axis)
    """
    if not dfs:
        raise ValueError("Step2: no CSV dataframes provided.")

    _apply_paper_style()

    any_df = next(iter(dfs.values()))
    layer_col = _infer_layer_col(any_df)
    pos_col = _infer_pos_col(any_df)

    drift_col = _pick_col(any_df, ["post_drift_mean", "post_drift", "drift_mean", "drift"])
    sim_col = _pick_col(any_df, ["post_sim_mean", "post_sim", "sim_mean", "sim"])
    flip_col = _pick_col(any_df, ["post_top1_flip_count", "top1_flip_count"])

    if drift_col is None and sim_col is None:
        raise ValueError("Step2: cannot find drift/sim column (e.g., drift_mean or post_sim_mean).")
    if flip_col is None:
        raise ValueError("Step2: cannot find post_top1_flip_count column.")

    # Build derived drift column per df
    dfs2: Dict[str, pd.DataFrame] = {}
    for alg, df in dfs.items():
        d = df.copy()
        if drift_col is not None and drift_col in d.columns:
            d["__drift__"] = pd.to_numeric(d[drift_col], errors="coerce")
        else:
            s = pd.to_numeric(d[sim_col], errors="coerce")
            d["__drift__"] = 1.0 - s
        dfs2[alg] = d

    algs = _nice_alg_order(list(dfs2.keys()))
    n_alg = len(algs)

    layers = _collect_layers_union(dfs2, layer_col)

    # Per-algorithm positions (based on drift) to avoid empty columns
    positions_by_alg: Dict[str, List[int]] = {}
    for alg in algs:
        df = dfs2[alg]
        pos_min, pos_max = _restrict_pos_range_single(df, pos_col, "__drift__")
        positions_by_alg[alg] = list(range(pos_min, pos_max + 1))

    mats_drift: Dict[str, np.ndarray] = {}
    mats_flip: Dict[str, np.ndarray] = {}

    for alg in algs:
        df = dfs2[alg]
        positions = positions_by_alg[alg]
        mats_drift[alg] = _pivot_matrix_fixed(df, layer_col, pos_col, "__drift__", layers, positions)
        mats_flip[alg] = _pivot_matrix_fixed(df, layer_col, pos_col, flip_col, layers, positions)

    all_drift = np.concatenate([m.flatten() for m in mats_drift.values()]) if mats_drift else np.array([0.0])
    if np.isfinite(np.nanmin(all_drift)) and np.isfinite(np.nanmax(all_drift)):
        vmin_d = float(np.nanmin(all_drift))
        vmax_d = float(np.nanmax(all_drift))
    else:
        vmin_d, vmax_d = 0.0, 1.0

    all_flip = np.concatenate([m.flatten() for m in mats_flip.values()]) if mats_flip else np.array([0.0])
    vmax_f = float(np.nanmax(all_flip)) if np.isfinite(np.nanmax(all_flip)) else 1.0
    vmax_f = max(vmax_f, 1.0)

    fig_w, fig_h = _figure_size_2_to_1(n_alg=n_alg, min_height=10.0, per_alg_width=6.2)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.25, wspace=0.02, hspace=0.02)

    gs = fig.add_gridspec(3, n_alg, height_ratios=[1.12, 1.12, 0.95])

    # Row 1: drift heatmaps
    ax_row1 = []
    for i, alg in enumerate(algs):
        ax = fig.add_subplot(gs[0, i])
        positions = positions_by_alg[alg]
        title = f"{_display_alg_name(alg)}: Post-decode Top-K Features Drift"
        im = _imshow_heatmap(
            ax,
            mats_drift[alg],
            layers,
            positions,
            title=title,
            xlabel="Generation Steps",
            ylabel="Layer" if i == 0 else "",
            vmin=vmin_d,
            vmax=vmax_d,
        )
        ax_row1.append((ax, im))

    cbar1 = fig.colorbar(ax_row1[-1][1], ax=[a for a, _ in ax_row1], fraction=0.045, pad=0.02)
    cbar1.ax.tick_params(labelsize=PAPER_STYLE["tick_size"])

    # Row 2: flip heatmaps
    ax_row2 = []
    for i, alg in enumerate(algs):
        ax = fig.add_subplot(gs[1, i])
        positions = positions_by_alg[alg]
        title = f"{_display_alg_name(alg)}: Top-1 Feature Flip Count"
        im = _imshow_heatmap(
            ax,
            mats_flip[alg],
            layers,
            positions,
            title=title,
            xlabel="Generation Steps",
            ylabel="Layer" if i == 0 else "",
            vmin=0.0,
            vmax=vmax_f,
        )
        ax_row2.append((ax, im))

    cbar2 = fig.colorbar(ax_row2[-1][1], ax=[a for a, _ in ax_row2], fraction=0.045, pad=0.02)
    cbar2.ax.tick_params(labelsize=PAPER_STYLE["tick_size"])

    # Row 3: mean drift curve
    ax3 = fig.add_subplot(gs[2, :])
    _plot_mean_curve_multi_progress(
        ax3,
        dfs2,
        pos_col,
        "__drift__",
        title="Mean Post-decode Top-K Drift During Generation",
        ylabel="Drift (1 − Similarity)",
        n_bins=int(progress_bins),
        min_points_per_bin=int(progress_min_points_per_bin),
        color_map=ALG_LINE_COLORS,
    )

    _save_figure(fig, out_dir, f"{prefix}_step2_post_drift")
    plt.close(fig)


# ----------------------------
# Step 3
# ----------------------------

def _extract_step3_top_by_layer(obj: Any) -> Optional[pd.DataFrame]:
    if not isinstance(obj, dict):
        return None
    top_by_layer = obj.get("top_by_layer", None)
    if not isinstance(top_by_layer, dict):
        return None

    rows: List[Dict[str, Any]] = []
    for lk, pairs in top_by_layer.items():
        try:
            layer = int(lk)
        except Exception:
            continue
        if not isinstance(pairs, list):
            continue
        for item in pairs:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            fid, score = item[0], item[1]
            try:
                fid_i = int(fid)
                score_f = float(score)
            except Exception:
                continue
            rows.append({"layer": layer, "feature_id": fid_i, "score": score_f, "abs_score": abs(score_f)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["layer", "abs_score"], ascending=[True, False]).reset_index(drop=True)
    return df


def plot_step3(obj: Any, out_dir: Path, stem: str, top_k: int = 25) -> None:
    _apply_paper_style()

    df = _extract_step3_top_by_layer(obj)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.25, wspace=0.03, hspace=0.03)
    gs = fig.add_gridspec(2, 2)

    ax_l1 = fig.add_subplot(gs[0, 0])
    ax_l2 = fig.add_subplot(gs[0, 1])
    ax_l3 = fig.add_subplot(gs[1, 0])
    ax_sum = fig.add_subplot(gs[1, 1])

    if df is None or df.empty:
        for ax in [ax_l1, ax_l2, ax_l3]:
            ax.axis("off")
        ax_sum.axis("off")
        ax_sum.text(
            0.5, 0.5,
            "No Parsable Records Found",
            ha="center", va="center",
            fontsize=PAPER_STYLE["title_size"],
            fontweight="bold",
        )
        _save_figure(fig, out_dir, stem)
        plt.close(fig)
        return

    ref_alg = obj.get("ref_alg", "") if isinstance(obj, dict) else ""
    other_alg = obj.get("other_alg", "") if isinstance(obj, dict) else ""
    max_steps_used = obj.get("max_steps_used", None) if isinstance(obj, dict) else None
    topn = obj.get("topn", None) if isinstance(obj, dict) else None

    layers = _coerce_int_sorted(df["layer"].unique().tolist())
    layers = layers[:3] if len(layers) > 3 else layers
    axes = [ax_l1, ax_l2, ax_l3]

    for idx, ax in enumerate(axes):
        if idx >= len(layers):
            ax.axis("off")
            continue
        L = layers[idx]
        dL = df[df["layer"] == L].copy().sort_values("abs_score", ascending=False)
        k = min(top_k, len(dL))
        dL = dL.head(k)

        labels = [f"F{int(fid)}" for fid in dL["feature_id"].tolist()]
        y = np.arange(k)
        ax.barh(y, dL["abs_score"].to_numpy())
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=PAPER_STYLE["tick_size"])
        ax.invert_yaxis()
        ax.set_title(f"Layer {L}: Top-{k} |Score|", fontweight="bold")
        ax.set_xlabel("|Score|", fontweight="bold", labelpad=6)
        ax.grid(True, alpha=PAPER_STYLE["grid_alpha"])
        ax.tick_params(axis="both", which="major", pad=PAPER_STYLE["tick_pad"])

    scores = df["score"].to_numpy()
    ax_sum.hist(scores, bins=50)
    ax_sum.set_title("All Layers: Order Sensitivity Distribution", fontweight="bold")
    ax_sum.set_xlabel("Order Sensitivity", fontweight="bold", labelpad=6)
    ax_sum.set_ylabel("Count", fontweight="bold", labelpad=6)
    ax_sum.grid(True, alpha=PAPER_STYLE["grid_alpha"])
    ax_sum.tick_params(axis="both", which="major", pad=PAPER_STYLE["tick_pad"])

    text = []
    if ref_alg or other_alg:
        text.append(f"Ref: {_display_alg_name(ref_alg)}")
        text.append(f"Other: {_display_alg_name(other_alg)}")
    if max_steps_used is not None:
        text.append(f"Max Steps Used: {max_steps_used}")
    if topn is not None:
        text.append(f"Topn Per Layer: {topn}")
    text.append(f"Num Records: {len(df)}")
    text.append(f"Mean Score: {float(np.mean(scores)):.3g}")
    text.append(f"Median Score: {float(np.median(scores)):.3g}")
    text.append(f"Max |Score|: {float(np.max(np.abs(scores))):.3g}")

    ax_sum.text(
        0.02, 0.98,
        "\n".join(text),
        transform=ax_sum.transAxes,
        ha="left", va="top",
        fontsize=PAPER_STYLE["tick_size"],
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    _save_figure(fig, out_dir, stem)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def _load_alg_csvs(metrics_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    """
    Load CSVs matching: <prefix>_<alg>.csv
    Returns: alg -> df
    """
    out: Dict[str, pd.DataFrame] = {}
    for p in sorted(metrics_dir.glob(f"{prefix}_*.csv")):
        alg = p.stem[len(prefix) + 1:]  # remove "<prefix>_"
        try:
            out[alg] = _read_csv(p)
        except Exception:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to dlm_order_* run directory.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save figures.")
    parser.add_argument("--prefix", type=str, default="", help="Filename prefix for output figures.")
    parser.add_argument("--top_k", type=int, default=25, help="Top-K features to show per layer in step3.")
    parser.add_argument(
        "--progress_bins",
        type=int,
        default=60,
        help="Number of bins for normalized progress curves (shared x-axis in [0,1]).",
    )
    parser.add_argument(
        "--progress_min_points_per_bin",
        type=int,
        default=1,
        help="Minimum number of points to keep a progress bin for curves.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics directory not found: {metrics_dir}")

    raw_prefix = args.prefix.strip() or run_dir.name
    prefix = _strip_timestamp(raw_prefix)

    # Step1
    step1_dfs = _load_alg_csvs(metrics_dir, "step1_mask_stability")
    if not step1_dfs:
        raise FileNotFoundError(f"No Step1 CSVs found under: {metrics_dir} (step1_mask_stability_*.csv)")
    plot_step1(
        step1_dfs,
        out_dir,
        prefix,
        progress_bins=int(args.progress_bins),
        progress_min_points_per_bin=int(args.progress_min_points_per_bin),
    )

    # Step2
    step2_dfs = _load_alg_csvs(metrics_dir, "step2_post_drift")
    if not step2_dfs:
        raise FileNotFoundError(f"No Step2 CSVs found under: {metrics_dir} (step2_post_drift_*.csv)")
    plot_step2(
        step2_dfs,
        out_dir,
        prefix,
        progress_bins=int(args.progress_bins),
        progress_min_points_per_bin=int(args.progress_min_points_per_bin),
    )

    # Step3
    step3_jsons = sorted(metrics_dir.glob("step3_order_sensitive_*_vs_*.json"))
    for jp in step3_jsons:
        obj3 = _read_json(jp)
        ref = obj3.get("ref_alg", "ref") if isinstance(obj3, dict) else "ref"
        other = obj3.get("other_alg", "other") if isinstance(obj3, dict) else "other"
        stem = _strip_timestamp(f"{prefix}_step3_order_sensitive_{ref}_vs_{other}")
        plot_step3(obj3, out_dir, stem, top_k=args.top_k)

    print(f"[OK] Saved figures to: {out_dir}")
    print(f"     - {prefix}_step1_mask_stability.(png|pdf)")
    print(f"     - {prefix}_step2_post_drift.(png|pdf)")
    if step3_jsons:
        print(f"     - {prefix}_step3_order_sensitive_<ref>_vs_<other>.(png|pdf)  (one per JSON)")
    else:
        print("     - (No Step3 JSONs found.)")


if __name__ == "__main__":
    main()
