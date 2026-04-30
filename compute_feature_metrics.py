from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


TOKEN_TYPE_TO_INDEX = {"unmasked": 0, "masked": 1}

DEFAULT_REFERENCE_ROWS = [
    (1, "early_masked", "masked", 2788),
    (1, "late_unmasked", "unmasked", 2280),
    (1, "bridge", "masked", 3316),
    (1, "bridge", "unmasked", 3316),
    (10, "early_masked", "masked", 4730),
    (10, "late_unmasked", "unmasked", 3806),
    (10, "bridge", "masked", 12343),
    (10, "bridge", "unmasked", 12343),
    (23, "early_masked", "masked", 6193),
    (23, "late_unmasked", "unmasked", 1311),
    (23, "bridge", "masked", 7116),
    (23, "bridge", "unmasked", 7116),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute temporal feature selection metrics from a layer statistics bundle."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to layer_stats_bundle.pt with shape [token_type, layer, stat, feature, timestep].",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("feature_metric_outputs"),
        help="Directory where CSV outputs are written.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 10, 23],
        help="Layers to score.",
    )
    parser.add_argument(
        "--stat_idx",
        type=int,
        default=1,
        help="Activation statistic index. Default 1 is top_10_token_average.",
    )
    parser.add_argument(
        "--top_k_per_type",
        type=int,
        default=5,
        help="Number of candidates to keep per layer and candidate type.",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="default",
        choices=["default", "gap_only", "corr_only", "max_gap", "freq_aware", "zscore_gap"],
        help="Scoring formula used for early, bridge, and late candidates.",
    )
    parser.add_argument(
        "--save_all_scores",
        action="store_true",
        help="Also save the full per-feature metric table.",
    )
    return parser.parse_args()


def load_bundle(path: Path) -> tuple[torch.Tensor, dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "tensor" not in bundle:
        raise ValueError("Expected a bundle dictionary containing a tensor field.")

    agg = bundle["tensor"].float()
    if agg.ndim != 5:
        raise ValueError(
            f"Expected tensor shape [token_type, layer, stat, feature, timestep], got {tuple(agg.shape)}."
        )
    if agg.shape[0] < 2:
        raise ValueError("Expected token_type dimension with unmasked and masked entries.")

    return agg, bundle


def pearson_corr_with_time(trace: torch.Tensor, token_type: int) -> float:
    """Compute Pearson correlation with time after removing invalid boundary timesteps."""
    time_steps = trace.shape[-1]

    if token_type == 0:
        y = trace[1:].float()
        x = torch.arange(1, time_steps).float()
    else:
        y = trace[:-1].float()
        x = torch.arange(0, time_steps - 1).float()

    y = y - y.mean()
    x = x - x.mean()

    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return 0.0

    return float((x @ y / denom).item())


def valid_trace(trace: torch.Tensor, token_type: int) -> torch.Tensor:
    """Drop t=0 for unmasked traces and the final timestep for masked traces."""
    if token_type == 0:
        return trace[1:].float()
    return trace[:-1].float()


def early_late_stats(trace: torch.Tensor, token_type: int) -> tuple[float, float, float]:
    """Return early mean, late mean, and maximum over valid timesteps."""
    valid = valid_trace(trace, token_type)
    window = max(1, valid.shape[0] // 3)

    early = valid[:window].mean().item()
    late = valid[-window:].mean().item()
    max_val = valid.max().item()

    return early, late, max_val


def mean_valid(trace: torch.Tensor, token_type: int) -> float:
    return valid_trace(trace, token_type).mean().item()


def compute_layer_feature_table(
    agg: torch.Tensor,
    layer_to_idx: dict[int, int],
    layer: int,
    stat_idx: int,
) -> pd.DataFrame:
    li = layer_to_idx[layer]

    unmasked = agg[0, li, stat_idx]
    masked = agg[1, li, stat_idx]

    freq_available = agg.shape[2] > 2
    if freq_available:
        unmasked_freq = agg[0, li, 2]
        masked_freq = agg[1, li, 2]

    rows = []

    for feature in range(unmasked.shape[0]):
        tr_u = unmasked[feature]
        tr_m = masked[feature]

        e_u, l_u, m_u = early_late_stats(tr_u, token_type=0)
        e_m, l_m, m_m = early_late_stats(tr_m, token_type=1)

        c_u = pearson_corr_with_time(tr_u, token_type=0)
        c_m = pearson_corr_with_time(tr_m, token_type=1)

        if freq_available:
            f_u = early_late_stats(unmasked_freq[feature], token_type=0)[2]
            f_m = early_late_stats(masked_freq[feature], token_type=1)[2]
            freq_proxy = max(f_u, f_m)
        else:
            freq_proxy = 1.0

        rows.append(
            {
                "layer": layer,
                "feature": feature,
                "E_u": e_u,
                "L_u": l_u,
                "M_u": m_u,
                "C_u": c_u,
                "E_m": e_m,
                "L_m": l_m,
                "M_m": m_m,
                "C_m": c_m,
                "freq_proxy": freq_proxy,
                "masked_gap_down": max(0.0, e_m - l_m),
                "unmasked_gap_up": max(0.0, l_u - e_u),
                "masked_down_corr": max(0.0, -c_m),
                "unmasked_up_corr": max(0.0, c_u),
            }
        )

    return pd.DataFrame(rows)


def add_scores(df: pd.DataFrame, metric_mode: str = "default") -> pd.DataFrame:
    df = df.copy()

    if metric_mode == "default":
        df["early_score"] = df["M_m"] * df["masked_gap_down"] * df["masked_down_corr"]
        df["late_score"] = df["M_u"] * df["unmasked_gap_up"] * df["unmasked_up_corr"]
        df["bridge_score"] = df["E_m"] * df["L_u"] * df["masked_down_corr"] * df["unmasked_up_corr"]

    elif metric_mode == "gap_only":
        df["early_score"] = df["masked_gap_down"]
        df["late_score"] = df["unmasked_gap_up"]
        df["bridge_score"] = df["E_m"] * df["L_u"]

    elif metric_mode == "corr_only":
        df["early_score"] = df["masked_down_corr"]
        df["late_score"] = df["unmasked_up_corr"]
        df["bridge_score"] = df["masked_down_corr"] * df["unmasked_up_corr"]

    elif metric_mode == "max_gap":
        df["early_score"] = df["M_m"] * df["masked_gap_down"]
        df["late_score"] = df["M_u"] * df["unmasked_gap_up"]
        df["bridge_score"] = df["E_m"] * df["L_u"]

    elif metric_mode == "freq_aware":
        df["early_score"] = (
            df["M_m"] * df["masked_gap_down"] * df["masked_down_corr"] * df["freq_proxy"]
        )
        df["late_score"] = (
            df["M_u"] * df["unmasked_gap_up"] * df["unmasked_up_corr"] * df["freq_proxy"]
        )
        df["bridge_score"] = (
            df["E_m"]
            * df["L_u"]
            * df["masked_down_corr"]
            * df["unmasked_up_corr"]
            * df["freq_proxy"]
        )

    elif metric_mode == "zscore_gap":
        for col in ["masked_gap_down", "unmasked_gap_up", "E_m", "L_u"]:
            std = df[col].std()
            if std == 0 or np.isnan(std):
                df[col + "_z"] = 0.0
            else:
                df[col + "_z"] = (df[col] - df[col].mean()) / std

        df["early_score"] = df["masked_gap_down_z"].clip(lower=0) * df["masked_down_corr"]
        df["late_score"] = df["unmasked_gap_up_z"].clip(lower=0) * df["unmasked_up_corr"]
        df["bridge_score"] = (
            df["E_m_z"].clip(lower=0)
            * df["L_u_z"].clip(lower=0)
            * df["masked_down_corr"]
            * df["unmasked_up_corr"]
        )

    else:
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    return df


def select_top_features(
    agg: torch.Tensor,
    layer_to_idx: dict[int, int],
    layers_to_study: Iterable[int],
    metric_mode: str,
    stat_idx: int,
    top_k_per_type: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_tables = []
    selected_tables = []

    for layer in layers_to_study:
        df = compute_layer_feature_table(agg, layer_to_idx, layer, stat_idx)
        df = add_scores(df, metric_mode=metric_mode)
        all_tables.append(df)

        early = (
            df.sort_values("early_score", ascending=False)
            .head(top_k_per_type)
            .assign(candidate_type="early")
        )
        late = (
            df.sort_values("late_score", ascending=False)
            .head(top_k_per_type)
            .assign(candidate_type="late")
        )
        bridge = (
            df.sort_values("bridge_score", ascending=False)
            .head(top_k_per_type)
            .assign(candidate_type="bridge")
        )

        selected_tables.extend([early, bridge, late])

    all_df = pd.concat(all_tables, ignore_index=True)
    selected_df = pd.concat(selected_tables, ignore_index=True)

    columns = [
        "layer",
        "feature",
        "candidate_type",
        "early_score",
        "bridge_score",
        "late_score",
        "E_m",
        "L_m",
        "M_m",
        "C_m",
        "E_u",
        "L_u",
        "M_u",
        "C_u",
        "freq_proxy",
    ]
    selected_df = selected_df[columns].sort_values(["layer", "candidate_type"])

    return all_df, selected_df


def compute_reference_metrics(
    agg: torch.Tensor,
    layer_to_idx: dict[int, int],
    rows: Iterable[tuple[int, str, str, int]],
    stat_idx: int,
) -> pd.DataFrame:
    output_rows = []

    for layer, kind, token_type_name, feature in rows:
        token_type = TOKEN_TYPE_TO_INDEX[token_type_name]
        li = layer_to_idx[layer]
        trace = agg[token_type, li, stat_idx, feature]

        early, late, max_val = early_late_stats(trace, token_type)
        corr = pearson_corr_with_time(trace, token_type)

        if agg.shape[2] > 2:
            freq_mean_valid = mean_valid(agg[token_type, li, 2, feature], token_type)
        else:
            freq_mean_valid = np.nan

        output_rows.append(
            {
                "layer": layer,
                "kind": kind,
                "token_type": token_type_name,
                "feature": feature,
                "early": early,
                "late": late,
                "max": max_val,
                "corr": corr,
                "freq_mean_valid": freq_mean_valid,
            }
        )

    return pd.DataFrame(output_rows)


def validate_layers(requested_layers: Iterable[int], available_layers: list[int]) -> None:
    missing = [layer for layer in requested_layers if layer not in available_layers]
    if missing:
        raise ValueError(f"Requested layers are missing from the bundle: {missing}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    agg, meta = load_bundle(args.bundle)
    layers = list(meta["layers"])
    validate_layers(args.layers, layers)
    validate_layers([row[0] for row in DEFAULT_REFERENCE_ROWS], layers)

    layer_to_idx = {layer: i for i, layer in enumerate(layers)}

    all_feature_scores, selected_features = select_top_features(
        agg=agg,
        layer_to_idx=layer_to_idx,
        layers_to_study=args.layers,
        metric_mode=args.metric_mode,
        stat_idx=args.stat_idx,
        top_k_per_type=args.top_k_per_type,
    )
    reference_metrics = compute_reference_metrics(
        agg=agg,
        layer_to_idx=layer_to_idx,
        rows=DEFAULT_REFERENCE_ROWS,
        stat_idx=args.stat_idx,
    )

    selected_path = args.out_dir / "selected_features.csv"
    reference_path = args.out_dir / "interpreted_feature_metrics.csv"

    selected_features.to_csv(selected_path, index=False)
    reference_metrics.to_csv(reference_path, index=False)

    if args.save_all_scores:
        all_path = args.out_dir / "all_feature_scores.csv"
        all_feature_scores.to_csv(all_path, index=False)
        print(f"Saved all feature scores: {all_path}")

    print(f"Saved selected features: {selected_path}")
    print(f"Saved interpreted feature metrics: {reference_path}")
    print(reference_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
