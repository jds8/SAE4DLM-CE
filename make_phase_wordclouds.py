#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create WordCloud phase comparisons for Dream decoded distributions.

This follows the Data Science for Psychology 5.2.1 mapping:
label is a word, color is the smoothed frequency ratio, and size is
abs(log2(frequency ratio)). Since there are three phases, the script creates
pairwise comparison clouds: early vs middle, middle vs late, and early vs late.

By default this script reads decoded_events.jsonl and merges tokenizer pieces
into more readable word level units before plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from matplotlib.patches import Patch
from wordcloud import WordCloud


PHASES = ("early", "middle", "late")
PAIRWISE_PHASES = (("early", "middle"), ("middle", "late"), ("early", "late"))
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "cannot", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}


@dataclass(frozen=True)
class ComparisonRow:
    label: str
    count_a: int
    count_b: int
    ratio: float
    log2_ratio: float
    magnitude: float


@dataclass(frozen=True)
class PhaseRow:
    label: str
    early_count: int
    middle_count: int
    late_count: int
    total_count: int
    early_share: float
    middle_share: float
    late_share: float
    dominant_phase: str
    dominance: float
    dominant_over_other_ratio: float
    dominant_log2_ratio: float
    size_value: float


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_phase_counts(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {
        str(phase): {str(token): int(count) for token, count in counts.items()}
        for phase, counts in obj.items()
        if isinstance(counts, dict)
    }


def normalize_label(token: str) -> Optional[str]:
    label = token.replace(chr(0x0120), " ").replace(chr(0x2581), " ").strip()
    label = label.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    label = re.sub(r"\s+", " ", label).strip()
    if not label:
        return None
    if len(label) > 28:
        label = label[:25] + "..."
    return label


def aggregate_labels(counts: Dict[str, int]) -> Dict[str, int]:
    aggregated: Dict[str, int] = {}
    for token, count in counts.items():
        label = normalize_label(token)
        if label is None:
            continue
        aggregated[label] = aggregated.get(label, 0) + int(count)
    return aggregated


def visible_token_to_text(token: str) -> str:
    return token.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def wordish_parts(text: str, min_word_len: int) -> List[str]:
    parts = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)
    return [part.lower() for part in parts if len(part) >= min_word_len]


def load_word_counts_from_events(path: str, *, min_word_len: int) -> Dict[str, Dict[str, int]]:
    events_by_run: Dict[str, List[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            events_by_run[str(row["run_id"])].append(row)

    counts: Dict[str, Counter] = {phase: Counter() for phase in PHASES}
    boundary_re = re.compile(r"[\s.,;:!?()\[\]{}<>/\\$=+*]")

    for rows in events_by_run.values():
        rows.sort(key=lambda row: int(row["pos_rel"]))
        current = ""
        current_phases: List[str] = []

        def flush_current() -> None:
            nonlocal current, current_phases
            if not current:
                return
            for word in wordish_parts(current, min_word_len=min_word_len):
                if current_phases:
                    counts[current_phases[0]][word] += 1
            current = ""
            current_phases = []

        for row in rows:
            token = visible_token_to_text(str(row.get("token", "")))
            phase = str(row.get("phase", ""))
            if not token:
                continue
            if token[:1].isspace():
                flush_current()
            current += token
            if phase in PHASES:
                current_phases.append(phase)
            if boundary_re.search(token[-1:]):
                flush_current()
        flush_current()

    return {phase: dict(counter) for phase, counter in counts.items()}


def apply_vocab_filter(phase_counts: Dict[str, Dict[str, int]], vocab_filter: str) -> Dict[str, Dict[str, int]]:
    vocab_filter = vocab_filter.lower()
    if vocab_filter == "all":
        return phase_counts
    if vocab_filter == "stopwords":
        return {
            phase: {word: count for word, count in counts.items() if word.lower() in STOPWORDS}
            for phase, counts in phase_counts.items()
        }
    if vocab_filter == "content":
        return {
            phase: {word: count for word, count in counts.items() if word.lower() not in STOPWORDS}
            for phase, counts in phase_counts.items()
        }
    raise ValueError(f"Unknown vocab filter: {vocab_filter}")


def make_comparison_rows(
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    *,
    smoothing: float,
    min_total_count: int,
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for label in sorted(set(counts_a) | set(counts_b)):
        count_a = int(counts_a.get(label, 0))
        count_b = int(counts_b.get(label, 0))
        total = count_a + count_b
        if total < min_total_count:
            continue
        ratio = (count_a + smoothing) / (count_b + smoothing)
        log2_ratio = math.log2(ratio)
        rows.append(
            ComparisonRow(
                label=label,
                count_a=count_a,
                count_b=count_b,
                ratio=ratio,
                log2_ratio=log2_ratio,
                magnitude=abs(log2_ratio),
            )
        )
    rows.sort(key=lambda row: (row.magnitude, row.count_a + row.count_b), reverse=True)
    return rows


def write_comparison_csv(path: str, rows: Sequence[ComparisonRow], phase_a: str, phase_b: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "word",
                f"{phase_a}_count",
                f"{phase_b}_count",
                f"{phase_a}_over_{phase_b}_ratio",
                "log2_ratio",
                "log2_ratio_magnitude",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    row.count_a,
                    row.count_b,
                    row.ratio,
                    row.log2_ratio,
                    row.magnitude,
                ]
            )


def make_three_phase_rows(
    phase_counts: Dict[str, Dict[str, int]],
    *,
    smoothing: float,
    min_total_count: int,
) -> List[PhaseRow]:
    labels = sorted(set().union(*(set(phase_counts.get(phase, {})) for phase in PHASES)))
    rows: List[PhaseRow] = []
    for label in labels:
        counts = {phase: int(phase_counts.get(phase, {}).get(label, 0)) for phase in PHASES}
        total_raw = sum(counts.values())
        if total_raw < min_total_count:
            continue
        denom = total_raw + smoothing * len(PHASES)
        shares = {phase: (counts[phase] + smoothing) / denom for phase in PHASES}
        dominant_phase = max(PHASES, key=lambda phase: shares[phase])
        dominance = shares[dominant_phase]
        other_phases = [phase for phase in PHASES if phase != dominant_phase]
        other_mean = sum(counts[phase] for phase in other_phases) / float(len(other_phases))
        dominant_over_other_ratio = (counts[dominant_phase] + smoothing) / (other_mean + smoothing)
        dominant_log2_ratio = math.log2(dominant_over_other_ratio)
        size_value = abs(dominant_log2_ratio)
        rows.append(
            PhaseRow(
                label=label,
                early_count=counts["early"],
                middle_count=counts["middle"],
                late_count=counts["late"],
                total_count=total_raw,
                early_share=shares["early"],
                middle_share=shares["middle"],
                late_share=shares["late"],
                dominant_phase=dominant_phase,
                dominance=dominance,
                dominant_over_other_ratio=dominant_over_other_ratio,
                dominant_log2_ratio=dominant_log2_ratio,
                size_value=size_value,
            )
        )
    rows.sort(key=lambda row: (row.size_value, row.total_count), reverse=True)
    return rows


def write_three_phase_csv(path: str, rows: Sequence[PhaseRow]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "word",
                "early_count",
                "middle_count",
                "late_count",
                "total_count",
                "early_share",
                "middle_share",
                "late_share",
                "dominant_phase",
                "dominance",
                "dominant_over_other_ratio",
                "dominant_log2_ratio",
                "size_value",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    row.early_count,
                    row.middle_count,
                    row.late_count,
                    row.total_count,
                    row.early_share,
                    row.middle_share,
                    row.late_share,
                    row.dominant_phase,
                    row.dominance,
                    row.dominant_over_other_ratio,
                    row.dominant_log2_ratio,
                    row.size_value,
                ]
            )


def color_func_from_ratios(rows: Sequence[ComparisonRow], cmap, norm):
    ratio_by_label = {row.label: row.log2_ratio for row in rows}

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return to_hex(cmap(norm(ratio_by_label.get(word, 0.0))))

    return color_func


def blend_hex(base_hex: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    base_hex = base_hex.lstrip("#")
    base = tuple(int(base_hex[i : i + 2], 16) for i in (0, 2, 4))
    neutral = (226, 232, 240)
    rgb = tuple(round(neutral[i] * (1.0 - amount) + base[i] * amount) for i in range(3))
    return "#" + "".join(f"{x:02x}" for x in rgb)


def color_func_from_phase_rows(rows: Sequence[PhaseRow]):
    phase_colors = {
        "early": "#F97316",
        "middle": "#7C3AED",
        "late": "#2563EB",
    }
    row_by_label = {row.label: row for row in rows}

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        row = row_by_label.get(word)
        if row is None:
            return "#64748B"
        strength = (row.dominance - (1.0 / len(PHASES))) / (1.0 - (1.0 / len(PHASES)))
        strength = max(0.15, min(1.0, strength))
        return blend_hex(phase_colors[row.dominant_phase], strength)

    return color_func


def save_comparison_cloud(
    rows: Sequence[ComparisonRow],
    *,
    out_path: str,
    phase_a: str,
    phase_b: str,
    seed: int,
    width: int,
    height: int,
    size_floor: float,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    if not rows:
        raise RuntimeError(f"No rows available for {phase_a} vs {phase_b}.")

    rows = list(rows)
    frequencies = {row.label: max(row.magnitude, 0.01) + float(size_floor) for row in rows}
    max_abs = max(abs(row.log2_ratio) for row in rows)
    max_abs = max(1.0, min(5.0, max_abs))
    cmap = LinearSegmentedColormap.from_list("phase_ratio", ["#1D4ED8", "#EDE9FE", "#FB923C"])
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    cloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        prefer_horizontal=0.98,
        random_state=seed,
        collocations=False,
        min_font_size=10,
        max_font_size=150,
        relative_scaling=0.75,
        margin=1,
    ).generate_from_frequencies(frequencies)
    cloud = cloud.recolor(color_func=color_func_from_ratios(rows, cmap, norm), random_state=seed)

    fig, ax = plt.subplots(figsize=(width / 160, height / 160), dpi=160)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{phase_a.title()} vs {phase_b.title()}", fontsize=15, pad=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"log2({phase_a} / {phase_b})", fontsize=9)
    cbar.set_ticks([-max_abs, 0, max_abs])
    cbar.set_ticklabels([phase_b.title(), "Equal", phase_a.title()])
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_frequency_cloud(
    counts: Dict[str, int],
    *,
    out_path: str,
    phase: str,
    seed: int,
    width: int,
    height: int,
    top_n: int,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    top_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    frequencies = {label: int(count) for label, count in top_items}
    color_by_phase = {
        "early": "#2563EB",
        "middle": "#7C3AED",
        "late": "#EA580C",
    }
    cloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        prefer_horizontal=0.98,
        random_state=seed,
        collocations=False,
        min_font_size=10,
        max_font_size=150,
        relative_scaling=0.6,
        margin=1,
        color_func=lambda *args, **kwargs: color_by_phase.get(phase, "#374151"),
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(width / 160, height / 160), dpi=160)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{phase.title()} Decoded Words", fontsize=15, pad=10)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_frequency_csv(
    path: str,
    phase_counts: Dict[str, Dict[str, int]],
    *,
    top_n: int,
) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "rank", "word", "count"])
        for phase in PHASES:
            top_items = sorted(
                phase_counts.get(phase, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_n]
            for rank, (word, count) in enumerate(top_items, start=1):
                writer.writerow([phase, rank, word, int(count)])


def make_ellipse_mask(width: int, height: int, *, pad_x: int = 150, pad_y: int = 140) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cx = width / 2.0
    cy = height / 2.0
    rx = (width - 2 * pad_x) / 2.0
    ry = (height - 2 * pad_y) / 2.0
    inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[inside] = 0
    return mask


def save_combined_frequency_cloud(
    phase_counts: Dict[str, Dict[str, int]],
    *,
    out_path: str,
    csv_path: str,
    seed: int,
    width: int,
    height: int,
    top_n: int,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    ensure_dir(os.path.dirname(csv_path))
    labels = sorted(set().union(*(set(phase_counts.get(phase, {})) for phase in PHASES)))
    rows = []
    for label in labels:
        counts = {phase: int(phase_counts.get(phase, {}).get(label, 0)) for phase in PHASES}
        total_count = sum(counts.values())
        if total_count <= 0:
            continue
        dominant_phase = max(PHASES, key=lambda phase: counts[phase])
        size_frequency = counts[dominant_phase]
        rows.append(
            {
                "word": label,
                "early_count": counts["early"],
                "middle_count": counts["middle"],
                "late_count": counts["late"],
                "total_count": total_count,
                "dominant_phase": dominant_phase,
                "size_frequency": size_frequency,
            }
        )
    rows.sort(key=lambda row: (row["size_frequency"], row["total_count"]), reverse=True)
    rows = rows[:top_n]
    frequencies = {row["word"]: row["size_frequency"] for row in rows if row["size_frequency"] > 0}
    phase_by_word = {row["word"]: row["dominant_phase"] for row in rows}
    color_by_phase = {
        "early": "#2563EB",
        "middle": "#7C3AED",
        "late": "#EA580C",
    }

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return color_by_phase.get(phase_by_word.get(word, "early"), "#374151")

    cloud = WordCloud(
        width=width,
        height=height,
        mask=make_ellipse_mask(width, height),
        background_color="white",
        prefer_horizontal=0.90,
        random_state=seed,
        collocations=False,
        min_font_size=12,
        max_font_size=155,
        relative_scaling=0.30,
        margin=1,
        color_func=color_func,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(width / 160, height / 160), dpi=160)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Early, Middle, Late Frequency", fontsize=16, pad=10)
    handles = [
        Patch(facecolor=color_by_phase["early"], label="Early"),
        Patch(facecolor=color_by_phase["middle"], label="Middle"),
        Patch(facecolor=color_by_phase["late"], label="Late"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=11)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "word",
                "early_count",
                "middle_count",
                "late_count",
                "total_count",
                "dominant_phase",
                "size_frequency",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_three_panel_frequency_cloud(
    phase_counts: Dict[str, Dict[str, int]],
    *,
    out_path: str,
    seed: int,
    width: int,
    height: int,
    top_n: int,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    color_by_phase = {
        "early": "#2563EB",
        "middle": "#7C3AED",
        "late": "#EA580C",
    }
    fig, axes = plt.subplots(1, 3, figsize=(width / 120, height / 180), dpi=160)
    for ax, phase in zip(axes, PHASES):
        top_items = sorted(
            phase_counts.get(phase, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        frequencies = {label: int(count) for label, count in top_items}
        cloud = WordCloud(
            width=max(400, width // 3),
            height=max(400, height),
            background_color="white",
            prefer_horizontal=0.98,
            random_state=seed,
            collocations=False,
            min_font_size=8,
            max_font_size=120,
            relative_scaling=0.75,
            margin=1,
            color_func=lambda *args, phase=phase, **kwargs: color_by_phase.get(phase, "#374151"),
        ).generate_from_frequencies(frequencies)
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{phase.title()} Frequency", fontsize=13, pad=8)
    fig.suptitle("Frequency Sized Decoded Words", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_three_phase_cloud(
    rows: Sequence[PhaseRow],
    *,
    out_path: str,
    seed: int,
    width: int,
    height: int,
    size_floor: float,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    if not rows:
        raise RuntimeError("No rows available for three phase WordCloud.")

    rows = list(rows)
    frequencies = {row.label: row.size_value + float(size_floor) for row in rows}
    cloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        prefer_horizontal=0.98,
        random_state=seed,
        collocations=False,
        min_font_size=10,
        max_font_size=150,
        relative_scaling=0.65,
        margin=1,
    ).generate_from_frequencies(frequencies)
    cloud = cloud.recolor(color_func=color_func_from_phase_rows(rows), random_state=seed)

    fig, ax = plt.subplots(figsize=(width / 160, height / 160), dpi=160)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Early, Middle, Late", fontsize=15, pad=10)
    handles = [
        Patch(facecolor="#F97316", label="Early"),
        Patch(facecolor="#7C3AED", label="Middle"),
        Patch(facecolor="#2563EB", label="Late"),
        Patch(facecolor="#E2E8F0", label="Mixed"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=11)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_outputs(
    phase_counts: Dict[str, Dict[str, int]],
    *,
    output_dir: str,
    top_n: int,
    min_total_count: int,
    smoothing: float,
    seed: int,
    width: int,
    height: int,
    size_floor: float,
) -> None:
    comparison_dir = os.path.join(output_dir, "comparative_wordclouds")
    all_phase_dir = os.path.join(output_dir, "all_phase_wordclouds")
    table_dir = os.path.join(output_dir, "comparative_wordcloud_tables")
    phase_dir = os.path.join(output_dir, "phase_wordclouds")
    frequency_dir = os.path.join(output_dir, "frequency_wordclouds")
    ensure_dir(comparison_dir)
    ensure_dir(all_phase_dir)
    ensure_dir(table_dir)
    ensure_dir(phase_dir)
    ensure_dir(frequency_dir)

    three_phase_rows = make_three_phase_rows(
        phase_counts,
        smoothing=smoothing,
        min_total_count=min_total_count,
    )[:top_n]
    save_three_phase_cloud(
        three_phase_rows,
        out_path=os.path.join(all_phase_dir, "early_middle_late.png"),
        seed=seed,
        width=width,
        height=height,
        size_floor=size_floor,
    )
    write_three_phase_csv(
        os.path.join(table_dir, "early_middle_late.csv"),
        three_phase_rows,
    )

    for phase_a, phase_b in PAIRWISE_PHASES:
        rows = make_comparison_rows(
            phase_counts[phase_a],
            phase_counts[phase_b],
            smoothing=smoothing,
            min_total_count=min_total_count,
        )
        top_rows = rows[:top_n]
        save_comparison_cloud(
            top_rows,
            out_path=os.path.join(comparison_dir, f"{phase_a}_vs_{phase_b}.png"),
            phase_a=phase_a,
            phase_b=phase_b,
            seed=seed,
            width=width,
            height=height,
            size_floor=size_floor,
        )
        write_comparison_csv(
            os.path.join(table_dir, f"{phase_a}_vs_{phase_b}.csv"),
            rows,
            phase_a,
            phase_b,
        )

    for phase in PHASES:
        save_frequency_cloud(
            phase_counts[phase],
            out_path=os.path.join(phase_dir, f"{phase}.png"),
            phase=phase,
            seed=seed,
            width=width,
            height=height,
            top_n=top_n,
        )

    save_three_panel_frequency_cloud(
        phase_counts,
        out_path=os.path.join(frequency_dir, "early_middle_late_frequency.png"),
        seed=seed,
        width=width,
        height=height,
        top_n=top_n,
    )
    write_frequency_csv(
        os.path.join(frequency_dir, "top_frequency_words_by_phase.csv"),
        phase_counts,
        top_n=top_n,
    )
    save_combined_frequency_cloud(
        phase_counts,
        out_path=os.path.join(frequency_dir, "early_middle_late_combined_frequency_blob.png"),
        csv_path=os.path.join(frequency_dir, "early_middle_late_combined_frequency_blob.csv"),
        seed=seed,
        width=width,
        height=height,
        top_n=max(top_n, 240),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make WordCloud plots from Dream decoded events or phase counts.")
    parser.add_argument("--events_jsonl", type=str, default=None)
    parser.add_argument("--counts_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=150)
    parser.add_argument("--min_total_count", type=int, default=3)
    parser.add_argument("--min_word_len", type=int, default=2)
    parser.add_argument("--vocab_filter", type=str, default="all", choices=["all", "stopwords", "content"])
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--size_floor", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.events_jsonl:
        phase_counts = load_word_counts_from_events(args.events_jsonl, min_word_len=int(args.min_word_len))
        default_parent = os.path.dirname(args.events_jsonl)
    elif args.counts_json:
        raw_counts = load_phase_counts(args.counts_json)
        phase_counts = {phase: aggregate_labels(raw_counts.get(phase, {})) for phase in PHASES}
        default_parent = os.path.dirname(args.counts_json)
    else:
        raise ValueError("Provide either --events_jsonl or --counts_json.")

    phase_counts = apply_vocab_filter(phase_counts, str(args.vocab_filter))
    output_dir = args.output_dir or os.path.join(default_parent, "wordcloud_plots")
    make_outputs(
        phase_counts,
        output_dir=output_dir,
        top_n=int(args.top_n),
        min_total_count=int(args.min_total_count),
        smoothing=float(args.smoothing),
        seed=int(args.seed),
        width=int(args.width),
        height=int(args.height),
        size_floor=float(args.size_floor),
    )
    print(f"[DONE] WordCloud plots written to: {output_dir}")
if __name__ == "__main__":
    main()
