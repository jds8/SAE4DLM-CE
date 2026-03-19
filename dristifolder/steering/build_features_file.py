# build_features_file.py
# -*- coding: utf-8 -*-
"""
Build per-(layer, k) steering feature files from autointerp result JSONs.

Input JSONs can be either:
  - "completed" results with a top-level key `per_latent_results`
  - "checkpoint" results with a top-level key `latents`

Output format (per file):
  {
    "<layer>": {
      "<latent_id>": {"explanation": "...", "score": 0.95},
      ...
    }
  }

Notes:
- One output JSON per input (unless the output already exists):
      <out_dir>/features_{model_tag}_layer{L}_l0_{K}.json
  where model_tag is "qwen2.5" for LLM files and "dream7b" for DLM files.
- Auto-parse layer and k (= L0) from filenames that strictly match any of:
      qwen2.5_7b_layer{L}_l0_{K}.json
      qwen2.5_7b_layer{L}_l0_{K}.json.ckpt.json
      dream_7b_layer{L}_l0_{K}.json
      dream_7b_layer{L}_l0_{K}.json.ckpt.json
- If the corresponding features_{model_tag}_layer{L}_l0_{K}.json already exists,
  the input file is skipped.
- Optional filters:
      --layers "1,5,10"  -> include only these layers
      --k 80             -> include only files whose L0 (k) equals 80
      --top_n 100        -> keep at most 100 latents per file
  If 'score' exists, truncation prefers highest score first.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

# Filename regexes:
#   - match both *.json and *.json.ckpt.json
#   - capture layer (L) and k (L0)
QWEN_RE  = re.compile(r"^qwen2\.5_7b_layer(\d+)_l0_(\d+)\.json(?:\.ckpt\.json)?$")
DREAM_RE = re.compile(r"^dream_7b_layer(\d+)_l0_(\d+)\.json(?:\.ckpt\.json)?$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--autointerp_dir",
        type=str,
        required=True,
        help="Directory containing autointerp result JSON files (completed or checkpoint).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write per-(layer,k) features_file JSONs.",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Optional comma-separated layer ids to include (e.g., '1,5,10').",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional filter: include only files whose L0 (k) equals this value.",
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Optional: keep at most top_n latents per file.",
    )
    return ap.parse_args()


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_latent_container(doc: dict) -> Dict:
    """
    Return the mapping that holds per-latent information.

    Supports both:
      - completed autointerp JSONs with key 'per_latent_results'
      - checkpoint JSONs with key 'latents'
    """
    per_latent = doc.get("per_latent_results")
    if isinstance(per_latent, dict) and per_latent:
        return per_latent

    latents = doc.get("latents")
    if isinstance(latents, dict) and latents:
        return latents

    return {}


def _collect_details(doc: dict, top_n: Optional[int]) -> Dict[str, Dict[str, float]]:
    """
    Build details mapping: { "<latent_id>": {"explanation": str, "score": float} }.

    - Handles both 'per_latent_results' and 'latents' containers.
    - If any score exists, sort by score desc before truncation.
      Otherwise, keep insertion order.
    """
    per_latent = _get_latent_container(doc)
    items: List[Tuple[int, float, str]] = []  # (latent_id, score_or_default, explanation)

    for _, item in per_latent.items():
        latent_raw = item.get("latent", None)

        # Accept either int or digit-like string for latent id
        if isinstance(latent_raw, int):
            latent_id = latent_raw
        elif isinstance(latent_raw, str) and latent_raw.isdigit():
            latent_id = int(latent_raw)
        else:
            continue

        score = item.get("score", None)
        score_val = float(score) if isinstance(score, (int, float)) else 0.0

        explanation_val = item.get("explanation", "")
        explanation = explanation_val if isinstance(explanation_val, str) else ""

        items.append((latent_id, score_val, explanation))

    if not items:
        return {}

    # Sort by score desc if any non-zero score; else keep order
    if any(s != 0.0 for _, s, _ in items):
        items.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[Tuple[int, float, str]] = []
    for lid, s, exp in items:
        if lid not in seen:
            seen.add(lid)
            deduped.append((lid, s, exp))

    # Truncate if requested
    if top_n is not None:
        deduped = deduped[:top_n]

    # Pack as strings → objects
    details: Dict[str, Dict[str, float]] = {}
    for lid, s, exp in deduped:
        details[str(lid)] = {"explanation": exp, "score": s}
    return details


def _match_file(fname: str) -> Optional[Tuple[str, int, int]]:
    """
    Try to match fname against known patterns.

    Returns (model_tag, layer, k) or None if not matched.
    model_tag in {"qwen2.5", "dream7b"}.
    """
    m = QWEN_RE.match(fname)
    if m:
        return "qwen2.5", int(m.group(1)), int(m.group(2))

    m = DREAM_RE.match(fname)
    if m:
        return "dream7b", int(m.group(1)), int(m.group(2))

    return None


def main():
    args = parse_args()
    _ensure_dir(args.out_dir)

    wanted_layers = None
    if args.layers:
        wanted_layers = {int(x.strip()) for x in args.layers.split(",") if x.strip()}

    print(f"[INFO] Scanning: {args.autointerp_dir}")
    print(f"[INFO] Output dir: {args.out_dir}")
    if wanted_layers is not None:
        print(f"[INFO] Filter layers: {sorted(wanted_layers)}")
    if args.k is not None:
        print(f"[INFO] Filter L0 (k): {args.k}")
    if args.top_n is not None:
        print(f"[INFO] Per-file top_n: {args.top_n}")

    num_written = 0
    num_skipped_empty = 0
    num_skipped_existing = 0
    num_scanned = 0

    for fname in sorted(os.listdir(args.autointerp_dir)):
        matched = _match_file(fname)
        if not matched:
            continue

        model_tag, layer, k_val = matched

        if wanted_layers is not None and layer not in wanted_layers:
            continue
        if args.k is not None and k_val != args.k:
            continue

        out_name = f"features_{model_tag}_layer{layer}_l0_{k_val}.json"
        out_path = os.path.join(args.out_dir, out_name)

        # Skip if output already exists
        if os.path.exists(out_path):
            num_skipped_existing += 1
            print(f"[SKIP] Output already exists for layer={layer}, k={k_val}: {out_name}")
            continue

        fpath = os.path.join(args.autointerp_dir, fname)
        doc = _load_json(fpath)
        if not doc:
            print(f"[WARN] Failed to load JSON from {fname}; skipping.")
            continue

        num_scanned += 1
        details = _collect_details(doc, args.top_n)
        if not details:
            num_skipped_empty += 1
            print(f"[WARN] No latents found in {fname}; skipping.")
            continue

        # Output ONLY the layer key mapping to details
        content: Dict[str, Dict[str, Dict[str, float]]] = {str(layer): details}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

        num_written += 1
        print(
            f"[OK] [{model_tag}] Wrote {out_name} with {len(details)} latents "
            f"(from {fname})."
        )

    print(
        "[DONE] Scanned files: {scanned}, written: {written}, "
        "empty-skipped: {empty_skipped}, existing-skipped: {existing_skipped}".format(
            scanned=num_scanned,
            written=num_written,
            empty_skipped=num_skipped_empty,
            existing_skipped=num_skipped_existing,
        )
    )


if __name__ == "__main__":
    main()
