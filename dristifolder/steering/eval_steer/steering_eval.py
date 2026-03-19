"""
End-to-end SAE steering evaluation with:
- One judge request per comparison returning both scores
- Concurrent judge requests within a feature (bounded)
- Batch PPL computation
- Checkpoint every N features
- Resume support:
    * Comparisons are considered done ONLY if concept_judge_ok is True

Adds CLI option:
  --device cuda:0
and forwards it to PPL computation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple, Optional

from config import EVAL_ROOT, CHECKPOINT_EVERY_N_FEATURES
from sae_data import find_steering_json_files
from llm_concept_judge import get_global_judge
from fluency_ppl import calc_ppl_batch

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def build_text(prefix: str, continuation: str) -> str:
    prefix = (prefix or "").strip()
    continuation = (continuation or "").strip()
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    return f"{prefix} {continuation}"


def comparison_has_scores(comp: Dict[str, Any]) -> bool:
    """
    A comparison is considered complete only if:
      - concept_judge_ok is True (prevents treating fallback scores as done)
      - concept scores exist
      - PPL scores exist
    """
    if comp.get("concept_judge_ok") is not True:
        return False

    required_keys = [
        "concept_score_without",
        "concept_score_after",
        "ppl_without",
        "ppl_after",
    ]
    for k in required_keys:
        if k not in comp or comp[k] is None:
            return False
    return True


def safe_write_json(data: Dict[str, Any], dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(dst_path)


def derive_output_subdir_name(source_dir: Path) -> str:
    name = source_dir.name
    suffix = "_sae_steering_results"
    replace_with = "_sae_steering_scores"
    if name.endswith(suffix):
        return name[: -len(suffix)] + replace_with
    return f"{name}_scores"


def _calc_ppl_batch_with_optional_device(texts: List[str], device: Optional[str]) -> List[float]:
    """
    Call calc_ppl_batch with device if supported by fluency_ppl.py, otherwise fall back.
    This keeps backward compatibility if calc_ppl_batch(texts) is the only signature.
    """
    if not device:
        return calc_ppl_batch(texts)

    try:
        # Many implementations expose calc_ppl_batch(texts, device=...)
        return calc_ppl_batch(texts, device=device)  # type: ignore[arg-type]
    except TypeError:
        # Signature does not accept device; use original call.
        return calc_ppl_batch(texts)


async def process_single_file_async(src_path: Path, dst_path: Path, device: Optional[str]) -> None:
    if dst_path.exists():
        with dst_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Dict[str, Any]] = json.load(f)
        print(f"       [RESUME] Using existing output file as base: {dst_path}")
    else:
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    judge = get_global_judge()

    feature_items = list(data.items())
    total_features = len(feature_items)

    if tqdm is not None:
        iterator = tqdm(
            feature_items,
            total=total_features,
            desc=f"Features ({src_path.name})",
            unit="feat",
            dynamic_ncols=True,
        )
    else:
        iterator = feature_items

    features_since_write = 0
    any_new_work_since_write = False

    for feat_idx, (feature_key, feature_obj) in enumerate(iterator, start=1):
        if tqdm is not None:
            iterator.set_postfix_str(f"feature={feature_key}")
        else:
            print(f"       [FEATURE] {feat_idx}/{total_features} feature={feature_key}")

        explanation = feature_obj.get("explanation", "")
        comparisons: List[Dict[str, Any]] = feature_obj.get("comparisons", [])

        concept_without_list: List[float] = []
        concept_after_list: List[float] = []
        ppl_without_list: List[float] = []
        ppl_after_list: List[float] = []

        pending_indices: List[int] = []
        pending_pairs: List[Tuple[str, str]] = []
        pending_texts_for_ppl: List[str] = []

        # Gather pending work
        for i, comp in enumerate(comparisons):
            if comparison_has_scores(comp):
                concept_without_list.append(float(comp["concept_score_without"]))
                concept_after_list.append(float(comp["concept_score_after"]))
                ppl_without_list.append(float(comp["ppl_without"]))
                ppl_after_list.append(float(comp["ppl_after"]))
                continue

            prefix = comp.get("prefix", "")
            wo = comp.get("without_steer_output", "")
            af = comp.get("after_steer_output", "")

            wo_text = build_text(prefix, wo)
            af_text = build_text(prefix, af)

            pending_indices.append(i)
            pending_pairs.append((wo_text, af_text))
            pending_texts_for_ppl.extend([wo_text, af_text])

        if pending_pairs:
            # Judge (concurrent, bounded inside judge)
            pair_scores = await judge.score_pairs(explanation, pending_pairs)
            # List[(score_without, score_after, ok)]

            # PPL batch for all pending texts (optionally on a specified device)
            ppls = _calc_ppl_batch_with_optional_device(pending_texts_for_ppl, device=device)

            for j, comp_idx in enumerate(pending_indices):
                comp = comparisons[comp_idx]

                s_wo, s_af, ok = pair_scores[j]
                ppl_wo = float(ppls[2 * j])
                ppl_af = float(ppls[2 * j + 1])

                comp["concept_score_without"] = float(s_wo)
                comp["concept_score_after"] = float(s_af)
                comp["concept_judge_ok"] = bool(ok)

                comp["ppl_without"] = ppl_wo
                comp["ppl_after"] = ppl_af

                # Only include in averages if ok=True (otherwise it would pollute statistics)
                if ok:
                    concept_without_list.append(float(s_wo))
                    concept_after_list.append(float(s_af))
                ppl_without_list.append(ppl_wo)
                ppl_after_list.append(ppl_af)

            any_new_work_since_write = True

        # Feature-level averages:
        # - concept averages computed only over ok=True comparisons
        # - ppl averages over all comparisons (ppl always computed)
        if comparisons and concept_without_list:
            feature_obj["avg_concept_score_without_steer"] = float(mean(concept_without_list))
            feature_obj["avg_concept_score_after_steer"] = float(mean(concept_after_list))
        else:
            feature_obj["avg_concept_score_without_steer"] = None
            feature_obj["avg_concept_score_after_steer"] = None

        if comparisons and ppl_without_list:
            feature_obj["avg_ppl_without_steer"] = float(mean(ppl_without_list))
            feature_obj["avg_ppl_after_steer"] = float(mean(ppl_after_list))
        else:
            feature_obj["avg_ppl_without_steer"] = None
            feature_obj["avg_ppl_after_steer"] = None

        feature_obj["comparisons"] = comparisons
        data[feature_key] = feature_obj

        # Checkpoint every N features
        features_since_write += 1
        if features_since_write >= CHECKPOINT_EVERY_N_FEATURES:
            if any_new_work_since_write or (not dst_path.exists()):
                safe_write_json(data, dst_path)
            features_since_write = 0
            any_new_work_since_write = False

    if tqdm is not None:
        iterator.close()

    # Final flush
    if any_new_work_since_write or (not dst_path.exists()):
        safe_write_json(data, dst_path)

    print("       [DONE] File processed (checkpoint every N features).")


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SAE steering results and write augmented JSON files "
                    "(resume + concurrent judge + batch PPL + checkpoint)."
    )
    parser.add_argument("source_dir", type=str, help="Path to steering results directory.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for PPL computation, e.g. 'cuda:0' or 'cpu'.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory is not valid: {source_dir}")

    json_files = find_steering_json_files(source_dir)
    if not json_files:
        print(f"[WARN] No steering JSON files found under: {source_dir}")
        return

    output_subdir_name = derive_output_subdir_name(source_dir)
    output_dir = EVAL_ROOT / output_subdir_name

    print(f"[INFO] Source directory : {source_dir}")
    print(f"[INFO] Output directory : {output_dir}")
    print(f"[INFO] Found {len(json_files)} steering JSON files.")
    if args.device:
        print(f"[INFO] Using device for PPL: {args.device}")

    for idx, src_path in enumerate(json_files, start=1):
        rel_path = src_path.relative_to(source_dir)
        dst_path = output_dir / rel_path

        # Keep this print as requested.
        print(f"[INFO] ({idx}/{len(json_files)}) Processing file: {src_path}")
        print(f"       -> Output file: {dst_path}")

        await process_single_file_async(src_path, dst_path, device=args.device)

    print("[INFO] All files processed.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
