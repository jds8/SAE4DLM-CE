# steer.py
# -*- coding: utf-8 -*-
import gc
import os
import re
import json
import time
import argparse
import random
from typing import Dict, List, Tuple, Optional
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as lm_pipeline,
)

from utils import (
    get_sae,
    try_get_final_norm_and_lm_head,
)
from sae_utils import init_hook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="qwen2.5_7b",
                        choices=["qwen2.5_7b", "gemma2_2b", "gemma2_9b", "gemma2_9b_it_131"])
    parser.add_argument('--features_file', type=str, required=True,
                        help="Path to a single features file OR a directory containing multiple features files.")
    parser.add_argument('--sae_root_dir', type=str, required=True,
                        help="Base dir containing resid_post_layer_{idx}/trainer_*/ with ae.pt & config.json")
    parser.add_argument('--sae_trainer', type=str, default=None,
                        help="Explicit trainer directory name like 'trainer_0'. Overrides --sae_k if provided.")
    parser.add_argument('--sae_k', type=int, default=None,
                        help="Select trainer whose config.json has trainer.k == sae_k (Top-K sparsity L0)")
    parser.add_argument('--amp_factor', type=float, default=1.2)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Folder to write steering results. Default: <auto-detected default path>")
    parser.add_argument('--cache_path', type=str, default=None,
                        help="Only used in SINGLE-FILE mode: if provided, write results to this JSON file.")
    parser.add_argument('--dtype', type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--do_sample', action='store_true', default=True)
    parser.add_argument('--trust_remote_code', action='store_true', default=True)
    parser.add_argument('--n_prefix', type=int, default=5,
                        help="Randomly sample n neutral prefixes from the built-in list for each feature.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Global seed for reproducible prefix sampling and generation.")
    return parser.parse_args()


def _select_model_name(model_type: str) -> str:
    if model_type == "qwen2.5_7b":
        return "Qwen/Qwen2.5-7B"
    if model_type == "gemma2_2b":
        return "google/gemma-2-2b"
    if model_type == "gemma2_9b":
        return "google/gemma-2-9b"
    if model_type == "gemma2_9b_it_131":
        return "google/gemma-2-9b-it"
    raise ValueError(f"Model type not supported: {model_type}")


def _seed_everywhere(seed: int):
    """Set seeds for Python and Torch (CPU/GPU) to make sampling reproducible."""
    import random as _random
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_dtype(dtype_flag: str):
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    if dtype_flag == "fp32":
        return torch.float32
    return "auto"


def _default_results_dir(sae_root_dir: str) -> str:
    """
    Choose a default results directory based on sae_root_dir.

    If sae_root_dir indicates the SAE was trained for Qwen2.5-7B, store results under:
      /home/dslabra5/sae4dlm/steering/steering_results_file/qwen_sae_steering_results

    Otherwise, fall back to the original default:
      <this_script_dir>/llm_sae_steering_results
    """
    sae_root_lower = (sae_root_dir or "").lower()
    # Heuristic detection for Qwen2.5-7B SAE directories
    if ("qwen2.5-7b" in sae_root_lower) or ("qwen_qwen2.5-7b" in sae_root_lower) or ("qwen2.5" in sae_root_lower):
        base_dir = "/home/dslabra5/sae4dlm/steering/steering_results_file"
        return os.path.join(base_dir, "qwen_sae_steering_results")

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "llm_sae_steering_results")


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# Strict features filename pattern: features_qwen2.5_layer{L}_l0_{K}.json
FEATURES_RE_STRICT = re.compile(r"^features_qwen2\.5_layer(\d+)_l0_(\d+)\.json$")
# Relaxed fallback: ...layer{L}..._l0_{K}.json
FEATURES_RE_RELAXED = re.compile(r"^.*layer(\d+).*_l0_(\d+)\.json$")


def _infer_layer_k_from_features_file(path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer (layer, k) from features_file name.
    Returns (layer:int, k:int) or (None, None) if not matched.
    """
    fname = os.path.basename(path)
    m = FEATURES_RE_STRICT.match(fname)
    if not m:
        m = FEATURES_RE_RELAXED.match(fname)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _gather_features_files(path: str) -> List[str]:
    """
    If 'path' is a file, return [path] (if it matches the relaxed pattern).
    If 'path' is a directory, return sorted list of matching files under it.
    """
    if os.path.isfile(path):
        L, K = _infer_layer_k_from_features_file(path)
        if L is None or K is None:
            raise ValueError(f"Features file name not recognized: {path}")
        return [path]

    if os.path.isdir(path):
        out = []
        for fname in sorted(os.listdir(path)):
            if FEATURES_RE_STRICT.match(fname) or FEATURES_RE_RELAXED.match(fname):
                out.append(os.path.join(path, fname))
        if not out:
            raise FileNotFoundError(f"No valid features files found under directory: {path}")
        return out

    raise FileNotFoundError(f"features_file path not found: {path}")


def _result_path_for_features_file(features_file: str, results_dir: str) -> str:
    """
    Derive output JSON path from a features_file name.
    Example:
      features_qwen2.5_layer5_l0_50.json -> steer_qwen2.5_layer5_l0_50.json
    """
    fname = os.path.basename(features_file)
    L, K = _infer_layer_k_from_features_file(fname)
    if L is not None and K is not None:
        base = f"steer_qwen2.5_layer{L}_l0_{K}.json"
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"steer_{ts}.json"
    return os.path.join(results_dir, base)


def _build_textgen_pipeline(model_name: str, dtype, device: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(None if dtype == "auto" else dtype),
        device_map={"": device},
        trust_remote_code=trust_remote_code
    )
    textgen = lm_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    if textgen.tokenizer.pad_token_id is None and hasattr(textgen.model.config, "eos_token_id"):
        textgen.tokenizer.pad_token_id = textgen.model.config.eos_token_id
    print(f"Device set to use {device}")
    return textgen


def _sanity_print_model(textgen):
    try:
        final_layer_norm, lm_head = try_get_final_norm_and_lm_head(textgen.model)
        ln_shape = getattr(getattr(final_layer_norm, "weight", None), "shape", None)
        print("[CHECK] final_layer_norm:", type(final_layer_norm), "weight shape:", ln_shape)
        print("[CHECK] lm_head:", type(lm_head), "weight shape:", getattr(lm_head.weight, "shape", None))
    except Exception as e:
        print("[WARN] final norm / lm_head resolution failed:", repr(e))

    # Also verify layers container exists (hook location)
    from sae_utils import _resolve_layers_container  # type: ignore
    layers_container = _resolve_layers_container(textgen.model)
    print("[CHECK] layers_container type:", type(layers_container), "num layers:", len(layers_container))
    if len(layers_container) > 0:
        print("[CHECK] first two blocks types:",
              type(layers_container[0]),
              type(layers_container[1]) if len(layers_container) > 1 else None)


def _load_features_by_layers_compat(path: str) -> Dict[int, List[int]]:
    """
    Load features from a single features_file with compatibility for:
      New format (preferred):
         { "<layer>": { "<id>": {"explanation": "...", "score": 0.95}, ... } }
      Old format:
         { "<layer>": [id1, id2, ...], "__details__": {...} }
    Returns: {layer_int: [feature_ids (ints) sorted asc]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    out: Dict[int, List[int]] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k.isdigit():
            continue
        layer = int(k)
        if isinstance(v, list):
            ids = [int(x) for x in v]
        elif isinstance(v, dict):
            ids = sorted(int(x) for x in v.keys() if str(x).isdigit())
        else:
            ids = []
        if ids:
            out[layer] = sorted(set(ids))
    return out


def _load_feature_details(path: str) -> Dict[int, Dict[int, Dict[str, object]]]:
    """
    Build { layer_int: { feature_id_int: {"explanation": str, "score": float} } }.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    details_by_layer: Dict[int, Dict[int, Dict[str, object]]] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k.isdigit():
            continue
        layer = int(k)
        if isinstance(v, dict):
            inner: Dict[int, Dict[str, object]] = {}
            for fid_str, info in v.items():
                if not str(fid_str).isdigit():
                    continue
                fid = int(fid_str)
                if isinstance(info, dict):
                    inner[fid] = {
                        "explanation": info.get("explanation", ""),
                        "score": info.get("score", None),
                    }
            if inner:
                details_by_layer[layer] = inner
    # minimal compat with old "__details__"
    if "__details__" in obj and isinstance(obj["__details__"], dict):
        for lk, lv in obj["__details__"].items():
            if str(lk).isdigit() and isinstance(lv, dict):
                layer = int(lk)
                inner = details_by_layer.get(layer, {})
                for fid_str, info in lv.items():
                    if str(fid_str).isdigit() and isinstance(info, dict):
                        fid = int(fid_str)
                        inner[fid] = {
                            "explanation": info.get("explanation", ""),
                            "score": info.get("score", None),
                        }
                if inner:
                    details_by_layer[layer] = inner
    return details_by_layer


def _pick_prefixes_per_feature(all_prefixes: List[str], n_prefix: int, seed: Optional[int], layer: int, feature: int) -> List[str]:
    """
    Deterministically sample n prefixes per-feature using a local RNG seeded
    from (global seed, layer, feature), independent of iteration order.
    """
    n = max(1, min(n_prefix, len(all_prefixes)))
    if seed is None:
        return random.sample(all_prefixes, n)
    local_rng = random.Random((hash((seed, layer, feature)) & 0xFFFFFFFF))
    return local_rng.sample(all_prefixes, n)


def _all_expected_keys(features_by_layers: Dict[int, List[int]]) -> List[str]:
    """Return ['L_F', ...] for all layer-feature pairs."""
    keys = []
    for layer, feats in features_by_layers.items():
        for f in feats:
            keys.append(f"{layer}_{f}")
    return keys


def _result_is_complete(out_path: str, expected_keys: List[str]) -> bool:
    """Return True if out_path exists and contains ALL expected_keys."""
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return False
    have = set(obj.keys())
    need = set(expected_keys)
    return need.issubset(have)


def _run_steering_for_one_features_file(
    textgen,
    tokenizer,
    device: str,
    sae_root_dir: str,
    features_file: str,
    results_dir: str,
    amp_factor: float,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    sae_trainer: Optional[str],
    sae_k_global: Optional[int],
    n_prefix: int,
    seed: Optional[int],
):
    """
    Run steering for a SINGLE features_file and write results to a per-file JSON in results_dir.
    Supports resume: if out JSON exists, will skip features already present.
    """
    out_path = _result_path_for_features_file(features_file, results_dir)
    _ensure_parent_dir(out_path)

    # Load features + details
    features_by_layers: Dict[int, List[int]] = _load_features_by_layers_compat(features_file)
    details_by_layer: Dict[int, Dict[int, Dict[str, object]]] = _load_feature_details(features_file)
    expected_keys = _all_expected_keys(features_by_layers)

    # If result already complete, skip entirely
    if _result_is_complete(out_path, expected_keys):
        print(f"[SKIP] Completed result exists for {os.path.basename(features_file)} -> {out_path}")
        return

    print(f"[INFO] Results will be written to: {out_path}")
    print(f"[INFO] Loaded features for {len(features_by_layers)} layer(s) from {features_file}")

    # Infer k from filename if needed
    _, inf_k = _infer_layer_k_from_features_file(features_file)
    local_k = sae_k_global if sae_k_global is not None else inf_k
    if sae_trainer is None and inf_k is not None and local_k == inf_k:
        print(f"[INFO] Inferred SAE L0 (k) from features_file name: k={local_k}")

    # Read existing cache to resume
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            generation_cache = json.load(f)
    else:
        generation_cache = {}

    # SAE cache PER FILE
    saes: Dict[int, nn.Module] = {}

    # Neutral prefix pool
    all_prefixes = [
        "Findings show that", "I once heard that", "Then the man said:", "I believe that",
        "The news mentioned", "She saw a", "It is observed that", "Studies indicate that",
        "According to reports,", "Research suggests that", "It has been noted that", "I remember when",
        "It all started when", "The legend goes that", "If I recall correctly,", "People often say that",
        "Once upon a time,", "It’s no surprise that", "Have you ever noticed that", "I couldn't believe when",
        "The first thing I heard was", "Let me tell you a story about", "Someone once told me that",
        "It might sound strange, but", "They always warned me that", "Nobody expected that", "Funny thing is,",
        "I never thought I'd say this, but", "What surprised me most was", "The other day, I overheard that",
        "Back in the day,", "You won’t believe what happened when", "A friend of mine once said,",
        "I just found out that", "It's been a long time since", "In my experience,",
        "The craziest part was when", "If you think about it,", "I was shocked to learn that",
        "For some reason,", "I can’t help but wonder if", "It makes sense that",
        "At first, I didn't believe that", "That reminds me of the time when", "It all comes down to",
        "One time, I saw that", "I was just thinking about how", "Imagine a world where",
        "They never expected that", "I always knew that"
    ]

    # Main steering loop (with resume)
    for layer, features in features_by_layers.items():
        # Load SAE for this layer (once) and ensure on device
        saes[layer] = get_sae(
            model_type="qwen2.5_7b",
            layer=layer,
            saes=saes,
            backend="dl_local",
            dl_local_dir=sae_root_dir,
            device=device,
            trainer_name=sae_trainer,
            k_topk=local_k,
        ).to(device)

        for feature in tqdm(features, desc=f"[{os.path.basename(features_file)}] Layer {layer} features"):
            layer_feature_key = f"{layer}_{feature}"

            # Resume: skip if already present
            if layer_feature_key in generation_cache:
                continue

            # Deterministic per-feature prefix sampling
            prefixes = _pick_prefixes_per_feature(all_prefixes, n_prefix, seed, layer, feature)

            # Per-feature details (may be missing)
            f_details = details_by_layer.get(layer, {}).get(feature, {})
            explanation = f_details.get("explanation", "")
            score = f_details.get("score", None)

            # Ensure SAE is on device
            saes[layer] = saes[layer].to(device)
            sae = saes[layer]

            # Prepare comparison container
            comparisons: List[Dict[str, str]] = []

            # Run WITHOUT steering first (baseline)
            with torch.no_grad():
                for prefix in prefixes:
                    # Reproducible baseline call
                    if seed is not None:
                        base_seed = (hash((layer, feature, prefix, "baseline", seed)) & 0xFFFFFFFF)
                        _seed_everywhere(base_seed)

                    out_base = textgen(
                        prefix,
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0]["generated_text"]

                    # WITH steering: attach hook only for this call
                    hook_args = SimpleNamespace(amp_factor=amp_factor)
                    handle = init_hook(textgen, sae, layer, feature, device, hook_args)
                    try:
                        if seed is not None:
                            steer_seed = (hash((layer, feature, prefix, "steered", seed)) & 0xFFFFFFFF)
                            _seed_everywhere(steer_seed)

                        out_steer = textgen(
                            prefix,
                            do_sample=do_sample,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                        )[0]["generated_text"]
                    finally:
                        handle.remove()

                    comparisons.append({
                        "prefix": prefix,
                        "without_steer_output": out_base,
                        "after_steer_output": out_steer,
                    })

            # Append result & flush to disk (safe for resume)
            generation_cache[layer_feature_key] = {
                "explanation": explanation,
                "score": score,
                "comparisons": comparisons
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(generation_cache, f, ensure_ascii=False, indent=2)

        # After finishing all features of this layer, move SAE back to CPU to free VRAM
        saes[layer] = saes[layer].cpu()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[DONE] Steering runs complete/updated for {os.path.basename(features_file)}. Results at: {out_path}")


def main():
    args = parse_args()

    # Resolve results dir
    results_dir = args.results_dir or _default_results_dir(args.sae_root_dir)
    _ensure_dir(results_dir)

    # Build model + tokenizer
    model_name = _select_model_name(args.model_type)
    dtype = _select_dtype(args.dtype)
    textgen = _build_textgen_pipeline(model_name, dtype, args.device, args.trust_remote_code)
    tokenizer = textgen.tokenizer

    # Sanity prints once
    _sanity_print_model(textgen)

    # Expand features_file into a list (single file or directory)
    features_files = _gather_features_files(args.features_file)
    is_dir_mode = os.path.isdir(args.features_file)
    if is_dir_mode and args.cache_path is not None:
        print("[WARN] --cache_path is ignored in DIRECTORY mode; per-file outputs will be used.")

    # Process each features file independently
    for ff in features_files:
        out_path_default = _result_path_for_features_file(ff, results_dir)

        # In directory mode, skip whole file if already complete
        if is_dir_mode and _result_is_complete(out_path_default, _all_expected_keys(_load_features_by_layers_compat(ff))):
            print(f"[SKIP] Already complete: {os.path.basename(out_path_default)}")
            continue

        # SINGLE-FILE mode + --cache_path override: still supports resume by reading that file if exists
        if (not is_dir_mode) and (args.cache_path is not None):
            _ensure_parent_dir(args.cache_path)
            # Run with default out path first (we want consistent naming in dir), then move/overwrite.
            _run_steering_for_one_features_file(
                textgen=textgen,
                tokenizer=tokenizer,
                device=args.device,
                sae_root_dir=args.sae_root_dir,
                features_file=ff,
                results_dir=results_dir,
                amp_factor=args.amp_factor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                sae_trainer=args.sae_trainer,
                sae_k_global=args.sae_k,
                n_prefix=args.n_prefix,
                seed=args.seed,
            )
            # Move/overwrite to cache_path
            if os.path.exists(out_path_default):
                os.replace(out_path_default, args.cache_path)
                print(f"[INFO] Moved result to --cache_path: {args.cache_path}")
            else:
                print(f"[WARN] Expected result not found at {out_path_default}; nothing moved.")
        else:
            _run_steering_for_one_features_file(
                textgen=textgen,
                tokenizer=tokenizer,
                device=args.device,
                sae_root_dir=args.sae_root_dir,
                features_file=ff,
                results_dir=results_dir,
                amp_factor=args.amp_factor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                sae_trainer=args.sae_trainer,
                sae_k_global=args.sae_k,
                n_prefix=args.n_prefix,
                seed=args.seed,
            )

    print("[ALL DONE] Steering for all features file(s) completed.")


if __name__ == "__main__":
    main()
