# autointerp_hf/run_dlm_eval.py
import argparse
import asyncio
import json
from typing import Tuple, List, Dict
import os
import sys
from dataclasses import replace
import re
import time
import random

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)

from .config import AutoInterpEvalConfig
from .data_utils import load_and_tokenize_dataset
from .hooks import get_feature_activation_sparsity_hf
from .autointerp import AutoInterpRunner
from .judge import AsyncOpenAIJudge, OpenAIJudgeConfig
from .eval_output import build_eval_output
from .utils import load_sae


# ---------------------------
# Dtype / IO helpers
# ---------------------------

def str_to_torch_dtype(s: str) -> torch.dtype:
    if s == "float32":
        return torch.float32
    elif s == "float16":
        return torch.float16
    elif s == "bfloat16":
        return torch.bfloat16
    elif s == "auto":
        return torch.float32
    else:
        raise ValueError(f"Unsupported torch_dtype string {s}")


def atomic_json_write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda x: getattr(x, "__dict__", str(x)))
    os.replace(tmp, path)


def load_checkpoint(ckpt_path: str) -> Dict[int, dict]:
    if not os.path.exists(ckpt_path):
        return {}
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        latents = data.get("latents", {})
        out: Dict[int, dict] = {}
        for k, v in latents.items():
            try:
                out[int(k)] = v
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[WARN] Failed to read checkpoint {ckpt_path}: {e}")
        return {}


def save_checkpoint(ckpt_path: str, meta: dict, latents_results: Dict[int, dict]) -> None:
    serializable = {
        "meta": meta,
        "latents": {str(k): v for k, v in latents_results.items()}
    }
    atomic_json_write(ckpt_path, serializable)


# ---------------------------
# Reproducibility
# ---------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy (if available), and PyTorch for reproducibility."""
    import os as _os
    import random as _random
    _os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    try:
        import numpy as _np  # optional
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------
# File discovery / naming
# ---------------------------

def find_sae_pt_files(path: str) -> List[str]:
    """
    Collect all 'ae.pt' files under a directory (recursive), or return [path] if a single file.
    """
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"[ERROR] --sae_path '{path}' is neither a file nor a directory")

    pt_files: List[str] = []
    for root, _, files in os.walk(path):
        for fname in files:
            if fname == "ae.pt":
                pt_files.append(os.path.join(root, fname))

    def sort_key(p: str) -> Tuple[int, int, str]:
        layer = 10**9
        trainer = 10**9
        parts = p.replace("\\", "/").split("/")
        for seg in parts:
            if seg.startswith("resid_post_layer_"):
                try:
                    layer = int(seg.split("_")[-1])
                except Exception:
                    pass
            if seg.startswith("trainer_"):
                try:
                    trainer = int(seg.split("_")[-1])
                except Exception:
                    pass
        return (layer, trainer, p)

    pt_files.sort(key=sort_key)
    return pt_files


def _canonicalize_dlm_model_basename(lm_name: str) -> str:
    """
    Canonicalize DLM model short name for filenames.

    Examples
    --------
    "Dream-org/Dream-v0-Base-7B"  -> "dream_7b"
    "Dream-org/Dream-v1-Base-13B" -> "dream_13b"
    Fallback to "dream" if no size token is found.
    """
    base = lm_name.split("/")[-1].lower().replace("-", "_")
    # Prefer the last "<digits><b/B>" occurrence (e.g., '..._7b'); this avoids taking "v0" -> "0b".
    matches = list(re.finditer(r'(\d+)\s*[bB]\b', base))
    if matches:
        size = matches[-1].group(1)
        return f"dream_{size}b"
    if "dream" in base:
        return "dream"
    return base


def derive_results_path_from_sae(sae_pt_path: str, cfg: AutoInterpEvalConfig) -> str:
    """
    Save under:
      autointerp_hf/dlm_sae_interp_results/<model>_layer<L>_l0_<k>.json
    """
    sae_dir = os.path.dirname(sae_pt_path)
    sae_cfg_path = os.path.join(sae_dir, "config.json")

    layer_val = None
    k_val = None
    lm_name_from_cfg = cfg.model_name_or_path  # fallback

    try:
        with open(sae_cfg_path, "r", encoding="utf-8") as cf:
            sae_cfg_json = json.load(cf)
        trainer_block = sae_cfg_json.get("trainer", {})
        layer_val = trainer_block.get("layer", None)
        k_val = trainer_block.get("k", None)
        lm_name_from_cfg = trainer_block.get("lm_name", lm_name_from_cfg)
    except Exception as e:
        print(f"[WARN] could not read SAE config.json at {sae_cfg_path}: {e}")

    if layer_val is None:
        try:
            layer_val = int(cfg.hook_module_path.split(".")[-1])
        except Exception:
            layer_val = "unknown"
    if k_val is None:
        k_val = "unknown"

    base_model_name = _canonicalize_dlm_model_basename(lm_name_from_cfg)
    auto_filename = f"{base_model_name}_layer{layer_val}_l0_{k_val}.json"

    this_dir = os.path.dirname(__file__)
    results_dir = os.path.join(this_dir, "dream_mask_sae_interp_results") # changed folder name
    os.makedirs(results_dir, exist_ok=True)

    final_out_path = os.path.join(results_dir, auto_filename)
    return final_out_path


def extract_layer_from_sae(sae_pt_path: str) -> int | None:
    """
    Prefer config.json['trainer']['layer']; fallback to 'resid_post_layer_<L>' in the path.
    """
    sae_dir = os.path.dirname(sae_pt_path)
    cfg_path = os.path.join(sae_dir, "config.json")

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        layer = js.get("trainer", {}).get("layer", None)
        if isinstance(layer, int):
            return layer
    except Exception:
        pass

    parts = sae_pt_path.replace("\\", "/").split("/")
    for seg in parts:
        if seg.startswith("resid_post_layer_"):
            tail = seg.rsplit("_", 1)[-1]
            try:
                return int(tail)
            except Exception:
                continue
    return None


def compute_hook_path(base_hook: str, layer: int) -> str:
    """
    Replace the last numeric segment in a dotted module path with <layer>,
    or if base_hook is 'auto' / '' / None, use 'model.layers.<layer>'.
    """
    if base_hook in ("auto", "", None):
        return f"model.layers.{layer}"
    parts = base_hook.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            parts[i] = str(layer)
            return ".".join(parts)
    return f"model.layers.{layer}"


# ---------------------------
# Single-SAE runner with resume & chunked latents
# ---------------------------

async def run_single_sae_and_save(
    sae_pt_path: str,
    cfg: AutoInterpEvalConfig,
    tokenizer,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    judge: AsyncOpenAIJudge,
    latent_batch_size: int,
) -> None:
    """
    Run one SAE with checkpoint/resume for DLMs.
    - Skips if final JSON exists
    - Otherwise reads <final>.ckpt.json and resumes remaining latents
    - Evaluates in chunks (latent_batch_size) to reuse activation collection
    - Writes checkpoint after each chunk; writes final JSON when done
    Note: attention_mask is expected to already be bool + [B,1,1,L] in main().
    """
    print(f"\n[INFO] ===== Processing SAE: {sae_pt_path} =====")

    layer = extract_layer_from_sae(sae_pt_path)
    if layer is not None:
        per_sae_hook_path = compute_hook_path(cfg.hook_module_path, layer)
        if per_sae_hook_path != cfg.hook_module_path:
            print(f"[INFO] Overriding hook_module_path: {cfg.hook_module_path} -> {per_sae_hook_path}")
        cfg_local = replace(cfg, hook_module_path=per_sae_hook_path)
    else:
        print("[WARN] Could not infer layer for this SAE; using base hook_module_path unchanged.")
        cfg_local = cfg

    final_out_path = derive_results_path_from_sae(sae_pt_path, cfg_local)
    ckpt_path = final_out_path + ".ckpt.json"

    if os.path.exists(final_out_path):
        print(f"[SKIP] Final result already exists: {final_out_path}")
        return

    partial_results: Dict[int, dict] = load_checkpoint(ckpt_path)
    already_done = set(partial_results.keys())
    print(f"[INFO] Already completed latents in checkpoint: {len(already_done)}")

    sae_torch_dtype = str_to_torch_dtype(cfg_local.torch_dtype)
    sae = load_sae(sae_pt_path, cfg_local.device, sae_torch_dtype)
    sae = sae.to(device=cfg_local.device, dtype=sae_torch_dtype)  # type: ignore
    sae.eval()  # type: ignore

    assert cfg_local.batch_size is not None, "Please provide --batch_size"
    sparsity = get_feature_activation_sparsity_hf(
        input_ids=input_ids,
        attention_mask=attention_mask,  # already bool + [B,1,1,L]
        model=model,
        sae=sae,
        batch_size=cfg_local.batch_size,
        hook_module_path=cfg_local.hook_module_path,
        tokenizer=tokenizer,
    )

    activation_counts = sparsity * cfg_local.total_tokens
    alive_latents = (
        torch.nonzero(activation_counts > cfg_local.dead_latent_threshold).squeeze(1).tolist()
    )
    if len(alive_latents) == 0:
        print("[WARN] No alive latents; skipping this SAE.")
        return

    need_total = cfg_local.n_latents
    already = len(already_done)
    if already >= need_total:
        print(f"[INFO] Checkpoint has {already} latents >= requested {need_total}; finalizing...")
        sae_metadata = {
            "sae_path": sae_pt_path,
            "hook_module_path": cfg_local.hook_module_path,
        }
        eval_output = build_eval_output(
            eval_config={"cfg": cfg_local.__dict__},
            model_name_or_path=cfg_local.model_name_or_path,
            hook_module_path=cfg_local.hook_module_path,
            sae_metadata=sae_metadata,
            results_dict=partial_results,
        )
        atomic_json_write(final_out_path, eval_output)
        try:
            os.remove(ckpt_path)
        except FileNotFoundError:
            pass
        print(f"[DONE] Saved eval results to {final_out_path}")
        metrics = eval_output.get("metrics", {})
        print(f"[INFO] Mean AutoInterp score = {metrics.get('autointerp_score', 0.0):.4f}, "
              f"StdDev = {metrics.get('autointerp_std_dev', 0.0):.4f}")
        return

    remaining_needed = need_total - already
    candidates = [i for i in alive_latents if i not in already_done]
    if len(candidates) == 0:
        print("[WARN] No remaining alive latents to evaluate; finalizing what we have.")
        sae_metadata = {
            "sae_path": sae_pt_path,
            "hook_module_path": cfg_local.hook_module_path,
        }
        eval_output = build_eval_output(
            eval_config={"cfg": cfg_local.__dict__},
            model_name_or_path=cfg_local.model_name_or_path,
            hook_module_path=cfg_local.hook_module_path,
            sae_metadata=sae_metadata,
            results_dict=partial_results,
        )
        atomic_json_write(final_out_path, eval_output)
        try:
            os.remove(ckpt_path)
        except FileNotFoundError:
            pass
        print(f"[DONE] Saved eval results to {final_out_path}")
        return

    k = min(remaining_needed, len(candidates))
    random.shuffle(candidates)
    latents_to_run = candidates[:k]

    meta = {
        "sae_path": sae_pt_path,
        "hook_module_path": cfg_local.hook_module_path,
        "target_n_latents": cfg_local.n_latents,
        "timestamp": int(time.time()),
    }

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Run by chunks to reuse activation collection per chunk
    for chunk_idx, chunk in enumerate(_chunks(latents_to_run, latent_batch_size), start=1):
        print(f"[INFO] Running latent chunk {chunk_idx}: {chunk} (size={len(chunk)}) ...")

        # Build a per-chunk config and set latents post-init to avoid init=False error
        cfg_batch = replace(cfg_local)
        cfg_batch.latents = chunk  # type: ignore[attr-defined]

        runner = AutoInterpRunner(
            cfg=cfg_batch,
            model=model,
            sae=sae,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sparsity=sparsity,
            judge=judge,
        )

        # One run collects activations once for the whole chunk, then returns {latent_id: {...}}
        results_chunk = await runner.run()

        if results_chunk:
            for lid, res in results_chunk.items():
                partial_results[int(lid)] = res
            save_checkpoint(ckpt_path, meta, partial_results)

        # Free a bit of memory
        try:
            del runner
            if torch.cuda.is_available() and "cuda" in str(cfg_local.device):
                torch.cuda.empty_cache()
        except Exception:
            pass

    if len(partial_results) >= cfg_local.n_latents:
        sae_metadata = {
            "sae_path": sae_pt_path,
            "hook_module_path": cfg_local.hook_module_path,
        }
        eval_output = build_eval_output(
            eval_config={"cfg": cfg_local.__dict__},
            model_name_or_path=cfg_local.model_name_or_path,
            hook_module_path=cfg_local.hook_module_path,
            sae_metadata=sae_metadata,
            results_dict=partial_results,
        )
        atomic_json_write(final_out_path, eval_output)
        try:
            os.remove(ckpt_path)
        except FileNotFoundError:
            pass
        print(f"[DONE] Saved eval results to {final_out_path}")
        metrics = eval_output.get("metrics", {})
        print(f"[INFO] Mean AutoInterp score = {metrics.get('autointerp_score', 0.0):.4f}, "
              f"StdDev = {metrics.get('autointerp_std_dev', 0.0):.4f}")
    else:
        print(f"[PAUSE] Checkpoint saved ({len(partial_results)}/{cfg_local.n_latents}). "
              f"Re-run the same command to resume.")


# ---------------------------
# MAIN
# ---------------------------

async def main(args):
    # Seed for reproducibility
    seed = getattr(args, "seed", 3407)
    set_global_seed(seed)
    print(f"[INFO] Global random seed set to {seed}")

    no_scoring = getattr(args, "no_scoring", False)
    no_demos = getattr(args, "no_demos", False)
    debug_judge = getattr(args, "debug_judge", False)

    cfg = AutoInterpEvalConfig(
        model_name_or_path=args.model_name_or_path,
        hook_module_path=args.hook_module_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        n_latents=args.n_latents,
        dead_latent_threshold=args.dead_latent_threshold,
        dataset_name=args.dataset_name,
        total_tokens=args.total_tokens,
        llm_context_size=args.context_length,
        batch_size=args.batch_size,
        scoring=(not no_scoring),
        use_demos_in_explanation=(not no_demos),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
    )

    torch_dtype_for_model = (
        None if cfg.torch_dtype == "auto" else str_to_torch_dtype(cfg.torch_dtype)
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype_for_model,
        )
    except Exception:
        model = AutoModel.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype_for_model,
        )
    model = model.to(cfg.device)
    model.eval()

    # Tokenize dataset; normalize attention mask to bool + [B,1,1,L] for DLM compatibility.
    input_ids, attention_mask = load_and_tokenize_dataset(
        dataset_name=cfg.dataset_name,
        context_length=cfg.llm_context_size,
        total_tokens=cfg.total_tokens,
        tokenizer=tokenizer,
        device=cfg.device,
    )
    attention_mask = attention_mask.to(dtype=torch.bool)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,L] -> [B,1,1,L]

    judge_cfg = OpenAIJudgeConfig(
        model=args.judge_model,
        base_url=args.judge_base_url,
        timeout=args.judge_timeout,
        max_retries=args.judge_max_retries,
        debug=debug_judge,
        debug_truncate=800,
    )
    judge = AsyncOpenAIJudge(judge_cfg)

    sae_targets = find_sae_pt_files(args.sae_path)
    print(f"[INFO] Discovered {len(sae_targets)} SAE(s) to evaluate.")
    if len(sae_targets) == 0:
        raise RuntimeError(f"No 'ae.pt' found under: {args.sae_path}")

    for idx, sae_pt in enumerate(sae_targets, start=1):
        print(f"[INFO] ({idx}/{len(sae_targets)}) -> {sae_pt}")
        try:
            await run_single_sae_and_save(
                sae_pt_path=sae_pt,
                cfg=cfg,
                tokenizer=tokenizer,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                judge=judge,
                latent_batch_size=args.latent_batch_size,
            )
        except Exception as e:
            print(f"[ERROR] failed on SAE: {sae_pt}\n{e}")

    print("\n[ALL DONE]")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=(
            "Run auto-interpretability for DLMs (HF hooks). "
            "Pass --sae_path as either a single ae.pt file OR a directory; "
            "if directory, the script will recursively evaluate all trainer_*/ae.pt within. "
            "Results are saved under autointerp_hf/dlm_sae_interp_results/. "
            "Checkpoint files use the suffix '.ckpt.json'."
        )
    )

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument(
        "--hook_module_path",
        type=str,
        required=True,
        help="Module path template, e.g. 'model.layers.1'. Or 'auto' to map to 'model.layers.<layer>' per SAE.",
    )
    p.add_argument("--sae_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )

    p.add_argument("--dataset_name", type=str, default="monology/pile-uncopyrighted")
    p.add_argument("--total_tokens", type=int, default=2_000_000)
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--n_latents", type=int, default=100)
    p.add_argument("--dead_latent_threshold", type=float, default=15)

    p.add_argument(
        "--latent_batch_size",
        type=int,
        default=16,
        help="How many latents to evaluate in one runner call to reuse activation collection. Larger is faster but uses more memory."
    )

    p.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    p.add_argument("--judge_base_url", type=str, default="https://api.shubiaobiao.cn/v1")
    p.add_argument("--judge_timeout", type=float, default=60.0)
    p.add_argument("--judge_max_retries", type=int, default=3)
    p.add_argument("--debug_judge", action="store_true")

    p.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility (Python/NumPy/Torch)."
    )

    p.add_argument("--no_scoring", action="store_true")
    p.add_argument("--no_demos", action="store_true")

    # Kept for backward compatibility; ignored when --sae_path is a directory.
    p.add_argument(
        "--output_json",
        type=str,
        default="autointerp_eval_results.json",
        help="Legacy/optional; final filename is auto-derived when processing directories.",
    )

    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
