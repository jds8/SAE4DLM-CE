# dlm_steer.py
# -*- coding: utf-8 -*-
"""
DLM (Dream-7B) steering with Sparse Autoencoders (SAEs).

Key differences vs AR-LLM:
- Dream is diffusion/denoising over the whole sequence, multiple steps.
- We steer by amplifying SAE features across positions on each forward pass, not just the last token.

This script:
- Loads per-(layer,k) feature files (new format: { "<layer>": { "<id>": {"explanation","score"}, ... } }).
- For each feature, selects n neutral prefixes, runs BASELINE vs STEERED generation using Dream's
  official diffusion_generate calling convention, and saves comparisons.
- Supports resume: already-finished (layer_feature) keys are skipped, fully-complete result files are skipped.

Steering scope:
  --token_scope all           -> amplify selected SAE features at ALL positions each step
  --token_scope topk_tokens   -> amplify only on top-K positions (per batch) where selected features are most active
"""

import gc
import os
import re
import json
import time
import argparse
import random
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import (
    get_sae,                         # local SAE loader
    try_get_final_norm_and_lm_head,  # sanity printing
)
from sae_utils import _resolve_layers_container  # locate decoder blocks


# -----------------------------
# Global defaults
# -----------------------------
DEFAULT_BASE_RESULTS_ROOT = "/home/dslabra5/sae4dlm/steering/steering_results_file"


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', type=str, default="Dream-org/Dream-v0-Base-7B",
                    help="HuggingFace repo/path for the Dream (DLM) model.")
    ap.add_argument('--features_file', type=str, required=True,
                    help="Path to a single features file OR a directory of such files.")
    ap.add_argument('--sae_root_dir', type=str, required=True,
                    help="Base dir with resid_post_layer_{L}/trainer_*/ (ae.pt, config.json) for the DLM.")
    ap.add_argument('--sae_trainer', type=str, default=None,
                    help="Pick a specific trainer folder name (e.g., 'trainer_0'). Overrides --sae_k if set.")
    ap.add_argument('--sae_k', type=int, default=None,
                    help="Pick trainer by config.trainer.k == this value.")
    ap.add_argument('--amp_factor', type=float, default=2.0,
                    help="Feature amplification multiplier (relative to per-position max act).")
    ap.add_argument('--device', type=str, default="cuda:0")
    ap.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help=(
            "Folder to write results. "
            "Default: under /home/dslabra5/sae4dlm/steering/steering_results_file, "
            "automatically choosing a subfolder based on --sae_root_dir: "
            "dream_mask_sae_steering_results for mask SAEs, "
            "dream_unmask_sae_steering_results for unmask SAEs."
        ),
    )
    ap.add_argument('--cache_path', type=str, default=None,
                    help="ONLY in single-file mode: override output JSON path.")
    ap.add_argument('--dtype', type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"],
                    help="Model dtype. Dream often benefits from bf16.")
    # DLM sampling knobs (official-style)
    ap.add_argument('--dlm_steps', type=int, default=20, help="Number of denoising steps (passed as 'steps').")
    ap.add_argument('--max_new_tokens', type=int, default=512, help="Max new tokens to generate.")
    ap.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    ap.add_argument('--top_p', type=float, default=0.95, help="Nucleus sampling top_p.")
    ap.add_argument('--alg', type=str, default="entropy", help="Dream generation algorithm name.")
    ap.add_argument('--alg_temp', type=float, default=0.0, help="Algorithm temperature for Dream (e.g., entropy).")
    ap.add_argument('--do_sample', action='store_true', default=True, help="Sampling flag if backend uses it.")
    # Neutral prefixes
    ap.add_argument('--n_prefix', type=int, default=5, help="Randomly select n neutral prefixes per feature.")
    ap.add_argument('--seed', type=int, default=42, help="Global seed (prefix/sample reproducibility).")
    # Steering scope in sequence
    ap.add_argument('--token_scope', type=str, default="all", choices=["all", "topk_tokens"],
                    help="Where to apply feature amplification across the sequence.")
    ap.add_argument('--topk_positions', type=int, default=3,
                    help="If token_scope=topk_tokens, amplify only on top-K positions per batch.")
    ap.add_argument('--trust_remote_code', action='store_true', default=True)
    return ap.parse_args()


# -----------------------------
# Utilities
# -----------------------------
FEATURES_RE_STRICT_QWEN = re.compile(r"^features_qwen2\.5_layer(\d+)_l0_(\d+)\.json$")
FEATURES_RE_STRICT_DREAM = re.compile(r"^features_dream7b_layer(\d+)_l0_(\d+)\.json$")
FEATURES_RE_RELAXED = re.compile(r"^.*layer(\d+).*_l0_(\d+)\.json$")


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
    Determine the default results directory based on sae_root_dir.

    - If sae_root_dir contains 'saes_mask_Dream-org_Dream-v0-Base-7B_top_k',
      use 'dream_mask_sae_steering_results'.
    - If sae_root_dir contains 'saes_unmask_Dream-org_Dream-v0-Base-7B_top_k',
      use 'dream_unmask_sae_steering_results'.
    - Otherwise fall back to 'dream_sae_steering_results'.
    """
    base = DEFAULT_BASE_RESULTS_ROOT
    sae_root_dir = os.path.abspath(sae_root_dir)

    subdir = "dream_sae_steering_results"
    if "saes_mask_Dream-org_Dream-v0-Base-7B_top_k" in sae_root_dir:
        subdir = "dream_mask_sae_steering_results"
    elif "saes_unmask_Dream-org_Dream-v0-Base-7B_top_k" in sae_root_dir:
        subdir = "dream_unmask_sae_steering_results"

    return os.path.join(base, subdir)


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _infer_layer_k_from_features_file(path: str) -> Tuple[Optional[int], Optional[int]]:
    fname = os.path.basename(path)
    for pat in (FEATURES_RE_STRICT_DREAM, FEATURES_RE_STRICT_QWEN, FEATURES_RE_RELAXED):
        m = pat.match(fname)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _gather_features_files(path: str) -> List[str]:
    if os.path.isfile(path):
        L, K = _infer_layer_k_from_features_file(path)
        if L is None or K is None:
            raise ValueError(f"Features file name not recognized: {path}")
        return [path]
    if os.path.isdir(path):
        out = []
        for fname in sorted(os.listdir(path)):
            if (FEATURES_RE_STRICT_DREAM.match(fname) or
                FEATURES_RE_STRICT_QWEN.match(fname) or
                FEATURES_RE_RELAXED.match(fname)):
                out.append(os.path.join(path, fname))
        if not out:
            raise FileNotFoundError(f"No valid features files found under directory: {path}")
        return out
    raise FileNotFoundError(f"features_file path not found: {path}")


def _result_path_for_features_file(features_file: str, results_dir: str) -> str:
    L, K = _infer_layer_k_from_features_file(features_file)
    if L is not None and K is not None:
        base = f"steer_dlm_layer{L}_l0_{K}.json"
    else:
        base = f"steer_dlm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    return os.path.join(results_dir, base)


def _seed_everywhere(seed: int):
    import random as _random
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_prefixes_per_feature(all_prefixes: List[str], n_prefix: int,
                               seed: Optional[int], layer: int, feature: int) -> List[str]:
    n = max(1, min(n_prefix, len(all_prefixes)))
    if seed is None:
        return random.sample(all_prefixes, n)
    rng = random.Random((hash((seed, layer, feature)) & 0xFFFFFFFF))
    return rng.sample(all_prefixes, n)


def _load_features_by_layers_compat(path: str) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[int, List[int]] = {}
    for k, v in obj.items():
        if not (isinstance(k, str) and k.isdigit()):
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
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    details_by_layer: Dict[int, Dict[int, Dict[str, object]]] = {}
    for k, v in obj.items():
        if not (isinstance(k, str) and k.isdigit()):
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
    # minimal compat for old "__details__" if present
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


def _all_expected_keys(features_by_layers: Dict[int, List[int]]) -> List[str]:
    keys = []
    for layer, feats in features_by_layers.items():
        for f in feats:
            keys.append(f"{layer}_{f}")
    return keys


def _result_is_complete(out_path: str, expected_keys: List[str]) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return False
    return set(expected_keys).issubset(set(obj.keys()))


# -----------------------------
# DLM hook (amplify across positions)
# -----------------------------
class AmplifySAEHookDLM:
    """
    Forward hook for DLM layers:
    - Works on [B, D] (expanded) or [B, T, D] tensors.
    - Amplifies selected SAE features across positions according to `token_scope`.
    - Residual error compensation: add (hidden - clean_recon) back to preserve baseline content.
    """
    def __init__(self, sae, feature_ids, amp_factor: float, device: str,
                 token_scope: str = "all", topk_positions: int = 4) -> None:
        self.sae = sae
        self.features = [int(f) for f in feature_ids]
        self.amp = float(amp_factor)
        self.device = torch.device(device)
        self.token_scope = token_scope
        self.topk_positions = int(topk_positions)

    def __call__(self, module, args, output):
        # unpack output
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            others = list(output[1:])
        else:
            hidden = output
            others = None

        if not torch.is_tensor(hidden):
            return output

        orig_device = hidden.device
        orig_dtype = hidden.dtype
        squeezed = False

        if hidden.ndim == 2:  # [B, D] -> [B, 1, D]
            hidden = hidden.unsqueeze(1)
            squeezed = True
        elif hidden.ndim != 3:
            return output  # unknown shape, do not modify

        # move to SAE device
        if hidden.device != self.device:
            hidden = hidden.to(self.device)

        # SAE encode
        acts = self.sae.encode(hidden)  # [B, T, F]

        # Clean reconstruction for residual error compensation
        with torch.no_grad():
            clean_acts = self.sae.encode(hidden)
            clean_rec = self.sae.decode(clean_acts)  # [B, T, D]

        B, T, F = acts.shape
        if F == 0:
            x_hat = self.sae.decode(acts)
        else:
            idx = torch.tensor(self.features, device=acts.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < F)]
            if idx.numel() > 0:
                if self.token_scope == "all":
                    per_pos_max = acts.amax(dim=-1, keepdim=True)  # [B, T, 1]
                    boost = per_pos_max * self.amp                # [B, T, 1]
                    K = idx.numel()
                    idx_exp = idx.view(1, 1, K).expand(B, T, K)
                    src = boost.expand(B, T, K)                   # [B, T, K]
                    acts = acts.scatter_add(dim=-1, index=idx_exp, src=src)
                elif self.token_scope == "topk_tokens":
                    sel = acts[..., idx]             # [B, T, K]
                    pos_score = sel.amax(dim=-1)     # [B, T]
                    Kpos = max(1, min(self.topk_positions, T))
                    _, topk_idx = torch.topk(pos_score, k=Kpos, dim=1)  # [B, Kpos]
                    mask = torch.zeros((B, T), dtype=torch.bool, device=acts.device)
                    mask.scatter_(dim=1, index=topk_idx, value=True)
                    per_pos_max = acts.amax(dim=-1, keepdim=True)  # [B, T, 1]
                    boost = per_pos_max * self.amp
                    boost = boost * mask.unsqueeze(-1)  # mask positions
                    K = idx.numel()
                    idx_exp = idx.view(1, 1, K).expand(B, T, K)
                    src = boost.expand(B, T, K)
                    acts = acts.scatter_add(dim=-1, index=idx_exp, src=src)
                else:
                    pass

            x_hat = self.sae.decode(acts)

        # residual error compensation (preserve baseline content)
        x_hat = x_hat + (hidden.to(torch.float32) - clean_rec.to(torch.float32))
        x_hat = x_hat.to(orig_dtype)

        if squeezed:
            x_hat = x_hat.squeeze(1)
        if x_hat.device != orig_device:
            x_hat = x_hat.to(orig_device)

        if others is not None:
            return tuple([x_hat] + others)
        else:
            return x_hat


def _register_dlm_hook(model, sae, layer: int, feature: int, device: str,
                       amp_factor: float, token_scope: str, topk_positions: int):
    """
    Register the DLM amplify hook on the given Transformer block index (resid_post).
    """
    hook = AmplifySAEHookDLM(
        sae=sae,
        feature_ids=[feature],
        amp_factor=amp_factor,
        device=device,
        token_scope=token_scope,
        topk_positions=topk_positions,
    )
    layers_container = _resolve_layers_container(model)
    block = layers_container[layer]
    handle = block.register_forward_hook(hook, always_call=True)
    return handle


# -----------------------------
# Dream generation wrappers
# -----------------------------
@torch.inference_mode()
def _dlm_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    *,
    steps: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    do_sample: bool = True,
    **extra_kwargs,
) -> str:
    """
    Tokenize the prompt (base format), prefer attention_mask=None fast path; fallback to all-ones mask if needed.
    Decode ONLY the newly generated part (drop the prompt).
    """
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    # Prefer the fast path: try attention_mask=None when tokenizer did not provide one
    use_mask_none = (attn is None)
    if not use_mask_none:
        attn = attn.to(device)

    prompt_len = input_ids.shape[1]

    # Try full-arg call first
    try:
        out = model.diffusion_generate(
            input_ids,
            attention_mask=(None if use_mask_none else attn),
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            do_sample=do_sample,
            **extra_kwargs,
        )
    except TypeError:
        # Fallback: if backend does not accept None mask, force an all-ones mask
        if use_mask_none:
            attn = torch.ones_like(input_ids, device=input_ids.device)
        # Also drop optional flags to be safe
        out = model.diffusion_generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
        )

    seq = out.sequences if hasattr(out, "sequences") else out.get("sequences", None)
    if isinstance(seq, torch.Tensor):
        full_ids = seq[0]  # decode full sequence (includes the prefix)
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        eos = getattr(tokenizer, "eos_token", None)
        return text.split(eos)[0] if (eos and eos in text) else text
    if isinstance(seq, list) and len(seq) and torch.is_tensor(seq[0]):
        full_ids = seq[0]  # decode full sequence (includes the prefix)
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        eos = getattr(tokenizer, "eos_token", None)
        return text.split(eos)[0] if (eos and eos in text) else text


@torch.inference_mode()
def _dlm_generate_preencoded(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    prompt_len: int,
    steps: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
    do_sample: bool,
    **extra_kwargs,
) -> str:
    """
    Faster path: reuse already tokenized & moved-to-device tensors.
    Prefer attention_mask=None when possible; fallback to all-ones if backend complains.
    Decode ONLY tokens after the prompt.
    """
    # input_ids / attention_mask are expected on device already
    use_mask_none = (attention_mask is None)
    try:
        out = model.diffusion_generate(
            input_ids,
            attention_mask=(None if use_mask_none else attention_mask),
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            do_sample=do_sample,
            **extra_kwargs,
        )
    except TypeError:
        if use_mask_none:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        out = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
        )

    seq = out.sequences if hasattr(out, "sequences") else out.get("sequences", None)
    if isinstance(seq, torch.Tensor):
        full_ids = seq[0]  # decode full sequence (includes the prefix)
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        eos = getattr(tokenizer, "eos_token", None)
        return text.split(eos)[0] if (eos and eos in text) else text
    if isinstance(seq, list) and len(seq) and torch.is_tensor(seq[0]):
        full_ids = seq[0]  # decode full sequence (includes the prefix)
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        eos = getattr(tokenizer, "eos_token", None)
        return text.split(eos)[0] if (eos and eos in text) else text


# -----------------------------
# Main steering logic (with resume)
# -----------------------------
def _run_dlm_steering_for_one_file(
    model,
    tokenizer,
    device: str,
    sae_root_dir: str,
    features_file: str,
    results_dir: str,
    amp_factor: float,
    dlm_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
    do_sample: bool,
    sae_trainer: Optional[str],
    sae_k_global: Optional[int],
    n_prefix: int,
    seed: Optional[int],
    token_scope: str,
    topk_positions: int,
):
    out_path = _result_path_for_features_file(features_file, results_dir)
    _ensure_parent_dir(out_path)

    features_by_layers = _load_features_by_layers_compat(features_file)
    details_by_layer = _load_feature_details(features_file)
    expected_keys = _all_expected_keys(features_by_layers)

    # Skip if complete
    if _result_is_complete(out_path, expected_keys):
        print(f"[SKIP] Completed: {os.path.basename(out_path)}")
        return

    print(f"[INFO] Results -> {out_path}")
    print(f"[INFO] Loaded {len(features_by_layers)} layer(s) from {features_file}")

    # pick k from filename if needed
    _, inf_k = _infer_layer_k_from_features_file(features_file)
    local_k = sae_k_global if sae_k_global is not None else inf_k
    if sae_trainer is None and inf_k is not None and local_k == inf_k:
        print(f"[INFO] Inferred SAE L0 (k): {local_k}")

    # resume
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    # prefixes pool
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

    # SAE cache per file
    saes: Dict[int, nn.Module] = {}

    for layer, feats in features_by_layers.items():
        # Load SAE (once per layer) and keep on target device
        saes[layer] = get_sae(
            model_type="dlm",  # informational only
            layer=layer,
            saes=saes,
            backend="dl_local",
            dl_local_dir=sae_root_dir,
            device=device,
            trainer_name=sae_trainer,
            k_topk=local_k,
        ).to(device)

        for feature in tqdm(feats, desc=f"[{os.path.basename(features_file)}] Layer {layer} features"):
            key = f"{layer}_{feature}"
            if key in cache:
                continue

            # deterministic per-feature prefixes
            prefixes = _pick_prefixes_per_feature(all_prefixes, n_prefix, seed, layer, feature)

            # details (may be absent)
            finfo = details_by_layer.get(layer, {}).get(feature, {})
            explanation = finfo.get("explanation", "")
            score = finfo.get("score", None)

            results = []
            sae = saes[layer]

            for prefix in prefixes:
                # ---- Pre-tokenize ONCE for both baseline & steered; move to device here ----
                enc = tokenizer(prefix, return_tensors="pt")
                enc_input_ids = enc["input_ids"].to(device)
                enc_attn = enc.get("attention_mask")
                if enc_attn is not None:
                    enc_attn = enc_attn.to(device)
                prompt_len = enc_input_ids.shape[1]

                # baseline
                if seed is not None:
                    _seed_everywhere(hash((layer, feature, prefix, "baseline", seed)) & 0xFFFFFFFF)
                baseline_text = _dlm_generate_preencoded(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=enc_input_ids,
                    attention_mask=enc_attn,       # None -> fast path, else use provided
                    prompt_len=prompt_len,
                    steps=dlm_steps,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    alg=alg,
                    alg_temp=alg_temp,
                    do_sample=do_sample,
                )

                # steered (attach hook only during this call)
                handle = _register_dlm_hook(
                    model=model,
                    sae=sae,
                    layer=layer,
                    feature=feature,
                    device=device,
                    amp_factor=amp_factor,
                    token_scope=token_scope,
                    topk_positions=topk_positions,
                )
                try:
                    if seed is not None:
                        _seed_everywhere(hash((layer, feature, prefix, "steered", seed)) & 0xFFFFFFFF)
                    steered_text = _dlm_generate_preencoded(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=enc_input_ids,     # reuse the same tensors
                        attention_mask=enc_attn,
                        prompt_len=prompt_len,
                        steps=dlm_steps,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        alg=alg,
                        alg_temp=alg_temp,
                        do_sample=do_sample,
                    )
                finally:
                    handle.remove()

                results.append({
                    "prefix": prefix,
                    "without_steer_output": baseline_text,
                    "after_steer_output": steered_text,
                })

            # append & flush (resume-friendly)
            cache[key] = {
                "explanation": explanation,
                "score": score,
                "comparisons": results,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        # free layer SAE
        saes[layer] = saes[layer].cpu()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[DONE] DLM steering complete/updated for {os.path.basename(features_file)} -> {out_path}")


# -----------------------------
# Build model
# -----------------------------
def _build_dlm_model(model_name: str, dtype, device: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=(None if dtype == "auto" else dtype),  # model impl warns against torch_dtype; use dtype here
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device).eval()
    print(f"Device set to use {device}")
    # sanity prints
    try:
        final_layer_norm, lm_head = try_get_final_norm_and_lm_head(model)
        ln_shape = getattr(getattr(final_layer_norm, "weight", None), "shape", None)
        print("[CHECK] final_layer_norm:", type(final_layer_norm), "weight shape:", ln_shape)
        if lm_head is not None:
            print("[CHECK] lm_head:", type(lm_head), "weight shape:",
                  getattr(getattr(lm_head, "weight", None), "shape", None))
    except Exception as e:
        print("[WARN] final norm / head resolution failed:", repr(e))
    layers = _resolve_layers_container(model)
    print("[CHECK] layers_container:", type(layers), "num layers:", len(layers))
    if len(layers) > 0:
        print("[CHECK] first blocks:", type(layers[0]), type(layers[1]) if len(layers) > 1 else None)
    return model, tokenizer


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Determine results_dir: use CLI if provided, otherwise infer from sae_root_dir
    results_dir = args.results_dir or _default_results_dir(args.sae_root_dir)
    _ensure_dir(results_dir)

    dtype = _select_dtype(args.dtype)
    model, tokenizer = _build_dlm_model(args.model_name, dtype, args.device, args.trust_remote_code)

    features_files = _gather_features_files(args.features_file)
    is_dir_mode = os.path.isdir(args.features_file)
    if is_dir_mode and args.cache_path is not None:
        print("[WARN] --cache_path is ignored in DIRECTORY mode.")

    # directory-level skip if fully complete
    for ff in features_files:
        out_path = _result_path_for_features_file(ff, results_dir)
        if is_dir_mode and _result_is_complete(out_path, _all_expected_keys(_load_features_by_layers_compat(ff))):
            print(f"[SKIP] Already complete: {os.path.basename(out_path)}")
            continue

        if (not is_dir_mode) and (args.cache_path is not None):
            # run then move to cache_path (still resume-friendly)
            _run_dlm_steering_for_one_file(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                sae_root_dir=args.sae_root_dir,
                features_file=ff,
                results_dir=results_dir,
                amp_factor=args.amp_factor,
                dlm_steps=args.dlm_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=args.alg_temp,
                do_sample=args.do_sample,
                sae_trainer=args.sae_trainer,
                sae_k_global=args.sae_k,
                n_prefix=args.n_prefix,
                seed=args.seed,
                token_scope=args.token_scope,
                topk_positions=args.topk_positions,
            )
            dst = args.cache_path
            _ensure_parent_dir(dst)
            if os.path.exists(out_path):
                os.replace(out_path, dst)
                print(f"[INFO] Moved result to --cache_path: {dst}")
            else:
                print(f"[WARN] Expected result not found at {out_path}; nothing moved.")
        else:
            _run_dlm_steering_for_one_file(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                sae_root_dir=args.sae_root_dir,
                features_file=ff,
                results_dir=results_dir,
                amp_factor=args.amp_factor,
                dlm_steps=args.dlm_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=args.alg_temp,
                do_sample=args.do_sample,
                sae_trainer=args.sae_trainer,
                sae_k_global=args.sae_k,
                n_prefix=args.n_prefix,
                seed=args.seed,
                token_scope=args.token_scope,
                topk_positions=args.topk_positions,
            )

    print("[ALL DONE] DLM steering completed for all file(s).")


if __name__ == "__main__":
    main()
