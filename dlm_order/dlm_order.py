# dlm_order.py
# -*- coding: utf-8 -*-
"""
Main pipeline for analyzing DLM inference order with SAEs.

We support fully customizable:
- DLM model name (e.g., Dream-org/Dream-v0-Base-7B)
- Official Dream generation parameters (steps, max_new_tokens, temperature, top_p, alg, alg_temp, do_sample)
- Additional custom kwargs passed to diffusion_generate (JSON string)
- SAE root path (local DL-SAE folder)
- SAE layers (e.g., 5,14,23)
- Position selection strategy (full_gen / update_only / update_plus_anchors / mask_only)
- Similarity metric for top-k features (jaccard / weighted_jaccard / cosine)
- Extra top-1 stability/drift indicators

Outputs are written under: <output_dir>/<exp_name>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm  
import sae_utils
import dream_infer


# =============================================================================
# 0) Defaults
# =============================================================================

DEFAULT_MODEL_NAME = "Dream-org/Dream-v0-Base-7B"

DEFAULT_GSM8K_PROMPT = """You are a helpful math tutor. Solve the problem step by step and give the final answer as an integer on the last line.
Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Reasoning: Let's think step by step.
Answer:"""


# =============================================================================
# 1) Helpers: IO and CLI parsing
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out


def parse_str_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_json_dict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {repr(e)}. Got: {s}")
    if not isinstance(obj, dict):
        raise ValueError("extra_generate_kwargs must be a JSON object (dict).")
    return obj


# =============================================================================
# 2) Position selection
# =============================================================================

@dataclass
class PositionPlan:
    """
    Plan for which positions to capture at a given step.
    """
    positions: List[int]
    reason: str


def _positions_from_diffs(diffs: List[Dict[str, Any]], step_t: int) -> List[int]:
    """
    diffs is a list for t=1..T-1 where diffs[idx]["step"] == idx+1 typically.
    We fetch changes for step_t if present.
    """
    if step_t <= 0:
        return []
    # diffs is aligned by index: diffs[t-1] is step t
    idx = step_t - 1
    if idx < 0 or idx >= len(diffs):
        return []
    row = diffs[idx]
    changes = row.get("changes", [])
    out = []
    for c in changes:
        out.append(int(c["pos_abs"]))
    return out


def select_positions(
    *,
    step_seqs: List[List[int]],
    diffs: List[Dict[str, Any]],
    step_t: int,
    input_len: int,
    gen_end_abs: int,
    mask_id: Optional[int],
    mode: str,
    include_answer_pos: bool,
    head_k: int,
    tail_k: int,
    extra_positions: List[int],
) -> PositionPlan:
    """
    Decide which absolute token positions to capture at this step.

    Modes:
    - full_gen: capture all positions in generated region [input_len, gen_end_abs)
    - update_only: capture only positions changed at this step (from diffs)
    - update_plus_anchors: changed positions + anchors (answer pos + head_k + tail_k + extra_positions)
    - mask_only: capture only generated positions that are currently mask tokens

    Note: positions are absolute indices in the full sequence (prompt+generated).
    """
    mode = (mode or "update_plus_anchors").lower()
    extra_positions = [int(p) for p in extra_positions]

    gen_positions = list(range(int(input_len), int(gen_end_abs)))

    anchors: List[int] = []
    if include_answer_pos:
        # A simple and strong default: first generated token position
        anchors.append(int(input_len))

    if head_k > 0:
        anchors.extend(list(range(int(input_len), min(int(gen_end_abs), int(input_len) + int(head_k)))))

    if tail_k > 0:
        start = max(int(input_len), int(gen_end_abs) - int(tail_k))
        anchors.extend(list(range(start, int(gen_end_abs))))

    anchors.extend(extra_positions)

    anchors = sorted(set([p for p in anchors if 0 <= p < len(step_seqs[step_t])]))

    if mode == "full_gen":
        pos = gen_positions
        return PositionPlan(positions=pos, reason="full_gen")

    if mode == "update_only":
        pos = _positions_from_diffs(diffs, step_t=step_t)
        pos = sorted(set([p for p in pos if 0 <= p < len(step_seqs[step_t])]))
        return PositionPlan(positions=pos, reason="update_only")

    if mode == "update_plus_anchors":
        upd = _positions_from_diffs(diffs, step_t=step_t)
        pos = sorted(set([p for p in upd + anchors if 0 <= p < len(step_seqs[step_t])]))
        return PositionPlan(positions=pos, reason="update_plus_anchors")

    if mode == "mask_only":
        if mask_id is None:
            # Without a mask id, fallback to full_gen
            return PositionPlan(positions=gen_positions, reason="mask_only_fallback_full_gen(no_mask_id)")
        ids = step_seqs[step_t]
        pos = [p for p in gen_positions if ids[p] == int(mask_id)]
        pos = sorted(set(pos + anchors))  # still keep anchors for monitoring
        return PositionPlan(positions=pos, reason="mask_only(+anchors)")

    # Default fallback
    upd = _positions_from_diffs(diffs, step_t=step_t)
    pos = sorted(set([p for p in upd + anchors if 0 <= p < len(step_seqs[step_t])]))
    return PositionPlan(positions=pos, reason=f"fallback_update_plus_anchors(mode={mode})")


# =============================================================================
# 3) Similarity metric between top-k feature sets
# =============================================================================

def compute_similarity(
    metric: str,
    a_ids: List[int],
    a_vals: List[float],
    b_ids: List[int],
    b_vals: List[float],
) -> float:
    metric = (metric or "jaccard").lower()
    if metric == "jaccard":
        return sae_utils.jaccard_ids(a_ids, b_ids)

    a_sp = sae_utils.topk_to_sparse(a_ids, a_vals)
    b_sp = sae_utils.topk_to_sparse(b_ids, b_vals)

    if metric == "weighted_jaccard":
        return sae_utils.weighted_jaccard(a_sp, b_sp)

    if metric == "cosine":
        return sae_utils.cosine_sparse(a_sp, b_sp)

    raise ValueError(f"Unknown sim_metric: {metric}. Choose from jaccard, weighted_jaccard, cosine.")


# =============================================================================
# 4) Online accumulators for Step1/Step2 metrics
# =============================================================================

@dataclass
class OnlineStats:
    """
    Track stability/drift metrics for one (layer, position) across steps.
    We update only when we have both prev and cur data for that position.
    """
    decode_step: Optional[int] = None

    # Pre-decode (mask-phase) stability
    pre_sim_sum: float = 0.0
    pre_sim_n: int = 0
    pre_lastk: deque = None  # type: ignore

    pre_top1_same: int = 0
    pre_top1_trans: int = 0
    pre_top1_val_delta_sum: float = 0.0

    # Post-decode drift
    post_drift_sum: float = 0.0
    post_drift_n: int = 0

    post_top1_flip: int = 0
    post_top1_n: int = 0
    post_top1_val_mean: float = 0.0
    post_top1_val_m2: float = 0.0  # for variance
    post_top1_val_delta_sum: float = 0.0

    def __post_init__(self):
        if self.pre_lastk is None:
            self.pre_lastk = deque(maxlen=5)

    def update_post_top1_welford(self, x: float) -> None:
        self.post_top1_n += 1
        n = self.post_top1_n
        delta = x - self.post_top1_val_mean
        self.post_top1_val_mean += delta / n
        delta2 = x - self.post_top1_val_mean
        self.post_top1_val_m2 += delta * delta2

    def post_top1_std(self) -> float:
        if self.post_top1_n <= 1:
            return 0.0
        var = self.post_top1_val_m2 / (self.post_top1_n - 1)
        return float(var ** 0.5)


# =============================================================================
# 5) Step3: order-sensitive features storage (per-step counts on disk)
# =============================================================================

def save_step_feature_counts_npz(path: str, counts: Counter) -> None:
    """
    Save a Counter(feature_id -> count) as a small NPZ for streaming later.
    """
    ensure_dir(os.path.dirname(path))
    fids = []
    cnts = []
    for fid, c in counts.items():
        fids.append(int(fid))
        cnts.append(int(c))
    arr_f = torch.tensor(fids, dtype=torch.int64).cpu().numpy()
    arr_c = torch.tensor(cnts, dtype=torch.int32).cpu().numpy()
    import numpy as np
    np.savez_compressed(path, fids=arr_f, cnts=arr_c)


def load_step_feature_counts_npz(path: str) -> Counter:
    import numpy as np
    d = np.load(path)
    fids = d["fids"]
    cnts = d["cnts"]
    out = Counter()
    for fid, c in zip(fids.tolist(), cnts.tolist()):
        out[int(fid)] = int(c)
    return out


# =============================================================================
# 6) Main analysis routine
# =============================================================================

@torch.inference_mode()
def forward_and_capture(
    model,
    cap_mgr: sae_utils.LayerCaptureManager,
    input_ids_1d: List[int],
    attn_mask_dtype: str,
    positions: List[int],
) -> sae_utils.CaptureResult:
    """
    Run one forward pass on a specific step sequence and capture hidden states at selected positions.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor(input_ids_1d, dtype=torch.long, device=device).unsqueeze(0)

    # Build attention mask
    if attn_mask_dtype == "bool":
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    elif attn_mask_dtype == "float":
        attention_mask = torch.ones_like(input_ids, dtype=torch.float32, device=device)
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    cap_mgr.clear()
    cap_mgr.set_positions(positions)

    # We call model forward; hooks capture block outputs.
    # Many HF models accept input_ids and attention_mask.
    # --- Fix attention_mask for Dream SDPA (must NOT be int/long) ---
    # Dream's attention uses torch.nn.functional.scaled_dot_product_attention,
    # which requires attn_mask dtype to be bool/float or match query dtype.
    # We treat attention_mask from tokenizer / our builder as a 0/1 "keep mask"
    # (1 = valid token, 0 = padding), and convert it to a boolean "padding mask"
    # (True = masked/padding, False = keep).
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=input_ids.device)

        # If it's an integer mask (0/1), convert to bool padding-mask: True where padding (==0)
        if attention_mask.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            attention_mask = attention_mask.eq(0)

        # If it's float but looks like 0/1 mask, also convert to bool padding-mask
        elif attention_mask.dtype in (torch.float16, torch.bfloat16, torch.float32):
            # heuristic: treat [0,1] as keep-mask
            if float(attention_mask.min()) >= 0.0 and float(attention_mask.max()) <= 1.0:
                attention_mask = attention_mask.eq(0)

        # If it's already bool, we assume it's a padding-mask (True=masked).
        elif attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(torch.bool)
    # --- end fix ---


    _ = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    cap = cap_mgr.get_capture()
    return cap


def analyze_one_alg(
    *,
    model,
    tokenizer,
    prompt: str,
    alg: str,
    args,
    exp_dir: str,
    saes: Dict[int, torch.nn.Module],
) -> Dict[str, Any]:
    """
    Run Dream generation for one alg (order strategy), then replay steps and compute:
    - Step1 mask stability (Jaccard/weighted/cosine + top1 stability)
    - Step2 post-decode drift (1-sim + top1 drift)
    - Step3 per-step feature counts (saved on disk for later cross-alg diff)
    """
    out_alg_dir = os.path.join(exp_dir, "runs", alg)
    ensure_dir(out_alg_dir)

    extra_kwargs = parse_json_dict(args.extra_generate_kwargs)

    hist = dream_infer.run_dream_history(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        alg=alg,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        alg_temp=args.alg_temp,
        do_sample=args.do_sample,
        seed=args.seed,
        strict_determinism=args.strict_determinism,
        use_chat_template=args.use_chat_template,
        mask_token_str=args.mask_token_str,
        extra_generate_kwargs=extra_kwargs,
    )

    # Save run metadata and diffs
    meta = {
        "alg": hist.alg,
        "model_name": args.model_name,
        "seed": args.seed,
        "strict_determinism": bool(args.strict_determinism),
        "steps_requested": int(args.steps),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "alg_temp": float(args.alg_temp),
        "do_sample": bool(args.do_sample),
        "extra_generate_kwargs": extra_kwargs,
        "input_len": int(hist.input_len),
        "gen_end_abs": int(hist.gen_end_abs),
        "num_steps_in_history": int(len(hist.step_seqs)),
        "final_sha256": hist.final_sha256,
        "final_text": hist.final_text,
        "mask_id": hist.mask_id,
        "positions_mode": args.positions_mode,
        "sim_metric": args.sim_metric,
        "sae_root_dir": args.sae_root_dir,
        "sae_layers": args.sae_layers,
        "sae_trainer": args.sae_trainer,
        "sae_k": args.sae_k,
        "sae_topk": args.sae_topk,
        "include_answer_pos": bool(args.include_answer_pos),
        "anchor_head_k": int(args.anchor_head_k),
        "anchor_tail_k": int(args.anchor_tail_k),
        "extra_positions": args.extra_positions,
        "attn_mask_dtype": args.attn_mask_dtype,
    }
    write_json(os.path.join(out_alg_dir, "meta.json"), meta)
    write_json(os.path.join(out_alg_dir, "diffs.json"), hist.diffs)

    # Pre-compute revise_count per position from diffs
    revise_count = Counter()
    for row in hist.diffs:
        for c in row.get("changes", []):
            revise_count[int(c["pos_abs"])] += 1

    # Build a capture manager once (register hooks once)
    cap_mgr = sae_utils.LayerCaptureManager(model=model, layers=args.sae_layers)

    # Online trackers:
    # stats[layer][pos] -> OnlineStats
    stats: Dict[int, Dict[int, OnlineStats]] = {l: {} for l in args.sae_layers}

    # prev cache for similarity computation: prev_feat[layer][pos] -> (ids[k], vals[k], top1_id, top1_val)
    prev_feat: Dict[int, Dict[int, Tuple[List[int], List[float], int, float]]] = {l: {} for l in args.sae_layers}

    # For Step3: write per-step feature counts for each layer
    counts_dir = os.path.join(exp_dir, "counts", alg)
    ensure_dir(counts_dir)
    for l in args.sae_layers:
        ensure_dir(os.path.join(counts_dir, f"layer{l}"))

    # Optionally store per-step per-layer per-pos top1 for debugging
    save_top1 = bool(args.save_top1_traces)
    top1_dir = os.path.join(exp_dir, "top1_traces", alg) if save_top1 else None
    if save_top1 and top1_dir is not None:
        for l in args.sae_layers:
            ensure_dir(os.path.join(top1_dir, f"layer{l}"))

    # Replay each step: forward -> capture -> SAE encode -> update stats
    max_steps = min(
        len(hist.step_seqs),
        int(args.max_steps_analyze) if args.max_steps_analyze > 0 else len(hist.step_seqs),
    )

    pbar = tqdm(
        range(max_steps),
        desc=f"[Replay] alg={alg}",
        unit="step",
        dynamic_ncols=True,
    )

    for t in pbar:
        ids_t = hist.step_seqs[t]

        plan = select_positions(
            step_seqs=hist.step_seqs,
            diffs=hist.diffs,
            step_t=t,
            input_len=hist.input_len,
            gen_end_abs=hist.gen_end_abs,
            mask_id=hist.mask_id,
            mode=args.positions_mode,
            include_answer_pos=args.include_answer_pos,
            head_k=args.anchor_head_k,
            tail_k=args.anchor_tail_k,
            extra_positions=args.extra_positions,
        )
        positions = plan.positions

        # update progress bar postfix (lightweight info)
        pbar.set_postfix({
            "P": len(positions),
            "mode": args.positions_mode,
        }, refresh=False)

        if not positions:
            continue

        cap = forward_and_capture(
            model=model,
            cap_mgr=cap_mgr,
            input_ids_1d=ids_t,
            attn_mask_dtype=args.attn_mask_dtype,
            positions=positions,
        )

        # For each layer: encode SAE, update metrics per position
        for layer in args.sae_layers:
            sae = saes[layer]
            hidden_PxD = cap.by_layer[layer]  # [P, D]

            topk_ids, topk_vals, top1_id, top1_val = sae_utils.encode_topk(
                sae=sae, hidden_PxD=hidden_PxD, k=int(args.sae_topk)
            )

            # Step3 per-step bag-of-features counts
            # Count features across selected positions at this step
            step_counts = Counter()
            # topk_ids: [P, K]
            # We count each feature once per position if it appears in top-k
            # (This matches C_{t,l}(f)=#{i: f in TopFeat(...)}.)
            topk_ids_cpu = topk_ids.detach().cpu()
            P = topk_ids_cpu.shape[0]
            K = topk_ids_cpu.shape[1]
            for pi in range(P):
                row_ids = set(int(x) for x in topk_ids_cpu[pi].tolist())
                for fid in row_ids:
                    step_counts[fid] += 1

            counts_path = os.path.join(counts_dir, f"layer{layer}", f"step{t:04d}.npz")
            save_step_feature_counts_npz(counts_path, step_counts)

            # Optional: save top1 trace for debugging
            if save_top1 and top1_dir is not None:
                # Save small JSONL line per step with positions and top1
                line = {
                    "step": int(t),
                    "positions": [int(p) for p in positions],
                    "top1_id": [int(x) for x in top1_id.detach().cpu().tolist()],
                    "top1_val": [float(x) for x in top1_val.detach().cpu().tolist()],
                }
                path = os.path.join(top1_dir, f"layer{layer}", "top1.jsonl")
                ensure_dir(os.path.dirname(path))
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            # Update Step1/Step2 online stats per position
            # We must align capture tensors with the positions list
            topk_ids_list = topk_ids.detach().cpu().tolist()
            topk_vals_list = topk_vals.detach().cpu().tolist()
            top1_id_list = top1_id.detach().cpu().tolist()
            top1_val_list = top1_val.detach().cpu().tolist()

            for j, pos_abs in enumerate(positions):
                pos_abs = int(pos_abs)
                tok_id = int(ids_t[pos_abs]) if 0 <= pos_abs < len(ids_t) else None
                if tok_id is None:
                    continue

                # Create stats record if missing
                if pos_abs not in stats[layer]:
                    stats[layer][pos_abs] = OnlineStats()

                st = stats[layer][pos_abs]

                # Determine decode_step (first time token is not mask)
                if hist.mask_id is not None:
                    if st.decode_step is None and tok_id != int(hist.mask_id):
                        st.decode_step = int(t)

                # If we have previous feature record for similarity computation
                if pos_abs in prev_feat[layer]:
                    prev_ids, prev_vals, prev_top1_id, prev_top1_val = prev_feat[layer][pos_abs]
                    cur_ids = [int(x) for x in topk_ids_list[j]]
                    cur_vals = [float(x) for x in topk_vals_list[j]]
                    cur_top1_id = int(top1_id_list[j])
                    cur_top1_val = float(top1_val_list[j])

                    sim = compute_similarity(
                        metric=args.sim_metric,
                        a_ids=cur_ids, a_vals=cur_vals,
                        b_ids=prev_ids, b_vals=prev_vals,
                    )

                    # Decide if this transition is pre-decode or post-decode
                    is_mask_now = (hist.mask_id is not None and tok_id == int(hist.mask_id))
                    # Pre-decode: we measure stability while still masked
                    if is_mask_now:
                        st.pre_sim_sum += float(sim)
                        st.pre_sim_n += 1
                        st.pre_lastk.append(float(sim))
                        st.pre_top1_trans += 1
                        if cur_top1_id == prev_top1_id:
                            st.pre_top1_same += 1
                        st.pre_top1_val_delta_sum += abs(cur_top1_val - float(prev_top1_val))
                    else:
                        # Post-decode: drift only after decode step is known
                        if st.decode_step is not None and t > st.decode_step:
                            st.post_drift_sum += float(1.0 - sim)
                            st.post_drift_n += 1

                            st.update_post_top1_welford(cur_top1_val)
                            st.post_top1_val_delta_sum += abs(cur_top1_val - float(prev_top1_val))

                            if cur_top1_id != prev_top1_id:
                                st.post_top1_flip += 1

                # Update prev cache
                prev_feat[layer][pos_abs] = (
                    [int(x) for x in topk_ids_list[j]],
                    [float(x) for x in topk_vals_list[j]],
                    int(top1_id_list[j]),
                    float(top1_val_list[j]),
                )

    # Close hooks
    cap_mgr.close()

    # Write Step1 and Step2 CSVs
    metrics_dir = os.path.join(exp_dir, "metrics")
    ensure_dir(metrics_dir)

    step1_path = os.path.join(metrics_dir, f"step1_mask_stability_{alg}.csv")
    step2_path = os.path.join(metrics_dir, f"step2_post_drift_{alg}.csv")

    # Step1
    with open(step1_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "alg", "layer", "pos_abs", "pos_rel_gen", "decode_step",
            "pre_sim_mean", "pre_sim_last5_mean",
            "pre_top1_id_lock_ratio", "pre_top1_val_delta_mean",
            "revise_count",
        ])
        for layer in args.sae_layers:
            for pos_abs, st in sorted(stats[layer].items(), key=lambda kv: kv[0]):
                pos_rel = pos_abs - int(hist.input_len)
                pre_mean = (st.pre_sim_sum / st.pre_sim_n) if st.pre_sim_n > 0 else 0.0
                last5 = list(st.pre_lastk)
                pre_last5_mean = (sum(last5) / len(last5)) if last5 else 0.0
                lock_ratio = (st.pre_top1_same / st.pre_top1_trans) if st.pre_top1_trans > 0 else 0.0
                val_delta_mean = (st.pre_top1_val_delta_sum / st.pre_top1_trans) if st.pre_top1_trans > 0 else 0.0
                w.writerow([
                    alg, layer, pos_abs, pos_rel,
                    (st.decode_step if st.decode_step is not None else -1),
                    float(pre_mean), float(pre_last5_mean),
                    float(lock_ratio), float(val_delta_mean),
                    int(revise_count.get(pos_abs, 0)),
                ])

    # Step2
    with open(step2_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "alg", "layer", "pos_abs", "pos_rel_gen", "decode_step",
            "drift_mean", "post_top1_flip_count", "post_top1_val_std", "post_top1_val_delta_mean",
            "revise_count",
        ])
        for layer in args.sae_layers:
            for pos_abs, st in sorted(stats[layer].items(), key=lambda kv: kv[0]):
                pos_rel = pos_abs - int(hist.input_len)
                drift = (st.post_drift_sum / st.post_drift_n) if st.post_drift_n > 0 else 0.0
                post_std = st.post_top1_std()
                post_delta_mean = (st.post_top1_val_delta_sum / st.post_drift_n) if st.post_drift_n > 0 else 0.0
                w.writerow([
                    alg, layer, pos_abs, pos_rel,
                    (st.decode_step if st.decode_step is not None else -1),
                    float(drift),
                    int(st.post_top1_flip),
                    float(post_std),
                    float(post_delta_mean),
                    int(revise_count.get(pos_abs, 0)),
                ])

    return {
        "alg": alg,
        "input_len": hist.input_len,
        "gen_end_abs": hist.gen_end_abs,
        "num_steps": max_steps,
        "mask_id": hist.mask_id,
        "final_sha256": hist.final_sha256,
        "final_text": hist.final_text,
        "counts_dir": os.path.join(exp_dir, "counts", alg),
        "metrics_step1": step1_path,
        "metrics_step2": step2_path,
    }


def compute_step3_order_sensitive(
    *,
    exp_dir: str,
    alg_ref: str,
    alg_other: str,
    layers: List[int],
    max_steps: int,
    topn: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Step3: For each layer l, compute order-sensitive features between alg_ref and alg_other:
      score(f) = sum_t |C^ref_{t,l}(f) - C^other_{t,l}(f)|

    We stream per-step counts from disk, so we do not store all per-step maps in memory.
    """
    scores_by_layer: Dict[int, Dict[int, float]] = {l: defaultdict(float) for l in layers}

    for t in range(max_steps):
        for l in layers:
            p_ref = os.path.join(exp_dir, "counts", alg_ref, f"layer{l}", f"step{t:04d}.npz")
            p_oth = os.path.join(exp_dir, "counts", alg_other, f"layer{l}", f"step{t:04d}.npz")

            if not (os.path.exists(p_ref) and os.path.exists(p_oth)):
                continue

            c_ref = load_step_feature_counts_npz(p_ref)
            c_oth = load_step_feature_counts_npz(p_oth)

            keys = set(c_ref.keys()) | set(c_oth.keys())
            sb = scores_by_layer[l]
            for fid in keys:
                sb[int(fid)] += abs(float(c_ref.get(fid, 0)) - float(c_oth.get(fid, 0)))

    # Extract top-n
    top_by_layer: Dict[int, List[Tuple[int, float]]] = {}
    for l in layers:
        items = list(scores_by_layer[l].items())
        items.sort(key=lambda kv: kv[1], reverse=True)
        top_by_layer[l] = [(int(fid), float(score)) for fid, score in items[: int(topn)]]
    return top_by_layer


# =============================================================================
# 7) CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true", default=True)

    # Prompt
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text. If not set, uses DEFAULT_GSM8K_PROMPT.")
    ap.add_argument("--prompt_file", type=str, default=None, help="Path to a text file containing the prompt.")
    ap.add_argument("--use_chat_template", action="store_true", default=True)

    # Dream official-ish generation args
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--alg_temp", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true", default=True)
    ap.add_argument("--algs", type=str, default="origin,entropy", help="Comma-separated alg names to compare.")
    ap.add_argument("--extra_generate_kwargs", type=str, default="",
                    help="JSON dict string passed into diffusion_generate, for custom official params (e.g., remask strategy).")

    # Determinism
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_determinism", action="store_true", default=True)

    # Mask id
    ap.add_argument("--mask_token_str", type=str, default=None,
                    help="Optional mask token string to infer mask_id if tokenizer.mask_token_id is missing.")

    # SAE
    ap.add_argument("--sae_root_dir", type=str, required=True,
                    help="Example: /home/dslabra5/sae4dlm/dictionary_learning_demo/saes_mask_Dream-org_Dream-v0-Base-7B_top_k")
    ap.add_argument("--sae_layers", type=str, default="5,14,23", help="Comma-separated SAE layers.")
    ap.add_argument("--sae_trainer", type=str, default=None,
                    help="Trainer folder name like 'trainer_0'. Overrides --sae_k if set.")
    ap.add_argument("--sae_k", type=int, default=50,
                    help="Select trainer by config.json trainer.k == this value (default=50).")
    ap.add_argument("--sae_topk", type=int, default=50,
                    help="Extract top-K features per token for metrics (default=50). Should match sae_k.")


    # Positions selection
    ap.add_argument("--positions_mode", type=str, default="update_plus_anchors",
                    choices=["full_gen", "update_only", "update_plus_anchors", "mask_only"])
    ap.add_argument("--include_answer_pos", action="store_true", default=True)
    ap.add_argument("--anchor_head_k", type=int, default=32)
    ap.add_argument("--anchor_tail_k", type=int, default=0)
    ap.add_argument("--extra_positions", type=str, default="", help="Comma-separated extra absolute positions to always include.")
    ap.add_argument("--attn_mask_dtype", type=str, default="long", choices=["long", "bool", "float"])

    # Similarity metric between consecutive steps (for Step1/Step2)
    ap.add_argument("--sim_metric", type=str, default="jaccard", choices=["jaccard", "weighted_jaccard", "cosine"])

    # Output
    ap.add_argument("--output_dir", type=str, default="/home/dslabra5/sae4dlm/dlm/output")
    ap.add_argument("--exp_name", type=str, default=None, help="Experiment name; if empty, auto timestamped.")
    ap.add_argument("--max_steps_analyze", type=int, default=0, help="0 means analyze all available history steps.")
    ap.add_argument("--save_top1_traces", action="store_true", default=False)

    # Step3
    ap.add_argument("--compute_step3", action="store_true", default=True)
    ap.add_argument("--step3_topn", type=int, default=20)

    return ap


def main():
    args = build_argparser().parse_args()

    # Prompt resolution
    if args.prompt_file:
        prompt = read_text_file(args.prompt_file)
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = DEFAULT_GSM8K_PROMPT

    algs = parse_str_list(args.algs)
    if len(algs) < 1:
        raise ValueError("You must provide at least one alg in --algs.")

    args.sae_layers = parse_int_list(args.sae_layers)
    if not args.sae_layers:
        raise ValueError("--sae_layers cannot be empty.")
    args.extra_positions = parse_int_list(args.extra_positions)

    # Build experiment folder
    exp_name = args.exp_name
    if not exp_name:
        exp_name = time.strftime("dlm_order_%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, exp_name)
    ensure_dir(exp_dir)

    # Save config snapshot
    write_json(os.path.join(exp_dir, "config.json"), {
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "algs": algs,
        "prompt_preview": prompt[:4096],
        "args": vars(args),
    })

    # Build model/tokenizer
    model, tokenizer = dream_infer.build_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    # Load SAEs once
    saes = sae_utils.load_saes(
        sae_root_dir=args.sae_root_dir,
        layers=args.sae_layers,
        device=args.device,
        trainer_name=args.sae_trainer,
        k_topk=args.sae_k,
    )
    # Move SAEs to device (LocalSAE buffers already on device, but keep for safety)
    for l in saes:
        saes[l] = saes[l].to(args.device).eval()

    # Run each alg analysis
    summaries: Dict[str, Any] = {}
    for alg in algs:
        summ = analyze_one_alg(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alg=alg,
            args=args,
            exp_dir=exp_dir,
            saes=saes,
        )
        summaries[alg] = summ

    write_json(os.path.join(exp_dir, "summaries.json"), summaries)

    # Step3: compare algs against first alg (ref)
    if args.compute_step3 and len(algs) >= 2:
        ref = algs[0]
        metrics_dir = os.path.join(exp_dir, "metrics")
        ensure_dir(metrics_dir)

        # Determine max_steps for step3 streaming
        max_steps = min(summaries[a]["num_steps"] for a in algs)

        for other in algs[1:]:
            top_by_layer = compute_step3_order_sensitive(
                exp_dir=exp_dir,
                alg_ref=ref,
                alg_other=other,
                layers=args.sae_layers,
                max_steps=max_steps,
                topn=int(args.step3_topn),
            )

            out_path = os.path.join(metrics_dir, f"step3_order_sensitive_{ref}_vs_{other}.json")
            write_json(out_path, {
                "ref_alg": ref,
                "other_alg": other,
                "max_steps_used": int(max_steps),
                "topn": int(args.step3_topn),
                "top_by_layer": {str(k): v for k, v in top_by_layer.items()},
            })

    print(f"[DONE] Outputs written to: {exp_dir}")


if __name__ == "__main__":
    main()
