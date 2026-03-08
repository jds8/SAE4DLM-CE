#!/usr/bin/env python3
# dream_infer.py
# -*- coding: utf-8 -*-
"""
Dream DLM inference helpers:
- Run diffusion_generate with output_history=True to obtain per-step token sequences (step_seqs).
- Normalize history format across possible returns.
- Early-stop using stop token sequences (EOS / im_end / etc.).
- (Origin-only) Hard truncate:
    A) after GSM8K-style final answer marker "#### <integer>"
    B) OR once we detect obvious tail collapse / repetition (e.g., "0000...") in the generated region.
  And propagate truncation to:
    - gen_end_abs
    - step_seqs[t] for all t (so replay forward will NOT see tail garbage tokens)
  Optionally reduce the number of kept steps when the truncated prefix becomes stable.
- Compute per-step diffs (which absolute positions changed).

This module does NOT do SAE encoding. It only produces the "order trajectory":
  step -> sequence token ids, and step -> changed positions.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# 0) Determinism helpers
# =============================================================================

def set_global_determinism(seed: int, strict: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if strict:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fall back if strict determinism is not supported by some ops.
            torch.use_deterministic_algorithms(False)


def sha256_int_list(xs: List[int]) -> str:
    m = hashlib.sha256()
    m.update((",".join(map(str, xs))).encode("utf-8"))
    return m.hexdigest()


# =============================================================================
# 1) Tokenization helpers
# =============================================================================

def to_prompt_inputs(
    tokenizer,
    text: str,
    device: str,
    use_chat_template: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert prompt text to (input_ids, attention_mask).
    For Dream chat-style models, apply_chat_template is often the correct method.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        return input_ids, attention_mask

    # Fallback to plain tokenization
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def build_stop_sequences(tokenizer) -> List[List[int]]:
    """
    Build stop token-id sequences. Prefer single-token stop markers if possible.
    We include:
      - tokenizer.eos_token_id (if any)
      - tokenized "<|im_end|>" (common for chat templates)
      - tokenized "</s>" and "<|endoftext|>" as fallbacks
    """
    stop_seqs: List[List[int]] = []

    if getattr(tokenizer, "eos_token_id", None) is not None:
        stop_seqs.append([int(tokenizer.eos_token_id)])

    for s in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) > 0:
                stop_seqs.append([int(x) for x in ids])
        except Exception:
            pass

    # Deduplicate
    uniq = []
    seen = set()
    for seq in stop_seqs:
        key = tuple(seq)
        if key not in seen:
            uniq.append(seq)
            seen.add(key)
    return uniq


def find_first_stop(gen_ids: List[int], stop_seqs: List[List[int]]) -> Optional[Tuple[int, int]]:
    """
    Return (index, length) of earliest stop sequence match in gen_ids.
    """
    best: Optional[Tuple[int, int]] = None
    n = len(gen_ids)
    for i in range(n):
        for seq in stop_seqs:
            m = len(seq)
            if m == 0 or i + m > n:
                continue
            if gen_ids[i: i + m] == seq:
                if best is None or i < best[0]:
                    best = (i, m)
        if best is not None and best[0] == i:
            return best
    return best


def apply_early_stop_from_history(
    step_seqs: List[List[int]],
    input_len: int,
    stop_seqs: List[List[int]],
    max_new_tokens: int,
) -> Tuple[List[List[int]], int]:
    """
    Use history to simulate "stop when stop token appears":
      - Find the earliest step where a stop sequence appears in the generated region.
      - Return truncated step_seqs (up to that step) and gen_end_abs (exclusive, stop token removed).
    If no stop token ever appears, keep all steps and gen_end_abs = input_len + max_new_tokens (bounded by sequence length).
    """
    if len(step_seqs) == 0:
        return step_seqs, input_len

    best_step: Optional[int] = None
    best_cut_abs: Optional[int] = None

    for t, ids in enumerate(step_seqs):
        gen_ids = ids[input_len:]
        hit = find_first_stop(gen_ids, stop_seqs)
        if hit is None:
            continue
        idx, _m = hit
        cut_abs = input_len + idx
        best_step = t
        best_cut_abs = cut_abs
        break

    if best_step is None:
        last_len = len(step_seqs[-1])
        gen_end_abs = min(last_len, input_len + int(max_new_tokens))
        return step_seqs, gen_end_abs

    truncated_steps = step_seqs[: best_step + 1]
    return truncated_steps, int(best_cut_abs)


# =============================================================================
# 1.5) Origin-only hard truncation:
#   A) GSM8K marker "#### <integer>"
#   B) Repeat/collapse detection (e.g., "0000..." or other low-diversity tail)
# =============================================================================

_GSM8K_FINAL_RE = re.compile(r"####\s*(-?\d+)")

# Heuristics for "repeat tail collapse" (origin only).
# You can tune these if needed.
REPEAT_MIN_IDENTICAL_SUFFIX = 64   # if the last N generated tokens are identical, treat as collapse
REPEAT_LOW_DIVERSITY_WINDOW = 160  # look at last W tokens
REPEAT_LOW_DIVERSITY_MAX_UNIQ = 3  # if <= this many unique token ids in that window
REPEAT_LOW_DIVERSITY_TOP_RATIO = 0.92  # dominant token ratio in that window
REPEAT_MIN_CUT_AFTER_GEN_TOKENS = 8    # do not cut too early (avoid false positives)


def _find_gsm8k_cut_abs_from_final_step(
    tokenizer,
    final_step_ids: List[int],
    input_len: int,
    max_new_tokens: int,
) -> Optional[int]:
    """
    Find a cut position (absolute token index, exclusive) from the FINAL step sequence:
    - Search for the LAST occurrence of pattern: "#### <integer>" in the GENERATED REGION.
    - Map the end-char position of that match back to a token boundary by accumulating
      per-token decoded piece lengths (good enough for ASCII-ish GSM8K answers).
    - Return cut_abs = input_len + token_index_end.

    If no match found, return None.
    """
    gen_ids = final_step_ids[input_len: input_len + int(max_new_tokens)]
    if not gen_ids:
        return None

    pieces = []
    for tid in gen_ids:
        pieces.append(tokenizer.decode([int(tid)], skip_special_tokens=False))

    text = "".join(pieces)
    matches = list(_GSM8K_FINAL_RE.finditer(text))
    if not matches:
        return None

    m = matches[-1]
    end_char = m.end()

    cum = 0
    for i, p in enumerate(pieces):
        cum += len(p)
        if cum >= end_char:
            return int(input_len + i + 1)

    return int(input_len + len(gen_ids))


def _find_repeat_collapse_cut_abs_from_final_step(
    tokenizer,
    final_step_ids: List[int],
    input_len: int,
    max_new_tokens: int,
) -> Optional[int]:
    """
    Detect obvious tail collapse / repetition from FINAL step sequence only, and return
    an absolute cut position (exclusive) BEFORE the collapsed region.

    Two signals:
    1) Very long identical-token suffix (e.g., "0000..." token repeated).
    2) Low-diversity tail window dominated by one token (e.g., mostly "0" or whitespace).

    If no collapse detected, return None.
    """
    from collections import Counter

    gen_ids = final_step_ids[input_len: input_len + int(max_new_tokens)]
    if not gen_ids:
        return None

    # ---- (1) identical-token suffix ----
    last_tid = int(gen_ids[-1])
    i = len(gen_ids) - 1
    while i >= 0 and int(gen_ids[i]) == last_tid:
        i -= 1
    run_len = (len(gen_ids) - 1) - i  # suffix length
    if run_len >= int(REPEAT_MIN_IDENTICAL_SUFFIX):
        cut_rel = i + 1
        if cut_rel >= int(REPEAT_MIN_CUT_AFTER_GEN_TOKENS):
            return int(input_len + cut_rel)

    # ---- (2) low-diversity, highly-dominated window ----
    w = int(min(REPEAT_LOW_DIVERSITY_WINDOW, len(gen_ids)))
    if w >= int(REPEAT_MIN_IDENTICAL_SUFFIX // 2):  # only if window is not tiny
        tail = [int(x) for x in gen_ids[-w:]]
        uniq = set(tail)
        if len(uniq) <= int(REPEAT_LOW_DIVERSITY_MAX_UNIQ):
            c = Counter(tail)
            top_tid, top_cnt = c.most_common(1)[0]
            if (top_cnt / w) >= float(REPEAT_LOW_DIVERSITY_TOP_RATIO):
                cut_rel = len(gen_ids) - w
                if cut_rel >= int(REPEAT_MIN_CUT_AFTER_GEN_TOKENS):
                    return int(input_len + cut_rel)

    return None


def apply_origin_hard_truncation(
    *,
    tokenizer,
    step_seqs: List[List[int]],
    input_len: int,
    gen_end_abs: int,
    max_new_tokens: int,
) -> Tuple[List[List[int]], int]:
    """
    Origin-only hard truncation:
    - Determine candidate cut_abs from:
        A) GSM8K marker "#### <integer>"
        B) repeat/collapse detection (e.g., "0000..." tail)
    - Choose the EARLIEST valid cut_abs (to remove junk as much as possible).
    - If cut_abs < gen_end_abs:
        * gen_end_abs := cut_abs
        * step_seqs[t] := step_seqs[t][:gen_end_abs] for all t
      This ensures replay forward does not see tail garbage tokens.
    - Optionally reduce number of steps kept:
        Find earliest t such that for ALL s>=t, truncated prefix equals final truncated prefix.
        Then keep step_seqs = step_seqs[:t+1].
    """
    if not step_seqs:
        return step_seqs, gen_end_abs

    final_ids_full = step_seqs[-1]

    cut_a = _find_gsm8k_cut_abs_from_final_step(
        tokenizer=tokenizer,
        final_step_ids=final_ids_full,
        input_len=input_len,
        max_new_tokens=max_new_tokens,
    )
    cut_b = _find_repeat_collapse_cut_abs_from_final_step(
        tokenizer=tokenizer,
        final_step_ids=final_ids_full,
        input_len=input_len,
        max_new_tokens=max_new_tokens,
    )

    cands = []
    for c in [cut_a, cut_b]:
        if c is None:
            continue
        c = int(c)
        # sanity bounds: never cut into prompt, never extend
        if c <= input_len:
            continue
        cands.append(c)

    if not cands:
        return step_seqs, gen_end_abs

    cut_abs = int(min(cands))
    cut_abs = int(min(cut_abs, gen_end_abs, len(final_ids_full)))

    # If we didn't shorten, no-op.
    if cut_abs >= int(gen_end_abs):
        return step_seqs, gen_end_abs

    # Truncate each step sequence so forward replay does NOT include tail tokens.
    new_steps: List[List[int]] = []
    for ids in step_seqs:
        new_steps.append(ids[: min(len(ids), cut_abs)])

    new_gen_end_abs = cut_abs

    # Try to safely reduce steps if the truncated prefix becomes stable.
    final_prefix = new_steps[-1]
    stable_t: Optional[int] = None
    T = len(new_steps)
    for t in range(T):
        if new_steps[t] != final_prefix:
            continue
        ok = True
        for s in range(t, T):
            if new_steps[s] != final_prefix:
                ok = False
                break
        if ok:
            stable_t = t
            break

    if stable_t is not None and stable_t < (T - 1):
        new_steps = new_steps[: stable_t + 1]

    return new_steps, int(new_gen_end_abs)


# =============================================================================
# 2) History normalization
# =============================================================================

def extract_history(out: Any) -> Optional[Any]:
    if hasattr(out, "history"):
        return out.history
    if hasattr(out, "sequences_history"):
        return out.sequences_history
    if isinstance(out, dict) and "history" in out:
        return out["history"]
    return None


def normalize_step_seqs(hist: Any) -> List[List[int]]:
    """
    Normalize model output history to Python lists:
      step_seqs[t] is a list[int] of token ids at step t for batch item 0.
    """
    step_seqs: List[List[int]] = []
    if hist is None:
        return step_seqs

    if isinstance(hist, (list, tuple)):
        for s in hist:
            if isinstance(s, torch.Tensor):
                arr = s.detach().cpu()
            else:
                arr = torch.tensor(s)
            if arr.dim() == 2:
                step_seqs.append(arr[0].tolist())
            elif arr.dim() == 1:
                step_seqs.append(arr.tolist())
            else:
                raise ValueError(f"Unsupported history item shape: {tuple(arr.shape)}")
        return step_seqs

    if isinstance(hist, torch.Tensor):
        arr = hist.detach().cpu()
        if arr.dim() == 3:
            for t in range(arr.shape[0]):
                step_seqs.append(arr[t, 0].tolist())
        elif arr.dim() == 2:
            for t in range(arr.shape[0]):
                step_seqs.append(arr[t].tolist())
        else:
            raise ValueError(f"Unsupported history tensor shape: {tuple(arr.shape)}")
        return step_seqs

    raise ValueError(f"Unsupported history type: {type(hist)}")


# =============================================================================
# 3) Step diffs (which absolute positions changed step-to-step)
# =============================================================================

def diff_by_step(
    tokenizer,
    step_seqs: List[List[int]],
    gen_start: int,
    gen_end_abs: int,
) -> List[Dict[str, Any]]:
    """
    Compare step t-1 -> t, record changed positions in [gen_start, gen_end_abs).
    Returns a list where each element is:
      {"step": t, "num_changes": int, "changes": [{"pos_abs", "pos_rel", "token_id", "token_str"}]}
    """
    diffs: List[Dict[str, Any]] = []
    if len(step_seqs) < 2:
        return diffs

    prev = step_seqs[0]
    for t in range(1, len(step_seqs)):
        cur = step_seqs[t]
        L = min(len(prev), len(cur), gen_end_abs)
        changes = []
        for pos in range(gen_start, L):
            if prev[pos] != cur[pos]:
                tid = cur[pos]
                piece = tokenizer.decode([tid], skip_special_tokens=False)
                piece_vis = piece.replace("\n", "\\n").replace("\t", "\\t")
                changes.append(
                    {
                        "pos_abs": pos,
                        "pos_rel": pos - gen_start,
                        "token_id": int(tid),
                        "token_str": piece_vis,
                    }
                )
        diffs.append({"step": t, "num_changes": len(changes), "changes": changes})
        prev = cur
    return diffs


# =============================================================================
# 4) Mask token id inference (robust)
# =============================================================================

def infer_mask_id(
    tokenizer,
    step_seqs: List[List[int]],
    input_len: int,
    mask_token_str: Optional[str] = None,
) -> Optional[int]:
    """
    Infer the mask token id for Dream diffusion steps.

    Priority:
    1) tokenizer.mask_token_id (if exists)
    2) encode mask_token_str (if provided)
    3) heuristic: most frequent token id in generated region at step 0
    """
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return int(tokenizer.mask_token_id)

    if mask_token_str:
        try:
            ids = tokenizer.encode(mask_token_str, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) == 1:
                return int(ids[0])
        except Exception:
            pass

    if not step_seqs:
        return None

    gen0 = step_seqs[0][input_len:]
    if not gen0:
        return None
    from collections import Counter
    c = Counter(gen0)
    mask_id, _ = c.most_common(1)[0]
    return int(mask_id)


# =============================================================================
# 5) Main run wrappers
# =============================================================================

@dataclass
class DreamHistoryResult:
    alg: str
    input_len: int
    gen_end_abs: int
    step_seqs: List[List[int]]
    diffs: List[Dict[str, Any]]
    final_ids: List[int]
    final_text: str
    final_sha256: str
    mask_id: Optional[int]


@torch.inference_mode()
def run_dream_history(
    model,
    tokenizer,
    prompt: str,
    *,
    alg: str,
    steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alg_temp: float,
    do_sample: bool,
    seed: int,
    strict_determinism: bool,
    use_chat_template: bool,
    mask_token_str: Optional[str],
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
) -> DreamHistoryResult:
    """
    Run Dream diffusion_generate with output_history=True and return normalized history.

    IMPORTANT:
    - For origin, Dream may keep a fixed-length generated region and never emit EOS,
      causing tail collapse into garbage (e.g., "0000...").
    - We hard-truncate origin outputs after:
        * GSM8K final answer marker "#### <integer>" OR
        * repeat/collapse detection
      and we truncate step_seqs[t] so replay forward does not see the tail.
    - For entropy, we keep the original behavior unchanged.
    """
    set_global_determinism(seed, strict=strict_determinism)

    input_ids, attention_mask = to_prompt_inputs(
        tokenizer=tokenizer,
        text=prompt,
        device=str(input_ids_device(model)),
        use_chat_template=use_chat_template,
    )
    input_len = int(input_ids.shape[1])

    kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        steps=int(steps),
        temperature=float(temperature),
        top_p=float(top_p),
        alg=str(alg),
        alg_temp=float(alg_temp),
        output_history=True,
        return_dict_in_generate=True,
        do_sample=bool(do_sample),
    )
    if extra_generate_kwargs:
        kwargs.update(extra_generate_kwargs)

    out = model.diffusion_generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )

    hist = extract_history(out)
    step_seqs = normalize_step_seqs(hist)
    if len(step_seqs) == 0:
        raise RuntimeError(
            "No history found. Ensure Dream supports output_history=True and exposes history/sequences_history."
        )

    # 1) Generic early-stop using EOS / im_end / etc.
    stop_seqs = build_stop_sequences(tokenizer)
    step_seqs, gen_end_abs = apply_early_stop_from_history(
        step_seqs=step_seqs,
        input_len=input_len,
        stop_seqs=stop_seqs,
        max_new_tokens=max_new_tokens,
    )

    # 2) Origin-only: hard truncation (GSM8K marker OR repeat/collapse).
    if str(alg).lower() == "origin":
        step_seqs, gen_end_abs = apply_origin_hard_truncation(
            tokenizer=tokenizer,
            step_seqs=step_seqs,
            input_len=input_len,
            gen_end_abs=gen_end_abs,
            max_new_tokens=max_new_tokens,
        )

    # Final ids are from the last kept step, truncated.
    final_ids_full = step_seqs[-1]
    final_ids = final_ids_full[:gen_end_abs]
    gen_ids = final_ids[input_len:gen_end_abs]
    final_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    final_sha = sha256_int_list(final_ids)

    diffs = diff_by_step(
        tokenizer=tokenizer,
        step_seqs=step_seqs,
        gen_start=input_len,
        gen_end_abs=gen_end_abs,
    )

    mask_id = infer_mask_id(
        tokenizer=tokenizer,
        step_seqs=step_seqs,
        input_len=input_len,
        mask_token_str=mask_token_str,
    )

    return DreamHistoryResult(
        alg=str(alg),
        input_len=input_len,
        gen_end_abs=int(gen_end_abs),
        step_seqs=step_seqs,
        diffs=diffs,
        final_ids=final_ids,
        final_text=final_text,
        final_sha256=final_sha,
        mask_id=mask_id,
    )


def input_ids_device(model) -> torch.device:
    """
    Choose a reasonable device for prompt tensors.
    """
    try:
        p0 = next(model.parameters())
        return p0.device
    except StopIteration:
        return torch.device("cpu")


def build_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
):
    """
    Build Dream model + tokenizer.
    dtype: one of {"auto","fp16","bf16","fp32"}
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if dtype == "fp16":
        dt = torch.float16
    elif dtype == "bf16":
        dt = torch.bfloat16
    elif dtype == "fp32":
        dt = torch.float32
    else:
        dt = None  # auto

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dt,
        trust_remote_code=trust_remote_code,
    ).to(device).eval()

    return model, tokenizer
