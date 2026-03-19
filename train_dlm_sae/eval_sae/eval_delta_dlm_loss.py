# eval_delta_lm_loss.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
import inspect
from typing import Iterable, List, Dict, Tuple, Optional, Any

import torch as t
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from dictionary_learning.dictionary_learning import utils


############################################
# General helpers
############################################

def is_dream_like(model) -> bool:
    """
    Heuristic check: model class name includes 'dream' or 'diffusion',
    or the model exposes diffusion_generate().
    """
    name = model.__class__.__name__.lower()
    return ("dream" in name) or ("diffusion" in name) or hasattr(model, "diffusion_generate")


def iter_texts(dataset_name: str, split: str = "train") -> Iterable[str]:
    """
    Stream raw text strings from a HF dataset using our utils helper.
    Assumes dataset only has a 'train' split (or we always read 'train').
    """
    return utils.hf_dataset_to_generator(dataset_name, split=split, streaming=True)


def _pick_best_tensor(cands: List[t.Tensor]) -> t.Tensor:
    """
    Choose a likely-logits tensor from candidate tensors:
      - Prefer rank-3 tensors (B, T, V)
      - Else prefer rank-2 tensors (T, V)
      - Else first tensor.
    """
    rank3 = [x for x in cands if isinstance(x, t.Tensor) and x.ndim == 3]
    if len(rank3) > 0:
        return rank3[0]
    rank2 = [x for x in cands if isinstance(x, t.Tensor) and x.ndim == 2]
    if len(rank2) > 0:
        return rank2[0]
    for x in cands:
        if isinstance(x, t.Tensor):
            return x
    raise RuntimeError("No tensor candidate found among provided candidates.")


def get_logits(outputs: Any) -> t.Tensor:
    """
    Robustly extract a tensor of logits (or equivalent) from model outputs.

    Handles:
    - dict-like / ModelOutput
    - tuple/list
    - object with .logits / .last_hidden_state
    """
    # Dict-like or ModelOutput (which typically subclasses dict)
    if isinstance(outputs, dict):
        # Common keys
        if "logits" in outputs and isinstance(outputs["logits"], t.Tensor):
            return outputs["logits"]
        if "last_hidden_state" in outputs and isinstance(outputs["last_hidden_state"], t.Tensor):
            return outputs["last_hidden_state"]

        # Gather all tensor values
        cands = [v for v in outputs.values() if isinstance(v, t.Tensor)]
        if len(cands) > 0:
            return _pick_best_tensor(cands)

        # Some values may be nested tuples/lists containing tensors
        nested = []
        for v in outputs.values():
            if isinstance(v, (tuple, list)):
                nested.extend([x for x in v if isinstance(x, t.Tensor)])
        if len(nested) > 0:
            return _pick_best_tensor(nested)

    # Object with attributes
    if hasattr(outputs, "logits") and isinstance(outputs.logits, t.Tensor):
        return outputs.logits
    if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, t.Tensor):
        return outputs.last_hidden_state

    # Tuple/list of tensors
    if isinstance(outputs, (tuple, list)):
        cands = [x for x in outputs if isinstance(x, t.Tensor)]
        if len(cands) > 0:
            return _pick_best_tensor(cands)

    raise RuntimeError(f"Cannot find logits in model outputs of type: {type(outputs)}")


############################################
# Helpers to deal with tuple-ish activations in hooks
############################################

def _first_tensor(obj: Any) -> Optional[t.Tensor]:
    """
    Recursively return the first torch.Tensor found inside `obj`.
    Priority:
      - Direct Tensor
      - Inside list/tuple
      - Inside dict values (common activation keys first)
    Returns None if not found.
    """
    if isinstance(obj, t.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found

    if isinstance(obj, dict):
        # Try some common activation-ish keys first
        preferred_keys = (
            "hidden_states",
            "last_hidden_state",
            "x",
            "resid",
            "resid_post",
            "out",
            "activations",
            "act",
            "states",
            "x_hat",
            "reconstruction",
            "decoded",
        )
        for k in preferred_keys:
            if k in obj and isinstance(obj[k], t.Tensor):
                return obj[k]
        # Fallback: recurse through values
        for v in obj.values():
            found = _first_tensor(v)
            if found is not None:
                return found

    # Anything else (custom class etc.): best effort via __dict__
    if hasattr(obj, "__dict__"):
        return _first_tensor(vars(obj))

    return None


def _replace_first_tensor(structure: Any, new_tensor: t.Tensor) -> Any:
    """
    Return `structure` but with its *first* encountered Tensor replaced
    by `new_tensor` (same container type preserved).
    """
    if isinstance(structure, t.Tensor):
        return new_tensor

    if isinstance(structure, list):
        new_items = []
        replaced = False
        for item in structure:
            if not replaced and _first_tensor(item) is not None:
                new_items.append(_replace_first_tensor(item, new_tensor))
                replaced = True
            else:
                new_items.append(item)
        return type(structure)(new_items)

    if isinstance(structure, tuple):
        new_items = []
        replaced = False
        for item in structure:
            if not replaced and _first_tensor(item) is not None:
                new_items.append(_replace_first_tensor(item, new_tensor))
                replaced = True
            else:
                new_items.append(item)
        return type(structure)(new_items)

    if isinstance(structure, dict):
        new_dict = {}
        replaced = False
        for k, v in structure.items():
            if (not replaced) and _first_tensor(v) is not None:
                new_dict[k] = _replace_first_tensor(v, new_tensor)
                replaced = True
            else:
                new_dict[k] = v
        return new_dict

    # Fallback for weird custom objects with attributes
    if hasattr(structure, "__dict__"):
        attrs = vars(structure).copy()
        replaced = False
        for k, v in attrs.items():
            if (not replaced) and _first_tensor(v) is not None:
                attrs[k] = _replace_first_tensor(v, new_tensor)
                replaced = True
        return attrs

    # Last-ditch fallback: just return new tensor
    return new_tensor


############################################
# Device / dtype alignment helpers
############################################

def _move_module_like_tensor(module: t.nn.Module, ref_like: Any):
    """
    Move a module to the same (device, dtype) as `ref_like`.
    `ref_like` may be a Tensor or a nested structure containing a Tensor.

    We cache the last (device, dtype) we moved to so we don't waste time
    doing redundant .to() every hook call.
    """
    ref_tensor = _first_tensor(ref_like)
    if ref_tensor is None:
        module.eval()
        return

    dev = ref_tensor.device
    dt = ref_tensor.dtype

    if getattr(module, "_dlm_cached_device", None) != dev or getattr(module, "_dlm_cached_dtype", None) != dt:
        module.to(device=dev, dtype=dt)
        module._dlm_cached_device = dev
        module._dlm_cached_dtype = dt
    module.eval()


def reconstruct_with_dictionary(dictionary: t.nn.Module, x: t.Tensor) -> t.Tensor:
    """
    Return x_hat reconstructed by the SAE.

    Try encode()/decode() first.
    Fallback: dictionary(x) heuristics.
    """
    with t.no_grad():
        _move_module_like_tensor(dictionary, x)

        # 1) Preferred API: encode/decode
        if hasattr(dictionary, "encode") and hasattr(dictionary, "decode"):
            try:
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                return x_hat.to(dtype=x.dtype, device=x.device)
            except Exception:
                pass

        # 2) Forward-based heuristics
        try:
            out = dictionary(x)
            # Case A: forward returns a Tensor directly
            if isinstance(out, t.Tensor):
                return out.to(dtype=x.dtype, device=x.device)

            # Case B: forward returns list/tuple -> choose same-shape tensor, else first tensor
            if isinstance(out, (list, tuple)):
                same = [o for o in out if isinstance(o, t.Tensor) and o.shape == x.shape]
                if len(same) > 0:
                    return same[0].to(dtype=x.dtype, device=x.device)
                tensors = [o for o in out if isinstance(o, t.Tensor)]
                if len(tensors) > 0:
                    return tensors[0].to(dtype=x.dtype, device=x.device)

            # Case C: forward returns dict -> look for common keys, else same-shape, else first tensor
            if isinstance(out, dict):
                for k in ("x_hat", "reconstruction", "act_hat", "decoded"):
                    if k in out and isinstance(out[k], t.Tensor):
                        return out[k].to(dtype=x.dtype, device=x.device)

                for v in out.values():
                    if isinstance(v, t.Tensor) and v.shape == x.shape:
                        return v.to(dtype=x.dtype, device=x.device)

                for v in out.values():
                    if isinstance(v, t.Tensor):
                        return v.to(dtype=x.dtype, device=x.device)
        except Exception:
            pass

        # 3) Last resort retry: encode() then decode() if only decode was missing earlier
        try:
            if hasattr(dictionary, "encode"):
                f = dictionary.encode(x)
                if hasattr(dictionary, "decode"):
                    x_hat = dictionary.decode(f)
                    return x_hat.to(dtype=x.dtype, device=x.device)
        except Exception:
            pass

        # 4) Total fallback: identity passthrough (warn once)
        if not hasattr(reconstruct_with_dictionary, "_warned"):
            print(
                "[Warn] SAE reconstruction fell back to identity; "
                "check dictionary.forward/encode/decode APIs.",
                flush=True,
            )
            reconstruct_with_dictionary._warned = True
        return x


def register_sae_splice_hook(submodule: t.nn.Module, dictionary: t.nn.Module, io: str = "out"):
    """
    Register a hook that splices SAE reconstruction into the model.

    We modify the *activations* flowing through `submodule`:
      - If io == "out": we edit the submodule's OUTPUT.
      - If io == "in": we edit the submodule's INPUT.

    We robustly handle modules that return tuples/dicts/etc.
    """

    if io == "out":
        def _hook(_, __, output):
            act = _first_tensor(output)
            if act is None:
                return output
            act_hat = reconstruct_with_dictionary(dictionary, act)
            new_output = _replace_first_tensor(output, act_hat)
            return new_output

        return submodule.register_forward_hook(_hook)

    elif io == "in":
        def _pre_hook(_, inputs):
            if len(inputs) == 0:
                return inputs
            x0 = inputs[0]
            act = _first_tensor(x0)
            if act is None:
                return inputs
            act_hat = reconstruct_with_dictionary(dictionary, act)
            new_x0 = _replace_first_tensor(x0, act_hat)
            return (new_x0,) + tuple(inputs[1:])

        return submodule.register_forward_pre_hook(_pre_hook)

    else:
        raise ValueError("io must be 'in' or 'out'")


############################################
# Tokenization / masking / timestep helpers
############################################

def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_len: int,
    device: t.device,
    add_special_tokens: bool = True,
) -> Dict[str, t.Tensor]:
    """
    Tokenize a batch of raw strings -> padded tensors on target device.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=add_special_tokens,
    )
    return {k: v.to(device) for k, v in enc.items()}


def build_random_mask(
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
    mask_prob: float,
    exclude_first_token: bool = True,
) -> t.Tensor:
    """
    Randomly select a subset of *non-padding* token positions to mask.
    Optionally skip the very first token in each sequence.
    """
    B, T = input_ids.shape
    cand = attention_mask.bool().clone()
    if exclude_first_token and T > 0:
        cand[:, 0] = False
    rand = t.rand_like(input_ids, dtype=t.float32)
    m = (rand < mask_prob) & cand
    return m


def _make_additive_float_mask_from_1d(attn_1d: t.Tensor, dtype: t.dtype) -> t.Tensor:
    """
    Build additive float mask for SDPA from (B, T) 0/1 mask.
    Return shape (B, 1, 1, T). Keep tokens -> 0.0, pad -> very negative.
    """
    am = attn_1d.to(dtype)
    am4 = am[:, None, None, :]
    minus_inf_val = t.finfo(am4.dtype).min
    add_mask = t.where(am4 > 0, t.zeros_like(am4), t.full_like(am4, minus_inf_val))
    return add_mask


def _safe_forward_with_masks(model, inputs: Dict[str, t.Tensor], prefer_additive: bool = True):
    """
    Robust forward pass, esp. for Dream/diffusion-like attention behaviors.
    We'll try multiple attention_mask formats.
    """
    try:
        param_dtype = next(model.parameters()).dtype
    except Exception:
        param_dtype = t.float32

    # 0) Prefer additive bias
    if prefer_additive and "attention_mask" in inputs:
        try:
            inp = dict(inputs)
            add_mask = _make_additive_float_mask_from_1d(inp["attention_mask"], dtype=param_dtype)
            add_mask = add_mask.to(device=inp["attention_mask"].device)
            inp["attention_mask"] = add_mask
            return model(**inp)
        except Exception:
            pass

    # 1) Bool mask
    try:
        inp = dict(inputs)
        if "attention_mask" in inp:
            inp["attention_mask"] = inp["attention_mask"].to(t.bool)
        return model(**inp)
    except Exception:
        pass

    # 2) No mask
    try:
        inp = dict(inputs)
        inp.pop("attention_mask", None)
        return model(**inp)
    except Exception:
        pass

    # 3) Last fallback: additive again
    inp = dict(inputs)
    if "attention_mask" in inp:
        add_mask = _make_additive_float_mask_from_1d(inp["attention_mask"], dtype=param_dtype)
        add_mask = add_mask.to(device=inp["attention_mask"].device)
        inp["attention_mask"] = add_mask
    return model(**inp)


def _maybe_add_time_condition(inputs: Dict[str, t.Tensor], p_scalar: float, model) -> Dict[str, t.Tensor]:
    """
    If the model's forward signature appears to accept a timestep/noise argument,
    add a per-sequence time-conditioning tensor derived from mask probability p.

    We detect common parameter names via inspect.signature(model.forward).
    - 't', 'time', 'timestep', 'timesteps'  (float or int)
    - 'noise_level', 'sigma'
    - 'diffusion_step'
    - 'time_ids' (int)

    For integer step-style parameters we map p in [0,1] to step in [0, T-1],
    where T is guessed from config if available (fallback T=1000).
    """
    try:
        sig = inspect.signature(model.forward)
        param_names = set(sig.parameters.keys())
    except Exception:
        # If we can't inspect, do nothing.
        return inputs

    # Derive a step index if needed
    T = None
    cfg = getattr(model, "config", None)
    for cand in ("num_diffusion_steps", "diffusion_steps", "n_timesteps", "timesteps", "T"):
        if hasattr(cfg, cand) and isinstance(getattr(cfg, cand), int):
            T = int(getattr(cfg, cand))
            break
    if T is None:
        T = 1000
    step = int(max(0, min(T - 1, round(p_scalar * (T - 1)))))

    B = inputs["input_ids"].shape[0]
    dev = inputs["input_ids"].device

    def add(name: str, tensor: t.Tensor):
        new_inputs = dict(inputs)
        new_inputs[name] = tensor
        return new_inputs

    # Float variants
    if "t" in param_names:
        return add("t", t.full((B,), float(p_scalar), dtype=t.float32, device=dev))
    if "time" in param_names:
        return add("time", t.full((B,), float(p_scalar), dtype=t.float32, device=dev))
    if "noise_level" in param_names:
        return add("noise_level", t.full((B,), float(p_scalar), dtype=t.float32, device=dev))
    if "sigma" in param_names:
        return add("sigma", t.full((B,), float(p_scalar), dtype=t.float32, device=dev))

    # Integer step variants
    if "timestep" in param_names:
        return add("timestep", t.full((B,), step, dtype=t.long, device=dev))
    if "timesteps" in param_names:
        return add("timesteps", t.full((B,), step, dtype=t.long, device=dev))
    if "diffusion_step" in param_names:
        return add("diffusion_step", t.full((B,), step, dtype=t.long, device=dev))
    if "time_ids" in param_names:
        return add("time_ids", t.full((B,), step, dtype=t.long, device=dev))

    # Nothing matched -> leave as-is
    return inputs


def ce_sum_with_mask(
    logits: t.Tensor,
    labels: t.Tensor,
    mask: t.Tensor,
    weight_scalar: Optional[float] = None,
) -> Tuple[t.Tensor, int]:
    """
    Compute token-level cross entropy on 'mask' positions only, returning SUM over those positions.

    Returns:
      loss_sum: scalar tensor (sum CE over selected positions, optionally scaled by weight_scalar).
      n:        int number of selected tokens.

    We use sum here so we can accumulate across batches exactly and divide once at the end.
    """
    n = int(mask.sum().item())
    if n == 0:
        return t.zeros((), device=labels.device, dtype=t.float32), 0

    if not isinstance(logits, t.Tensor):
        raise RuntimeError(f"ce_sum_with_mask expected logits as Tensor but got {type(logits)}")

    per_tok_loss = F.cross_entropy(logits[mask], labels[mask], reduction="none")  # (n,)
    if weight_scalar is not None:
        per_tok_loss = per_tok_loss * float(weight_scalar)
    loss_sum = per_tok_loss.sum()
    return loss_sum, n


############################################
# Main batch loss eval helper (DLM-aligned)
############################################

@t.no_grad()
def dlm_batch_losses(
    model,
    tokenizer,
    submodule,
    dictionary,
    texts: List[str],
    max_len: int,
    mask_token_id: int,
    device: t.device,
    io: str = "out",
    t_min: float = 0.05,
    t_max: float = 0.50,
    fixed_t: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, t.Tensor]:
    """
    For a batch of texts, evaluate two ΔLM losses:

      A) mask-only (task-aligned):
         - Sample t ~ U[t_min, t_max] (unless fixed_t is given).
         - Mask tokens independently with prob t (absorbing [MASK]).
         - Loss computed only on masked positions, weighted by 1/t.

      B) unmask-only (representation-aligned auxiliary):
         - Loss computed only on unmasked (valid & not masked) positions.
         - Weighted by 1/(1 - t) to stabilize averaging across t.
         - First token excluded to align with masking construction.

      Both are reported for clean model and SAE-spliced model.
    """
    # 1) pick t for this batch
    if fixed_t is not None:
        t_prob = float(fixed_t)
    else:
        t_prob = float(t.empty((), device=device).uniform_(t_min, t_max).item())
    t_prob = max(1e-6, min(0.999, t_prob))  # clamp for safety
    inv_t_weight = 1.0 / t_prob
    inv_unmask_weight = 1.0 / max(1e-6, (1.0 - t_prob))

    # 2) tokenize + mask
    batch = tokenize_batch(
        tokenizer,
        texts,
        max_len=max_len,
        device=device,
        add_special_tokens=True,
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    m = build_random_mask(
        input_ids,
        attention_mask,
        mask_prob=t_prob,
        exclude_first_token=True,
    )
    masked_ids = input_ids.clone()
    masked_ids[m] = mask_token_id

    # Build unmask selection (exclude first token to mirror 'm' logic)
    unmask_sel = (~m) & attention_mask.bool()
    if unmask_sel.shape[1] > 0:
        unmask_sel[:, 0] = False

    # Common inputs (+ optional time conditioning)
    base_inputs = {"input_ids": masked_ids, "attention_mask": attention_mask}
    base_inputs = _maybe_add_time_condition(base_inputs, p_scalar=t_prob, model=model)

    # 3) clean forward
    outputs_clean = _safe_forward_with_masks(model, base_inputs, prefer_additive=True)
    logits_clean = get_logits(outputs_clean)

    # 4) SAE forward (splice)
    handle = register_sae_splice_hook(submodule, dictionary, io=io)
    try:
        outputs_sae = _safe_forward_with_masks(model, base_inputs, prefer_additive=True)
        logits_sae = get_logits(outputs_sae)
    finally:
        handle.remove()

    # --- A) mask-only ---
    loss_clean_mask_sum, n_mask = ce_sum_with_mask(logits_clean, input_ids, m, weight_scalar=inv_t_weight)
    loss_sae_mask_sum, _       = ce_sum_with_mask(logits_sae,   input_ids, m, weight_scalar=inv_t_weight)

    # --- B) unmask-only ---
    loss_clean_unmask_sum, n_unmask = ce_sum_with_mask(
        logits_clean, input_ids, unmask_sel, weight_scalar=inv_unmask_weight
    )
    loss_sae_unmask_sum, _ = ce_sum_with_mask(
        logits_sae, input_ids, unmask_sel, weight_scalar=inv_unmask_weight
    )

    if verbose:
        B, T = input_ids.shape
        print(
            f"[ΔLoss-Batch] t={t_prob:.3f}  BxT={B}x{T}  "
            f"masked={n_mask}  unmask={n_unmask}  "
            f"mask(clean/sae)={float(loss_clean_mask_sum.item()):.3f}/{float(loss_sae_mask_sum.item()):.3f}  "
            f"unmask(clean/sae)={float(loss_clean_unmask_sum.item()):.3f}/{float(loss_sae_unmask_sum.item()):.3f}",
            flush=True,
        )

    return {
        # mask-only sums + count
        "loss_clean_mask_sum": loss_clean_mask_sum,
        "loss_sae_mask_sum": loss_sae_mask_sum,
        "n_masked_tokens": t.tensor(n_mask, device=device),
        # unmask-only sums + count
        "loss_clean_unmask_sum": loss_clean_unmask_sum,
        "loss_sae_unmask_sum": loss_sae_unmask_sum,
        "n_unmasked_tokens": t.tensor(n_unmask, device=device),
        # bookkeeping
        "t_used": t.tensor(t_prob, device=device, dtype=t.float32),
    }


############################################
# Held-out stream helper
############################################

def heldout_stream(dataset: str, skip_first_n_examples: int = 0) -> Iterable[str]:
    """
    Build a generator over held-out evaluation text from `dataset`.

    Because our dataset only has a single split ("train"),
    we simulate "held-out" by skipping the first `skip_first_n_examples`
    examples (which you assume were used during SAE training),
    and then yielding subsequent examples.

    This matches the "same distribution, held-out slice" idea
    from Llama Scope / Gemma Scope.
    """
    base_gen = iter_texts(dataset_name=dataset, split="train")

    # Drop the first N examples
    skipped = 0
    while skipped < skip_first_n_examples:
        try:
            next(base_gen)
        except StopIteration:
            break
        skipped += 1

    # Yield the rest forever (or until StopIteration)
    for txt in base_gen:
        yield txt


def find_sae_trainer_dirs(root: str) -> List[str]:
    """
    Any directory whose basename starts with 'trainer_' is treated
    as an SAE checkpoint directory. We return the absolute paths
    to those folders, sorted for reproducibility.
    """
    trainer_dirs: List[str] = []

    # Walk the entire subtree under `root`
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        # Check if this directory itself is a trainer directory
        if base.startswith("trainer_") and os.path.isdir(dirpath):
            trainer_dirs.append(dirpath)

    # Sort so results are deterministic across runs
    trainer_dirs = sorted(trainer_dirs)
    return trainer_dirs


############################################
# Main entry point
############################################

def main():
    parser = argparse.ArgumentParser(
        "Compute Dream-ΔLoss (Delta LM loss for diffusion-like LMs) on a held-out slice."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ae_root", type=str, required=True)
    parser.add_argument("--token_budget", type=int, default=50_000_000)
    parser.add_argument("--batch_size_text", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )

    # IMPORTANT: same distribution as SAE training
    parser.add_argument(
        "--heldout_dataset",
        type=str,
        default="common-pile/comma_v0.1_training_dataset",
        help="Dataset to stream eval text from. Must match SAE training distribution.",
    )
    parser.add_argument(
        "--skip_first_n_examples",
        type=int,
        default=0,
        help=(
            "Skip this many examples from the dataset stream before evaluation. "
            "Use this to avoid reusing the exact samples seen during SAE training."
        ),
    )

    # DLM timestep / masking config
    parser.add_argument("--t_min", type=float, default=0.05, help="Lower bound for per-batch t sampling.")
    parser.add_argument("--t_max", type=float, default=0.50, help="Upper bound for per-batch t sampling.")
    parser.add_argument(
        "--fixed_t",
        type=float,
        default=None,
        help="If set, use this fixed corruption level t instead of sampling t ~ U[t_min, t_max].",
    )
    parser.add_argument(
        "--io",
        type=str,
        default="out",
        choices=["in", "out"],
        help="Hook direction: splice SAE into submodule input or output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch status messages.",
    )
    args = parser.parse_args()

    ############################################
    # Warnings / deprecations handling
    ############################################
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )

    # Map string dtype -> torch dtype
    dtype_map = {"float32": t.float32, "bfloat16": t.bfloat16, "float16": t.float16}
    load_dtype = dtype_map[args.dtype]

    ############################################
    # Tokenizer
    ############################################
    print(f"[Setup] Loading tokenizer for {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ############################################
    # Model
    ############################################
    print("[Setup] Loading model (this can take minutes) ...", flush=True)

    # transformers API compatibility dance
    try:
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype=load_dtype,          # preferred in newer transformers
        ).eval()
    except TypeError:
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=load_dtype,    # fallback for older transformers
       ).eval()

    if not is_dream_like(model):
        print(
            "[Warn] The loaded model does not look like a Dream/diffusion LM; "
            "we will still use the DLM masking objective (masked-only CE with 1/t) as the task-aligned metric, "
            "and report an auxiliary unmask-only metric.",
            flush=True,
        )

    # Figure out which token to use for masking
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        # Some decoder-only models have no [MASK]; fallback to <unk> or <eos>
        fallback_id = (
            tokenizer.unk_token_id
            if getattr(tokenizer, "unk_token_id", None) is not None
            else tokenizer.eos_token_id
        )
        mask_token_id = fallback_id
        print(
            f"[Info] tokenizer.mask_token_id not found; using fallback id = {mask_token_id}",
            flush=True,
        )

    ############################################
    # Scan for all SAE checkpoints under ae_root
    ############################################
    print(f"[Scan] Scanning SAE folders under: {args.ae_root}", flush=True)
    sae_dirs: List[str] = find_sae_trainer_dirs(args.ae_root)
    if len(sae_dirs) == 0:
        print(
            "[Scan] No trainer_* folders found via recursive scan. "
            "Falling back to utils.get_nested_folders(...)",
            flush=True,
        )
        sae_dirs = utils.get_nested_folders(args.ae_root)

    print(f"[Scan] Found {len(sae_dirs)} SAE folders.", flush=True)
    if args.verbose:
        for d in sae_dirs:
            print(f"  - {d}", flush=True)

    def new_stream():
        # Same distribution as training, skip head for held-out
        return heldout_stream(
            dataset=args.heldout_dataset,
            skip_first_n_examples=args.skip_first_n_examples,
        )

    ############################################
    # Loop over each SAE, evaluate ΔLoss (mask & unmask)
    ############################################
    for idx, d in enumerate(sae_dirs, start=1):
        print(f"\n[Eval {idx}/{len(sae_dirs)}] Loading SAE from: {d}", flush=True)

        try:
            dictionary, cfg = utils.load_dictionary(d, device=args.device)
            model_dtype = next(model.parameters()).dtype
            dictionary.to(dtype=model_dtype)
            dictionary.eval()

            layer = cfg["trainer"]["layer"]
            submodule = utils.get_submodule(model, layer)
            print(f"[Eval] Target layer = {layer} | io = {args.io}", flush=True)

            # Aggregators (sums over tokens, then averaged at the end)
            sum_masked_tokens = 0
            sum_unmasked_tokens = 0

            loss_clean_mask_sum_total = 0.0
            loss_sae_mask_sum_total   = 0.0

            loss_clean_unmask_sum_total = 0.0
            loss_sae_unmask_sum_total   = 0.0

            stream = new_stream()

            pbar = tqdm(
                total=args.token_budget,
                unit="tok",
                dynamic_ncols=True,
                desc=f"ΔLoss(mask) tokens ({os.path.basename(d)})",
            )

            # Keep consuming text until we hit the token budget (measured on MASKED tokens)
            while sum_masked_tokens < args.token_budget:
                texts: List[str] = []
                for _ in range(args.batch_size_text):
                    try:
                        texts.append(next(stream))
                    except StopIteration:
                        break

                if not texts:
                    break  # dataset stream exhausted

                batch_losses = dlm_batch_losses(
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    dictionary=dictionary,
                    texts=texts,
                    max_len=args.max_len,
                    mask_token_id=mask_token_id,
                    device=t.device(args.device if ("cuda" in args.device or "cpu" in args.device) else "cpu"),
                    io=args.io,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    fixed_t=args.fixed_t,  # None -> random t ~ U[t_min, t_max]
                    verbose=args.verbose,
                )

                # Accumulate mask-only
                n_mask = int(batch_losses["n_masked_tokens"].item())
                if n_mask > 0:
                    loss_clean_mask_sum_total += float(batch_losses["loss_clean_mask_sum"].item())
                    loss_sae_mask_sum_total   += float(batch_losses["loss_sae_mask_sum"].item())
                    sum_masked_tokens         += n_mask
                    pbar.update(n_mask)

                # Accumulate unmask-only
                n_unmask = int(batch_losses["n_unmasked_tokens"].item())
                if n_unmask > 0:
                    loss_clean_unmask_sum_total += float(batch_losses["loss_clean_unmask_sum"].item())
                    loss_sae_unmask_sum_total   += float(batch_losses["loss_sae_unmask_sum"].item())
                    sum_unmasked_tokens         += n_unmask

            pbar.close()

            # Avoid division by zero just in case
            sum_masked_tokens   = max(sum_masked_tokens, 1)
            sum_unmasked_tokens = max(sum_unmasked_tokens, 1)

            # --- mask-only report (task-aligned) ---
            avg_clean_mask = loss_clean_mask_sum_total / sum_masked_tokens
            avg_sae_mask   = loss_sae_mask_sum_total   / sum_masked_tokens
            delta_mask     = avg_sae_mask - avg_clean_mask

            out_mask = {
                "tokens_masked_evaluated": float(sum_masked_tokens),
                "dream_weighted_loss_clean(mask)": float(avg_clean_mask),
                "dream_weighted_loss_sae(mask)": float(avg_sae_mask),
                "delta_lm_loss(mask)": float(delta_mask),
                "weighting": "mask-only CE weighted by 1/t",
                "t_min": float(args.t_min),
                "t_max": float(args.t_max),
                "fixed_t": (None if args.fixed_t is None else float(args.fixed_t)),
                "max_len": int(args.max_len),
                "batch_size_text": int(args.batch_size_text),
                "io": args.io,
                "heldout_dataset": args.heldout_dataset,
                "skip_first_n_examples": int(args.skip_first_n_examples),
            }

            # --- unmask-only report (auxiliary) ---
            avg_clean_unmask = loss_clean_unmask_sum_total / sum_unmasked_tokens
            avg_sae_unmask   = loss_sae_unmask_sum_total   / sum_unmasked_tokens
            delta_unmask     = avg_sae_unmask - avg_clean_unmask

            out_unmask = {
                "tokens_unmasked_evaluated": float(sum_unmasked_tokens),
                "dream_weighted_loss_clean(unmask)": float(avg_clean_unmask),
                "dream_weighted_loss_sae(unmask)": float(avg_sae_unmask),
                "delta_lm_loss(unmask)": float(delta_unmask),
                "weighting": "unmask-only CE weighted by 1/(1 - t), first token excluded",
                "t_min": float(args.t_min),
                "t_max": float(args.t_max),
                "fixed_t": (None if args.fixed_t is None else float(args.fixed_t)),
                "max_len": int(args.max_len),
                "batch_size_text": int(args.batch_size_text),
                "io": args.io,
                "heldout_dataset": args.heldout_dataset,
                "skip_first_n_examples": int(args.skip_first_n_examples),
            }

            # Save two files as requested
            out_path_mask = os.path.join(d, "delta_lm_loss(mask).json")
            out_path_unmask = os.path.join(d, "delta_lm_loss(unmask).json")
            with open(out_path_mask, "w") as f:
                json.dump(out_mask, f, indent=2)
            with open(out_path_unmask, "w") as f:
                json.dump(out_unmask, f, indent=2)

            print(
                f"[Done] {d} → ΔLM(mask)={out_mask['delta_lm_loss(mask)']:.6f}  "
                f"ΔLM(unmask)={out_unmask['delta_lm_loss(unmask)']:.6f}  "
                f"masked_tokens={int(out_mask['tokens_masked_evaluated']):,}  "
                f"unmasked_tokens={int(out_unmask['tokens_unmasked_evaluated']):,}",
                flush=True,
            )

        except Exception as e:
            print(f"[Eval] Failed on {d}: {e}", flush=True)


if __name__ == "__main__":
    main()
