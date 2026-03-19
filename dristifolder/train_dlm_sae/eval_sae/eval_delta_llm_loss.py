# eval_delta_lm_loss_qwen_ar.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
from typing import Iterable, List, Dict, Tuple, Optional, Any

import torch as t
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dictionary_learning.dictionary_learning import utils


# =========================================================
# General helpers
# =========================================================

def is_dream_like(model) -> bool:
    """
    Heuristic check for diffusion-style LMs (e.g., Dream-7B).
    """
    name = model.__class__.__name__.lower()
    return ("dream" in name) or ("diffusion" in name) or hasattr(model, "diffusion_generate")


def is_ar_like(model) -> bool:
    """
    Heuristic check for autoregressive decoder-only LMs.
    """
    # Many AR LMs expose `.generate` and are `AutoModelForCausalLM` instances
    name = model.__class__.__name__.lower()
    return ("causallm" in name) or hasattr(model, "generate")


def iter_texts(dataset_name: str, split: str = "train") -> Iterable[str]:
    """
    Stream raw text strings from a HF dataset via project utils.
    Assumes dataset at least has a 'train' split (we always read 'train').
    """
    return utils.hf_dataset_to_generator(dataset_name, split=split, streaming=True)


def _pick_best_tensor(cands: List[t.Tensor]) -> t.Tensor:
    """
    Choose a likely-logits tensor from candidates.
    Preference: (B, T, V) -> (T, V) -> first tensor.
    """
    rank3 = [x for x in cands if isinstance(x, t.Tensor) and x.ndim == 3]
    if rank3:
        return rank3[0]
    rank2 = [x for x in cands if isinstance(x, t.Tensor) and x.ndim == 2]
    if rank2:
        return rank2[0]
    for x in cands:
        if isinstance(x, t.Tensor):
            return x
    raise RuntimeError("No tensor candidate found among provided candidates.")


def get_logits(outputs: Any) -> t.Tensor:
    """
    Robustly extract a logits-like tensor from model outputs.
    Handles dict-like, tuple/list, or objects with .logits / .last_hidden_state.
    """
    if isinstance(outputs, dict):
        if "logits" in outputs and isinstance(outputs["logits"], t.Tensor):
            return outputs["logits"]
        if "last_hidden_state" in outputs and isinstance(outputs["last_hidden_state"], t.Tensor):
            return outputs["last_hidden_state"]
        cands = [v for v in outputs.values() if isinstance(v, t.Tensor)]
        if cands:
            return _pick_best_tensor(cands)
        nested = []
        for v in outputs.values():
            if isinstance(v, (tuple, list)):
                nested.extend([x for x in v if isinstance(x, t.Tensor)])
        if nested:
            return _pick_best_tensor(nested)

    if hasattr(outputs, "logits") and isinstance(outputs.logits, t.Tensor):
        return outputs.logits
    if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, t.Tensor):
        return outputs.last_hidden_state

    if isinstance(outputs, (tuple, list)):
        cands = [x for x in outputs if isinstance(x, t.Tensor)]
        if cands:
            return _pick_best_tensor(cands)

    raise RuntimeError(f"Cannot find logits in model outputs of type: {type(outputs)}")


# =========================================================
# Helpers to handle tuple/dict-ish activations in hooks
# =========================================================

def _first_tensor(obj: Any) -> Optional[t.Tensor]:
    """
    Recursively return the first torch.Tensor found inside `obj`.
    """
    if isinstance(obj, t.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found

    if isinstance(obj, dict):
        preferred_keys = (
            "hidden_states", "last_hidden_state", "x", "resid", "resid_post",
            "out", "activations", "act", "states", "x_hat", "reconstruction", "decoded",
        )
        for k in preferred_keys:
            if k in obj and isinstance(obj[k], t.Tensor):
                return obj[k]
        for v in obj.values():
            found = _first_tensor(v)
            if found is not None:
                return found

    if hasattr(obj, "__dict__"):
        return _first_tensor(vars(obj))

    return None


def _replace_first_tensor(structure: Any, new_tensor: t.Tensor) -> Any:
    """
    Return `structure` but with its first encountered Tensor replaced by `new_tensor`.
    Container type is preserved as much as possible.
    """
    if isinstance(structure, t.Tensor):
        return new_tensor

    if isinstance(structure, list):
        new_items, replaced = [], False
        for item in structure:
            if not replaced and _first_tensor(item) is not None:
                new_items.append(_replace_first_tensor(item, new_tensor))
                replaced = True
            else:
                new_items.append(item)
        return type(structure)(new_items)

    if isinstance(structure, tuple):
        new_items, replaced = [], False
        for item in structure:
            if not replaced and _first_tensor(item) is not None:
                new_items.append(_replace_first_tensor(item, new_tensor))
                replaced = True
            else:
                new_items.append(item)
        return type(structure)(new_items)

    if isinstance(structure, dict):
        new_dict, replaced = {}, False
        for k, v in structure.items():
            if (not replaced) and _first_tensor(v) is not None:
                new_dict[k] = _replace_first_tensor(v, new_tensor)
                replaced = True
            else:
                new_dict[k] = v
        return new_dict

    if hasattr(structure, "__dict__"):
        attrs = vars(structure).copy()
        replaced = False
        for k, v in attrs.items():
            if (not replaced) and _first_tensor(v) is not None:
                attrs[k] = _replace_first_tensor(v, new_tensor)
                replaced = True
        return attrs

    return new_tensor


# =========================================================
# Device / dtype alignment
# =========================================================

def _move_module_like_tensor(module: t.nn.Module, ref_like: Any):
    """
    Move a module to the same (device, dtype) as `ref_like`.
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
    Return SAE reconstruction for activation x. Tries encode/decode first,
    falls back to forward-based heuristics. Matches dtype/device of x.
    """
    with t.no_grad():
        _move_module_like_tensor(dictionary, x)

        # Preferred: encode/decode
        if hasattr(dictionary, "encode") and hasattr(dictionary, "decode"):
            try:
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                return x_hat.to(dtype=x.dtype, device=x.device)
            except Exception:
                pass

        # Fallback: forward -> pick a plausible reconstruction tensor
        try:
            out = dictionary(x)
            if isinstance(out, t.Tensor):
                return out.to(dtype=x.dtype, device=x.device)
            if isinstance(out, (list, tuple)):
                same = [o for o in out if isinstance(o, t.Tensor) and o.shape == x.shape]
                if same:
                    return same[0].to(dtype=x.dtype, device=x.device)
                tensors = [o for o in out if isinstance(o, t.Tensor)]
                if tensors:
                    return tensors[0].to(dtype=x.dtype, device=x.device)
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

        # Last retry: encode then decode if possible
        try:
            if hasattr(dictionary, "encode"):
                f = dictionary.encode(x)
                if hasattr(dictionary, "decode"):
                    x_hat = dictionary.decode(f)
                    return x_hat.to(dtype=x.dtype, device=x.device)
        except Exception:
            pass

        if not hasattr(reconstruct_with_dictionary, "_warned"):
            print("[Warn] SAE reconstruction fell back to identity; check SAE API.", flush=True)
            reconstruct_with_dictionary._warned = True
        return x


def register_sae_splice_hook(submodule: t.nn.Module, dictionary: t.nn.Module, io: str = "out"):
    """
    Register a hook that splices SAE reconstructions into a module's activations.
    """
    if io == "out":
        def _hook(_, __, output):
            act = _first_tensor(output)
            if act is None:
                return output
            act_hat = reconstruct_with_dictionary(dictionary, act)
            return _replace_first_tensor(output, act_hat)
        return submodule.register_forward_hook(_hook)

    if io == "in":
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

    raise ValueError("io must be 'in' or 'out'")


# =========================================================
# Forward helpers (AR and Dream)
# =========================================================

def _make_additive_float_mask_from_1d(attn_1d: t.Tensor, dtype: t.dtype) -> t.Tensor:
    """
    Build additive float mask for SDPA from (B, T) 0/1 mask.
    Keep tokens -> 0.0, pad -> very negative (min for dtype).
    """
    am = attn_1d.to(dtype)
    am4 = am[:, None, None, :]
    minus_inf_val = t.finfo(am4.dtype).min
    add_mask = t.where(am4 > 0, t.zeros_like(am4), t.full_like(am4, minus_inf_val))
    return add_mask


def _safe_forward_with_masks(model, inputs: Dict[str, t.Tensor], prefer_additive: bool = True):
    """
    Forward with robust attention_mask handling (useful for Dream-like LMs).
    """
    try:
        param_dtype = next(model.parameters()).dtype
    except Exception:
        param_dtype = t.float32

    if prefer_additive and "attention_mask" in inputs:
        try:
            inp = dict(inputs)
            add_mask = _make_additive_float_mask_from_1d(inp["attention_mask"], dtype=param_dtype)
            add_mask = add_mask.to(device=inp["attention_mask"].device)
            inp["attention_mask"] = add_mask
            return model(**inp)
        except Exception:
            pass

    try:
        inp = dict(inputs)
        if "attention_mask" in inp:
            inp["attention_mask"] = inp["attention_mask"].to(t.bool)
        return model(**inp)
    except Exception:
        pass

    try:
        inp = dict(inputs)
        inp.pop("attention_mask", None)
        return model(**inp)
    except Exception:
        pass

    inp = dict(inputs)
    if "attention_mask" in inp:
        add_mask = _make_additive_float_mask_from_1d(inp["attention_mask"], dtype=param_dtype)
        add_mask = add_mask.to(device=inp["attention_mask"].device)
        inp["attention_mask"] = add_mask
    return model(**inp)


# ---------------- AR (autoregressive) loss ----------------

def next_token_ce(
    logits: t.Tensor,
    input_ids: t.Tensor,
    attention_mask: t.Tensor,
) -> Tuple[t.Tensor, int]:
    """
    Compute next-token cross entropy over non-padding positions.

    We shift:
      - logits: use logits[:, :-1, :]
      - labels: use input_ids[:, 1:]

    Valid positions mask: attention_mask[:, 1:] == 1
    """
    if not isinstance(logits, t.Tensor):
        raise RuntimeError("next_token_ce expects logits as Tensor.")

    # Align shapes
    logits = logits[:, :-1, :]                 # (B, T-1, V)
    labels = input_ids[:, 1:].contiguous()     # (B, T-1)
    mask = attention_mask[:, 1:].to(t.bool)    # (B, T-1)

    if mask.sum().item() == 0:
        return t.zeros((), device=logits.device, dtype=t.float32), 0

    B, Tp, V = logits.shape
    logits_flat = logits.reshape(B * Tp, V)
    labels_flat = labels.reshape(B * Tp)
    mask_flat = mask.reshape(B * Tp)

    # Keep only valid positions
    logits_sel = logits_flat[mask_flat]        # (N, V)
    labels_sel = labels_flat[mask_flat]        # (N,)
    loss = F.cross_entropy(logits_sel, labels_sel, reduction="mean")
    n = int(mask_flat.sum().item())
    return loss, n


@t.no_grad()
def ar_batch_losses(
    model,
    tokenizer,
    submodule,
    dictionary,
    texts: List[str],
    max_len: int,
    device: t.device,
    io: str = "out",
) -> Dict[str, t.Tensor]:
    """
    For a batch of texts on an AR LM:
      1) Tokenize
      2) Compute next-token CE clean
      3) Splice SAE and compute next-token CE again
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    input_ids: t.Tensor = inputs["input_ids"]
    attention_mask: t.Tensor = inputs["attention_mask"]

    # Clean forward
    outputs_clean = model(**inputs)
    logits_clean = get_logits(outputs_clean)
    loss_clean, n_tok = next_token_ce(logits_clean, input_ids, attention_mask)

    # SAE-spliced forward
    handle = register_sae_splice_hook(submodule, dictionary, io=io)
    try:
        outputs_sae = model(**inputs)
        logits_sae = get_logits(outputs_sae)
        loss_sae, _ = next_token_ce(logits_sae, input_ids, attention_mask)
    finally:
        handle.remove()

    return {
        "loss_clean": loss_clean,
        "loss_sae": loss_sae,
        "n_eval_tokens": t.tensor(n_tok, device=device),
    }


# ---------------- Dream (diffusion) loss ----------------

def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_len: int,
    device: t.device,
    add_special_tokens: bool = True,
) -> Dict[str, t.Tensor]:
    """
    Tokenize a batch of strings -> padded tensors on target device.
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
    Randomly select a subset of non-padding token positions to mask.
    """
    B, T = input_ids.shape
    cand = attention_mask.bool().clone()
    if exclude_first_token and T > 0:
        cand[:, 0] = False
    rand = t.rand_like(input_ids, dtype=t.float32)
    m = (rand < mask_prob) & cand
    return m


def masked_ce(
    logits: t.Tensor,
    labels: t.Tensor,
    mask: t.Tensor,
) -> Tuple[t.Tensor, int]:
    """
    Compute token-level CE only on masked positions (Dream-style eval).
    """
    n = int(mask.sum().item())
    if n == 0:
        return t.zeros((), device=labels.device, dtype=t.float32), 0
    logits_2d = logits[mask]    # (n, V)
    labels_1d = labels[mask]    # (n,)
    loss = F.cross_entropy(logits_2d, labels_1d, reduction="mean")
    return loss, n


@t.no_grad()
def dream_batch_losses(
    model,
    tokenizer,
    submodule,
    dictionary,
    texts: List[str],
    max_len: int,
    mask_token_id: int,
    mask_prob: float,
    device: t.device,
    io: str = "out",
) -> Dict[str, t.Tensor]:
    """
    Dream/diffusion-style batch eval:
      1) Tokenize + randomly mask some tokens -> masked_ids
      2) Clean forward on masked_ids
      3) SAE-spliced forward on masked_ids
      4) Compute CE only on masked positions
    """
    batch = tokenize_batch(tokenizer, texts, max_len=max_len, device=device, add_special_tokens=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    m = build_random_mask(input_ids, attention_mask, mask_prob=mask_prob, exclude_first_token=True)
    masked_ids = input_ids.clone()
    masked_ids[m] = mask_token_id

    outputs_clean = _safe_forward_with_masks(model, {"input_ids": masked_ids, "attention_mask": attention_mask}, True)
    logits_clean = get_logits(outputs_clean)
    loss_clean, n_tok = masked_ce(logits_clean, input_ids, m)

    handle = register_sae_splice_hook(submodule, dictionary, io=io)
    try:
        outputs_sae = _safe_forward_with_masks(model, {"input_ids": masked_ids, "attention_mask": attention_mask}, True)
        logits_sae = get_logits(outputs_sae)
        loss_sae, _ = masked_ce(logits_sae, input_ids, m)
    finally:
        handle.remove()

    return {
        "loss_clean": loss_clean,
        "loss_sae": loss_sae,
        "n_eval_tokens": t.tensor(n_tok, device=device),
    }


# =========================================================
# Held-out stream helper
# =========================================================

def heldout_stream(dataset: str, skip_first_n_examples: int = 0) -> Iterable[str]:
    """
    Build a generator over held-out evaluation text from `dataset`.

    Since the dataset only has a single split ("train"), we simulate a
    held-out region by skipping the first `skip_first_n_examples` examples
    (assumed to have been used during SAE training) and then yielding the rest.
    """
    base_gen = iter_texts(dataset_name=dataset, split="train")

    skipped = 0
    while skipped < skip_first_n_examples:
        try:
            next(base_gen)
        except StopIteration:
            break
        skipped += 1

    for txt in base_gen:
        yield txt


def find_sae_trainer_dirs(root: str) -> List[str]:
    """
    Recursively find SAE checkpoint directories whose basename starts with 'trainer_'.
    """
    trainer_dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith("trainer_") and os.path.isdir(dirpath):
            trainer_dirs.append(dirpath)
    return sorted(trainer_dirs)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        "Compute Delta LM loss for SAEs on AR LMs (Qwen2.5-7B) with auto Dream fallback."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ae_root", type=str, required=True)
    parser.add_argument("--token_budget", type=int, default=1_000_000)
    parser.add_argument("--batch_size_text", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--heldout_dataset",
        type=str,
        default="common-pile/comma_v0.1_training_dataset",
        help="Eval dataset stream. Should match SAE training distribution.",
    )
    parser.add_argument(
        "--skip_first_n_examples",
        type=int,
        default=0,
        help="Skip this many examples to simulate a held-out slice.",
    )
    parser.add_argument("--mask_prob", type=float, default=0.3, help="Used only for Dream-like evaluation.")
    parser.add_argument(
        "--io", type=str, default="out", choices=["in", "out"],
        help="Hook direction: splice SAE into submodule input or output.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Silence noisy torch.load warning about weights_only default flips
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )

    # Map string -> torch dtype
    dtype_map = {"float32": t.float32, "bfloat16": t.bfloat16, "float16": t.float16}
    load_dtype = dtype_map[args.dtype]

    # ---------------- Tokenizer ----------------
    print(f"[Setup] Loading tokenizer for {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    # Ensure pad token exists (common for decoder-only LMs to miss it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Model ----------------
    print("[Setup] Loading model (this can take minutes) ...", flush=True)

    model = None
    load_err = None
    # Prefer AR head if available (better logits behavior for Qwen2.5)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=load_dtype,   # works across transformers versions
        ).eval()
    except Exception as e:
        load_err = e

    if model is None:
        # Fallback to generic AutoModel
        try:
            model = AutoModel.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=load_dtype,
            ).eval()
        except Exception as e2:
            raise RuntimeError(f"Failed to load model as CausalLM ({load_err}) and as AutoModel ({e2}).")

    # Determine evaluation mode
    use_dream_eval = is_dream_like(model) and not is_ar_like(model)
    if use_dream_eval:
        print("[Mode] Detected Dream/diffusion-like model → using masked-token CE protocol.", flush=True)
    else:
        print("[Mode] Detected autoregressive model → using next-token CE protocol.", flush=True)

    # Mask token id (only used for Dream-style)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if use_dream_eval and mask_token_id is None:
        fallback_id = tokenizer.unk_token_id if getattr(tokenizer, "unk_token_id", None) is not None else tokenizer.eos_token_id
        mask_token_id = fallback_id
        print(f"[Info] tokenizer.mask_token_id not found; using fallback id = {mask_token_id}", flush=True)

    # ---------------- Scan SAE dirs ----------------
    print(f"[Scan] Scanning SAE folders under: {args.ae_root}", flush=True)
    sae_dirs: List[str] = find_sae_trainer_dirs(args.ae_root)
    if len(sae_dirs) == 0:
        print("[Scan] No trainer_* folders found. Falling back to utils.get_nested_folders(...)", flush=True)
        sae_dirs = utils.get_nested_folders(args.ae_root)
    print(f"[Scan] Found {len(sae_dirs)} SAE folders.", flush=True)
    if args.verbose:
        for d in sae_dirs:
            print(f"  - {d}", flush=True)

    def new_stream():
        return heldout_stream(
            dataset=args.heldout_dataset,
            skip_first_n_examples=args.skip_first_n_examples,
        )

    # ---------------- Evaluate each SAE ----------------
    for idx, d in enumerate(sae_dirs, start=1):
        print(f"\n[Eval {idx}/{len(sae_dirs)}] Loading SAE from: {d}", flush=True)
        try:
            dictionary, cfg = utils.load_dictionary(d, device=args.device)

            # Match SAE dtype to model compute dtype
            model_dtype = next(model.parameters()).dtype
            dictionary.to(dtype=model_dtype)
            dictionary.eval()

            # Locate target submodule to splice into
            layer = cfg["trainer"]["layer"]
            submodule = utils.get_submodule(model, layer)
            print(f"[Eval] Target layer = {layer} | io = {args.io}", flush=True)

            sum_tokens = 0
            w_loss_clean = 0.0
            w_loss_sae = 0.0
            stream = new_stream()

            pbar = tqdm(
                total=args.token_budget,
                unit="tok",
                dynamic_ncols=True,
                desc=f"ΔLoss tokens ({os.path.basename(d)})",
            )

            while sum_tokens < args.token_budget:
                texts: List[str] = []
                for _ in range(args.batch_size_text):
                    try:
                        texts.append(next(stream))
                    except StopIteration:
                        break
                if not texts:
                    break

                if use_dream_eval:
                    batch_losses = dream_batch_losses(
                        model=model,
                        tokenizer=tokenizer,
                        submodule=submodule,
                        dictionary=dictionary,
                        texts=texts,
                        max_len=args.max_len,
                        mask_token_id=mask_token_id,
                        mask_prob=args.mask_prob,
                        device=t.device(args.device if ("cuda" in args.device or "cpu" in args.device) else "cpu"),
                        io=args.io,
                    )
                else:
                    batch_losses = ar_batch_losses(
                        model=model,
                        tokenizer=tokenizer,
                        submodule=submodule,
                        dictionary=dictionary,
                        texts=texts,
                        max_len=args.max_len,
                        device=t.device(args.device if ("cuda" in args.device or "cpu" in args.device) else "cpu"),
                        io=args.io,
                    )

                n = int(batch_losses["n_eval_tokens"].item())
                if n == 0:
                    continue

                w_loss_clean += float(batch_losses["loss_clean"].item()) * n
                w_loss_sae += float(batch_losses["loss_sae"].item()) * n
                sum_tokens += n
                pbar.update(n)

            pbar.close()
            sum_tokens = max(sum_tokens, 1)

            avg_clean = w_loss_clean / sum_tokens
            avg_sae = w_loss_sae / sum_tokens
            delta = avg_sae - avg_clean

            # Output keys keep "lm" naming for AR and "dream" naming for Dream-like
            if use_dream_eval:
                out = {
                    "tokens_evaluated": float(sum_tokens),
                    "dream_loss_clean": float(avg_clean),
                    "dream_loss_sae": float(avg_sae),
                    "delta_dream_loss": float(delta),
                    "mask_prob": float(args.mask_prob),
                    "max_len": int(args.max_len),
                    "batch_size_text": int(args.batch_size_text),
                    "io": args.io,
                    "heldout_dataset": args.heldout_dataset,
                    "skip_first_n_examples": int(args.skip_first_n_examples),
                }
                out_path = os.path.join(d, "delta_eval_dream.json")
            else:
                out = {
                    "tokens_evaluated": float(sum_tokens),
                    "lm_loss_clean": float(avg_clean),
                    "lm_loss_sae": float(avg_sae),
                    "delta_lm_loss": float(delta),
                    "max_len": int(args.max_len),
                    "batch_size_text": int(args.batch_size_text),
                    "io": args.io,
                    "heldout_dataset": args.heldout_dataset,
                    "skip_first_n_examples": int(args.skip_first_n_examples),
                }
                out_path = os.path.join(d, "delta_eval_ar.json")

            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            tag = "ΔDreamLoss" if use_dream_eval else "ΔLMLoss"
            print(f"[Done] {d} → {tag}={delta:.6f}  tokens={int(sum_tokens):,}", flush=True)

        except Exception as e:
            print(f"[Eval] Failed on {d}: {e}", flush=True)


if __name__ == "__main__":
    main()
