"""
Utilities for evaluating dictionaries (SAEs) on a model and dataset.
This version is adapted to work directly with Hugging Face Transformers models,
and is robust to Dream/DLM SDPA attention_mask dtype requirements.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Tuple, Optional

import torch as t

# Import the activation buffer that we actually use in the Dream pipeline
# (implemented with Transformers, not nnsight).
from .pytorch_buffer import ActivationBuffer

# -----------------------------------------------------------------------------
# Mask utilities (mirror the robust handling in pytorch_buffer.py)
# -----------------------------------------------------------------------------

def _make_additive_float_mask_from_1d(am: t.Tensor) -> t.Tensor:
    """
    Convert a (batch, seqlen) 0/1 mask into an additive float mask of shape (batch, 1, 1, seqlen),
    where keep positions -> 0.0 and pad positions -> -inf, which SDPA interprets as masked.
    """
    if am.dtype not in (t.float32, t.float16, t.bfloat16):
        am = am.float()
    am4 = am[:, None, None, :]
    finfo = t.finfo(am4.dtype)
    minus_inf = finfo.min
    add_mask = t.where(am4 > 0, t.zeros_like(am4), t.full_like(am4, minus_inf))
    return add_mask


def _extract_logits(model_output: Any) -> t.Tensor:
    """
    Extract logits from various HF outputs. Prefer `.logits`. If the output is a Tensor, assume it is logits.
    """
    if isinstance(model_output, t.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output and isinstance(model_output["logits"], t.Tensor):
        return model_output["logits"]
    # Fallback: try to locate a tensor of rank-3 as logits
    if isinstance(model_output, dict):
        for v in model_output.values():
            if isinstance(v, t.Tensor) and v.ndim == 3:
                return v
    raise TypeError(f"Cannot find logits in model output of type {type(model_output)}")


def _model_forward_logits_three_ways(model, inputs: Dict[str, t.Tensor]) -> t.Tensor:
    """
    Run a forward pass robustly w.r.t. Dream/SDPA attention_mask requirements and return logits.
      1) Try with attention_mask cast to bool.
      2) Try without attention_mask.
      3) Try with additive float mask (pad -> -inf).
    """
    # 1) bool mask
    try:
        with t.no_grad():
            inp = dict(inputs)
            if "attention_mask" in inp:
                inp["attention_mask"] = inp["attention_mask"].to(t.bool)
            out = model(**inp)
            return _extract_logits(out)
    except Exception as e:
        print(f"[Eval Forward 1: bool mask] failed: {type(e).__name__}: {e}")

    # 2) no mask
    try:
        with t.no_grad():
            inp = dict(inputs)
            inp.pop("attention_mask", None)
            out = model(**inp)
            return _extract_logits(out)
    except Exception as e:
        print(f"[Eval Forward 2: no mask] failed: {type(e).__name__}: {e}")

    # 3) additive float mask
    try:
        with t.no_grad():
            inp = dict(inputs)
            if "attention_mask" in inp:
                add_mask = _make_additive_float_mask_from_1d(inp["attention_mask"])
                inp["attention_mask"] = add_mask.to(device=add_mask.device)
            out = model(**inp)
            return _extract_logits(out)
    except Exception as e:
        print(f"[Eval Forward 3: additive float mask] failed: {type(e).__name__}: {e}")
        raise


# -----------------------------------------------------------------------------
# Hook helpers for intervention (replace submodule activations)
# -----------------------------------------------------------------------------

def _normalize_hook_output(outputs: Any) -> t.Tensor:
    """
    Normalize a submodule's forward output into a Tensor.
    If it's a tuple, take the first tensor.
    If it's a dict/ModelOutput, prefer `last_hidden_state` / `hidden_states[-1]` / `logits`.
    """
    if isinstance(outputs, t.Tensor):
        return outputs
    if isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], t.Tensor):
        return outputs[0]
    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs and isinstance(outputs["last_hidden_state"], t.Tensor):
            return outputs["last_hidden_state"]
        if "hidden_states" in outputs and isinstance(outputs["hidden_states"], (list, tuple)) and len(outputs["hidden_states"]) > 0:
            if isinstance(outputs["hidden_states"][-1], t.Tensor):
                return outputs["hidden_states"][-1]
        if "logits" in outputs and isinstance(outputs["logits"], t.Tensor):
            return outputs["logits"]
        # Fallback: grab first tensor
        for v in outputs.values():
            if isinstance(v, t.Tensor):
                return v
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)) and len(outputs.hidden_states) > 0:
        if isinstance(outputs.hidden_states[-1], t.Tensor):
            return outputs.hidden_states[-1]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise TypeError(f"Cannot normalize submodule output of type: {type(outputs)}")


def _make_out_replacement_hook(dictionary, model_dtype, normalize_batch: bool, mode: str):
    """
    Build a forward hook to replace a submodule's OUTPUT.
    mode in {"recon", "zero"}.
    """
    assert mode in {"recon", "zero"}

    def hook_fn(_, __, output):
        x = _normalize_hook_output(output)  # (B, L, D)
        if mode == "zero":
            z = t.zeros_like(x)
            if isinstance(output, tuple):
                return (z,) + output[1:]
            return z

        # mode == "recon"
        # Optional batch normalization (match training-time normalization if requested)
        if normalize_batch:
            scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
            x_in = x * scale
        else:
            x_in = x

        x_hat = dictionary(x_in).to(dtype=model_dtype, device=x.device)

        if normalize_batch:
            x_hat = x_hat / scale

        if isinstance(output, tuple):
            return (x_hat,) + output[1:]
        return x_hat

    return hook_fn


def _make_in_replacement_pre_hook(dictionary, model_dtype, normalize_batch: bool, mode: str):
    """
    Build a forward PRE-hook to replace a submodule's INPUT.
    mode in {"recon", "zero"}.
    Note: inputs is a tuple; we replace the first positional tensor in it.
    """
    assert mode in {"recon", "zero"}

    def pre_hook_fn(_, inputs):
        if not isinstance(inputs, tuple) or len(inputs) == 0:
            return inputs
        x = inputs[0]
        if not isinstance(x, t.Tensor):
            return inputs

        if mode == "zero":
            z = t.zeros_like(x)
            return (z,) + inputs[1:]

        # mode == "recon"
        if normalize_batch:
            scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
            x_in = x * scale
        else:
            x_in = x

        x_hat = dictionary(x_in).to(dtype=model_dtype, device=x.device)

        if normalize_batch:
            x_hat = x_hat / scale

        return (x_hat,) + inputs[1:]

    return pre_hook_fn


# -----------------------------------------------------------------------------
# Loss-recovered (Transformers-only implementation)
# -----------------------------------------------------------------------------

def _loss_recovered_transformers(
    text_batch,                            # list[str] or token ids tensor
    model,                                 # HF Transformers model
    tokenizer,                             # HF tokenizer (for CE ignore_index)
    submodule,                             # nn.Module to intervene on
    dictionary,                            # SAE dictionary
    max_len: Optional[int] = None,
    normalize_batch: bool = False,
    io: str = "out",                       # 'in' or 'out'
    device: str = "cpu",
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Compute (loss_original, loss_reconstructed, loss_zero) using only Transformers + hooks.
    Robust to Dream/DLM masks.
    """
    # Tokenize if needed
    if isinstance(text_batch, t.Tensor):
        inputs = {"input_ids": text_batch.to(device)}
        # Build a mask of ones if none provided (no padding case)
        inputs["attention_mask"] = t.ones_like(inputs["input_ids"])
    else:
        inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            max_length=max_len if max_len is not None else tokenizer.model_max_length,
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # logits_original
    logits_original = _model_forward_logits_three_ways(model, inputs)

    # logits_reconstructed (with replacement hook)
    if io == "out":
        h = submodule.register_forward_hook(
            _make_out_replacement_hook(dictionary, model.dtype, normalize_batch, mode="recon")
        )
        try:
            logits_reconstructed = _model_forward_logits_three_ways(model, inputs)
        finally:
            h.remove()
    elif io == "in":
        h = submodule.register_forward_pre_hook(
            _make_in_replacement_pre_hook(dictionary, model.dtype, normalize_batch, mode="recon")
        )
        try:
            logits_reconstructed = _model_forward_logits_three_ways(model, inputs)
        finally:
            h.remove()
    else:
        raise NotImplementedError("io='in_and_out' is not supported in Transformers-only loss_recovered.")

    # logits_zero (with replacement hook)
    if io == "out":
        h = submodule.register_forward_hook(
            _make_out_replacement_hook(dictionary, model.dtype, normalize_batch, mode="zero")
        )
        try:
            logits_zero = _model_forward_logits_three_ways(model, inputs)
        finally:
            h.remove()
    elif io == "in":
        h = submodule.register_forward_pre_hook(
            _make_in_replacement_pre_hook(dictionary, model.dtype, normalize_batch, mode="zero")
        )
        try:
            logits_zero = _model_forward_logits_three_ways(model, inputs)
        finally:
            h.remove()
    else:
        raise NotImplementedError("io='in_and_out' is not supported in Transformers-only loss_recovered.")

    # Prepare labels for CE
    tokens = inputs["input_ids"]
    # Use pad_token_id for ignore_index if available
    ignore_index = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else -100
    ce = t.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _ce_loss_from_logits(logits: t.Tensor) -> t.Tensor:
        # logits: (B, T, V), we compute next-token loss
        B, T, V = logits.shape
        return ce(logits[:, :-1, :].reshape(-1, V), tokens[:, 1:].reshape(-1))

    loss_original = _ce_loss_from_logits(logits_original)
    loss_reconstructed = _ce_loss_from_logits(logits_reconstructed)
    loss_zero = _ce_loss_from_logits(logits_zero)

    return loss_original, loss_reconstructed, loss_zero


# -----------------------------------------------------------------------------
# Public evaluation API
# -----------------------------------------------------------------------------

@t.no_grad()
def evaluate(
    dictionary,                               # SAE dictionary (nn.Module-like)
    activations,                              # generator of activations; if ActivationBuffer, also compute loss recovered
    max_len: int = 128,                       # max context length for loss recovered
    batch_size: int = 128,                    # batch size for loss recovered (in texts)
    io: str = "out",                          # 'in' or 'out' (io='in_and_out' not supported for LR here)
    normalize_batch: bool = False,            # normalize batch before passing through dictionary
    tracer_args: dict = {'use_cache': False, 'output_attentions': False},  # kept for signature compatibility (unused)
    device: str = "cpu",
    n_batches: int = 1,
) -> Dict[str, float]:
    """
    Evaluate an SAE dictionary on a stream of activations and (optionally) compute loss recovered
    by intervening on the underlying Transformers model.

    Dream/DLM compatibility:
      - Forward passes are robust to attention_mask dtype via three-way fallback (bool / none / additive-float).
      - Works without nnsight; uses pure Transformers + forward hooks.

    Notes:
      - io='in_and_out' is not supported in the Transformers-only loss-recovered path.
      - For loss-recovered to run, `activations` must be an ActivationBuffer that exposes:
          - .model (Transformers model), .submodule (target nn.Module), .tokenizer
    """
    assert n_batches > 0, "n_batches must be > 0"
    out = defaultdict(float)
    active_features = t.zeros(dictionary.dict_size, dtype=t.float32, device=device)

    for _ in range(n_batches):
        try:
            x = next(activations).to(device)  # (N, D) or (B*L, D)
            if normalize_batch:
                x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim ** 0.5)
        except StopIteration:
            raise StopIteration(
                "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            )

        # Forward through dictionary to get reconstruction + sparse codes
        x_hat, f = dictionary(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()

        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32)
        assert features_BF.shape[-1] == dictionary.dict_size
        assert len(features_BF.shape) == 2

        active_features += features_BF.sum(dim=0)

        # cosine similarity between x and x_hat
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

        # variance explained
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = (1 - residual_variance / total_variance)

        # relative reconstruction bias (Equation 10 from https://arxiv.org/abs/2404.16014)
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2) ** 2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

        out["l2_loss"] += l2_loss.item()
        out["l1_loss"] += l1_loss.item()
        out["l0"] += l0.item()
        out["frac_variance_explained"] += float(frac_variance_explained)
        out["cossim"] += cossim.item()
        out["l2_ratio"] += l2_ratio.item()
        out["relative_reconstruction_bias"] += relative_reconstruction_bias.item()

        # ---------------- Loss recovered (only if we have a Transformers-backed ActivationBuffer) ----------------
        if isinstance(activations, ActivationBuffer):
            # Build a fresh batch of text for LR
            try:
                text_batch = activations.text_batch(batch_size=batch_size)
            except StopIteration:
                # No more text; skip loss recovered for this round
                continue

            try:
                loss_original, loss_reconstructed, loss_zero = _loss_recovered_transformers(
                    text_batch=text_batch,
                    model=activations.model,
                    tokenizer=activations.tokenizer,
                    submodule=activations.submodule,
                    dictionary=dictionary,
                    max_len=max_len,
                    normalize_batch=normalize_batch,
                    io=io,
                    device=device,
                )
                frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)

                out["loss_original"] += loss_original.item()
                out["loss_reconstructed"] += loss_reconstructed.item()
                out["loss_zero"] += loss_zero.item()
                out["frac_recovered"] += frac_recovered.item()
            except NotImplementedError as e:
                print(f"[LossRecovered] Skipped: {e}")
            except Exception as e:
                print(f"[LossRecovered] Failed: {type(e).__name__}: {e}")
                # Skip LR if something unexpected happens; continue with reconstruction metrics

    # Averages
    out = {key: value / n_batches for key, value in out.items()}
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()

    return out
