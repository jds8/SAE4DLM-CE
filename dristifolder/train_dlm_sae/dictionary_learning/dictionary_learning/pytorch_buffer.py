import os
import gc
import random
import contextlib
from typing import Any

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class EarlyStopException(Exception):
    """Custom exception to stop model forward pass early after capturing activations."""
    pass


# ------------------------------ Dream/DLM detection ------------------------------

def _is_dream_model(model) -> bool:
    """
    Heuristic to detect Dream/Diffusion LMs.
    Many Dream builds expose dream/diffusion identifiers or diffusion_generate API.
    """
    name = model.__class__.__name__.lower()
    return ("dream" in name) or ("diffusion" in name) or hasattr(model, "diffusion_generate")


# ------------------------------ Output normalization ------------------------------

def _normalize_hook_output(outputs: Any) -> t.Tensor:
    """
    Normalize various submodule outputs into a single Tensor.

    Handles:
      - Tensor
      - tuple with first item being Tensor
      - dictionaries / ModelOutput-like objects (prefers last_hidden_state / hidden_states[-1] / logits)
    """
    if isinstance(outputs, t.Tensor):
        return outputs

    if isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], t.Tensor):
        return outputs[0]

    if isinstance(outputs, dict):
        # Common keys in HF outputs / custom modules
        if "last_hidden_state" in outputs and isinstance(outputs["last_hidden_state"], t.Tensor):
            return outputs["last_hidden_state"]
        if "hidden_states" in outputs and isinstance(outputs["hidden_states"], (list, tuple)) and len(outputs["hidden_states"]) > 0:
            if isinstance(outputs["hidden_states"][-1], t.Tensor):
                return outputs["hidden_states"][-1]
        if "logits" in outputs and isinstance(outputs["logits"], t.Tensor):
            return outputs["logits"]
        # Fallback: pick the first tensor-looking value
        for v in outputs.values():
            if isinstance(v, t.Tensor):
                return v

    # HuggingFace ModelOutput-like (attributes)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)) and len(outputs.hidden_states) > 0:
        if isinstance(outputs.hidden_states[-1], t.Tensor):
            return outputs.hidden_states[-1]
    if hasattr(outputs, "logits"):
        return outputs.logits

    raise TypeError(f"Cannot normalize submodule output of type: {type(outputs)}")


# ------------------------------ SDPA mask helpers ------------------------------

def _make_additive_float_mask_from_1d(am: t.Tensor) -> t.Tensor:
    """
    Convert a (batch, seqlen) 0/1 mask into an additive float mask of shape (batch, 1, 1, seqlen),
    where keep positions -> 0.0 and pad positions -> -inf, which SDPA interprets as masked.
    """
    if am.dtype not in (t.float32, t.float16, t.bfloat16):
        am = am.float()
    # (B, T) -> (B, 1, 1, T)
    am4 = am[:, None, None, :]
    finfo = t.finfo(am4.dtype)
    minus_inf = finfo.min  # sufficiently negative to fully mask
    add_mask = t.where(am4 > 0, t.zeros_like(am4), t.full_like(am4, minus_inf))
    return add_mask


def _safe_forward_with_masks(model, inputs_BL: dict) -> None:
    """
    Safely call the model forward with attention_mask handling that is compatible
    with SDPA in Dream-style implementations.

    For Dream models:
      - try WITHOUT attention_mask
      - then try with additive float 4D mask.

    For non-Dream models:
      - try bool mask
      - then try without mask
      - then try additive float mask
    """
    is_dream = _is_dream_model(model)

    if is_dream:
        attempt_order = ("none", "additive")
    else:
        attempt_order = ("bool", "none", "additive")

    for attempt in attempt_order:
        try:
            with t.no_grad():
                inp = dict(inputs_BL)

                if attempt == "bool":
                    if "attention_mask" in inp:
                        inp["attention_mask"] = inp["attention_mask"].to(t.bool)
                elif attempt == "none":
                    inp.pop("attention_mask", None)
                elif attempt == "additive":
                    if "attention_mask" in inp:
                        add_mask = _make_additive_float_mask_from_1d(inp["attention_mask"])
                        inp["attention_mask"] = add_mask.to(device=add_mask.device)

                _ = model(**inp)
                return  # success (or EarlyStopException thrown inside hook)
        except EarlyStopException:
            return
        except Exception as e:
            # terse logs
            if attempt == "bool":
                print(f"[MaskTry-1 bool] Forward failed: {type(e).__name__}: {e}")
            elif attempt == "none":
                print(f"[MaskTry-2 none] Forward failed: {type(e).__name__}: {e}")
            else:
                print(f"[MaskTry-3 additive] Forward failed: {type(e).__name__}: {e}")

    raise RuntimeError("All attention_mask strategies failed during model forward.")


# ------------------------------ Forward masking helpers ------------------------------

def _build_random_mask(attn_mask: t.Tensor, p: float, keep_first: bool = True) -> t.Tensor:
    """
    Build a boolean mask (B,T) where True indicates the position will be replaced by MASK token.
    Only considers positions where attn_mask==1 (i.e., real tokens).
    """
    cand = attn_mask.bool().clone()
    B, T = cand.shape
    if keep_first and T > 0:
        cand[:, 0] = False  # keep BOS/start unmasked
    randu = t.rand_like(attn_mask, dtype=t.float32)
    m = (randu < p) & cand
    return m


# ------------------------------ Main activation collection ------------------------------

def collect_activations(
    model: AutoModelForCausalLM,
    submodule: t.nn.Module,
    inputs_BL: dict[str, t.Tensor],
    use_no_grad: bool = True,
) -> t.Tensor:
    """
    Register a forward hook on `submodule` to capture its activations, then short-circuit the
    remainder of the forward pass via EarlyStopException.
    """
    activations_BLD = None

    def hook_fn(_, __, outputs):
        nonlocal activations_BLD
        activations_BLD = _normalize_hook_output(outputs)
        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(hook_fn)
    context_manager = t.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        with context_manager:
            _safe_forward_with_masks(model, inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {type(e).__name__}: {e}")
        raise
    finally:
        handle.remove()

    if activations_BLD is None:
        raise RuntimeError("Failed to collect activations. The hook might not have run correctly.")
    return activations_BLD


# ------------------------------ Activation reservoir buffer ------------------------------

class ActivationBuffer:
    """
    Buffer of token-level activations. Now supports DLM forward noising + position selection.
      - DLM_MASK_POLICY: 'unmask' | 'mask' | 'all' | 'clean'
          * For Dream models (detected automatically), default 'unmask'.
          * For non-Dream models, default 'clean'.
      - DLM_T_MIN / DLM_T_MAX: float range to sample p ~ U[t_min, t_max] as mask probability per batch.
      - DLM_DEBUG_PRINT=1: print per-batch stats
      - DLM_DEBUG_TOKENS=1: also print a compact token visualization for the first item in batch
      - DLM_DEBUG_EVERY=N: print every N batches (default 1)
    """

    def __init__(
        self,
        data=None,  # generator which yields text data (alias: `generator`)
        model: AutoModelForCausalLM = None,
        submodule: t.nn.Module = None,
        d_submodule=None,
        io="out",
        n_ctxs=3e4,
        ctx_len=128,
        refresh_batch_size=512,
        out_batch_size=8192,
        device: str | None = "cuda:0",
        remove_bos: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        # --- Backward-compatibility alias for `generator=` ---
        if data is None and "generator" in kwargs:
            data = kwargs.pop("generator")
        if data is None:
            raise ValueError("`data` (or alias `generator`) must be provided.")
        if kwargs:
            # Silently ignore unknown kwargs to be lenient
            pass

        if io not in ["in", "out"]:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == "in":
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except Exception:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")

        # --- Resolve device ---
        resolved_device = None
        if device is not None:
            try:
                cand = t.device(device)
                if cand.type == "cuda":
                    if t.cuda.is_available():
                        try:
                            _ = t.empty(0, device=cand)
                            resolved_device = cand
                        except Exception:
                            resolved_device = None
                    else:
                        resolved_device = None
                else:
                    resolved_device = cand
            except Exception:
                resolved_device = None

        if resolved_device is None:
            try:
                resolved_device = next(submodule.parameters()).device
            except Exception:
                try:
                    resolved_device = next(model.parameters()).device
                except Exception:
                    resolved_device = t.device("cpu")

        self.device = resolved_device

        # Allocate buffers on the resolved device
        self.activations = t.empty(0, d_submodule, device=self.device, dtype=model.dtype)
        self.read = t.zeros(0, device=self.device).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.remove_bos = remove_bos
        self.add_special_tokens = add_special_tokens

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model.name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---------------- DLM noising policy (env-driven; default by model type) ----------------
        if _is_dream_model(self.model):
            default_policy = "unmask"
        else:
            default_policy = "clean"

        self.dlm_mask_policy = os.getenv("DLM_MASK_POLICY", default_policy).strip().lower()
        if self.dlm_mask_policy not in ("unmask", "mask", "all", "clean"):
            print(f"[Warn] Unknown DLM_MASK_POLICY='{self.dlm_mask_policy}', fallback to '{default_policy}'.")
            self.dlm_mask_policy = default_policy

        # per-batch p ~ U[t_min, t_max]
        try:
            self.dlm_t_min = float(os.getenv("DLM_T_MIN", "0.05"))
            self.dlm_t_max = float(os.getenv("DLM_T_MAX", "0.50"))
            if self.dlm_t_min < 0.0 or self.dlm_t_max < 0.0 or self.dlm_t_min > self.dlm_t_max:
                raise ValueError
        except Exception:
            self.dlm_t_min, self.dlm_t_max = 0.05, 0.50

        # choose a MASK token id (fallback to unk/eos)
        self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else self.tokenizer.eos_token_id

        # --------------- Debug printing controls ---------------
        self.dlm_debug = os.getenv("DLM_DEBUG_PRINT", "0") == "1"
        self.dlm_debug_tokens = os.getenv("DLM_DEBUG_TOKENS", "0") == "1"
        try:
            self.dlm_debug_every = max(1, int(os.getenv("DLM_DEBUG_EVERY", "1")))
        except Exception:
            self.dlm_debug_every = 1
        self._debug_batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a random batch of unread token-level activations of shape [out_batch_size, d_submodule].
        Automatically refresh the reservoir if remaining unread entries < half capacity.
        """
        with t.no_grad():
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            unreads = (~self.read).nonzero(as_tuple=False).squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    # ------------------------- Data Ingestion Utilities -------------------------

    def text_batch(self, batch_size=None):
        """Return a list of raw text strings from the underlying generator."""
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [next(self.data) for _ in range(batch_size)]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def tokenized_batch(self, batch_size=None):
        """
        Tokenize a batch into model inputs. Keep padding=True; SDPA dtype/shape fixes are handled during forward.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.ctx_len,
            padding=True,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        ).to(self.device)

    # ------------------------------ Reservoir Refresh ---------------------------

    def _apply_forward_noising(self, inputs: dict, p: float):
        """
        Build masked inputs by replacing positions ~ Bernoulli(p) (where attention_mask==1) with MASK token.
        Returns: masked_inputs(dict), masked_positions(bool tensor of shape (B,T))
        """
        input_ids = inputs["input_ids"].clone()
        attn = inputs["attention_mask"].clone()
        m = _build_random_mask(attn, p=p, keep_first=True)
        masked_ids = input_ids.clone()
        masked_ids[m] = self.mask_token_id
        masked_inputs = {"input_ids": masked_ids, "attention_mask": attn}
        return masked_inputs, m

    def refresh(self):
        """
        Refill the activation reservoir with either:
          - CLEAN activations (LLM-style), or
          - DLM forward-noised activations with position selection (mask/unmask/all).
        """
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()

        # Keep only unread entries
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.model.dtype,
        )
        if current_idx > 0:
            new_activations[: current_idx] = self.activations
        self.activations = new_activations

        # keep sampling batches until reservoir is full
        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                inputs = self.tokenized_batch()

                # build attn mask for selection
                attn_mask = inputs.get("attention_mask", None)
                if attn_mask is None:
                    pad_id = self.tokenizer.pad_token_id
                    attn_mask = (inputs["input_ids"] != pad_id).long()
                attn_mask = attn_mask.to(self.device)

                # choose whether to apply DLM forward noising
                using_clean = (self.dlm_mask_policy == "clean")
                if using_clean:
                    forward_inputs = inputs
                    masked_positions = None
                    p = 0.0
                else:
                    # sample p ~ U[t_min, t_max] per batch
                    p = random.uniform(self.dlm_t_min, self.dlm_t_max)
                    forward_inputs, masked_positions = self._apply_forward_noising(inputs, p=p)

                hidden_states = collect_activations(self.model, self.submodule, forward_inputs)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                # optional BOS removal for all policies (applies to both H and masks)
                if self.remove_bos:
                    hidden_states = hidden_states[:, 1:, :]
                    attn_mask = attn_mask[:, 1:]
                    if masked_positions is not None:
                        masked_positions = masked_positions[:, 1:]

                # build position selection mask
                if using_clean or masked_positions is None:
                    # CLEAN path: select all true tokens (non-pad)
                    select_pos = attn_mask.bool()
                else:
                    if self.dlm_mask_policy == "mask":
                        select_pos = masked_positions
                    elif self.dlm_mask_policy == "unmask":
                        select_pos = (~masked_positions) & attn_mask.bool()
                    else:  # 'all'
                        select_pos = attn_mask.bool()

                # -------- Debug print: show what exactly we are selecting --------
                self._debug_batch_idx += 1
                if self.dlm_debug and (self._debug_batch_idx % self.dlm_debug_every == 0):
                    B, T = attn_mask.shape
                    valid = int(attn_mask.sum().item())
                    masked_cnt = int(masked_positions.sum().item()) if masked_positions is not None else 0
                    selected_cnt = int(select_pos.sum().item())
                    ratio = (selected_cnt / max(valid, 1)) if valid > 0 else 0.0
                    print(
                        f"[DLM] policy={self.dlm_mask_policy:6s} p={p:.3f} "
                        f"BxT={B}x{T} valid={valid} masked={masked_cnt} "
                        f"selected={selected_cnt} ({ratio:.2%}) "
                        f"{'(BOS-removed)' if self.remove_bos else ''}"
                    )

                    if self.dlm_debug_tokens and B >= 1:
                        # First sample visualization (truncate for readability)
                        show_T = min(T, 64)
                        ids0 = forward_inputs["input_ids"][0][:show_T].tolist()
                        toks0 = self.tokenizer.convert_ids_to_tokens(ids0, skip_special_tokens=False)
                        attn0 = attn_mask[0][:show_T].tolist()
                        if masked_positions is not None:
                            m0 = masked_positions[0][:show_T].tolist()
                        else:
                            m0 = [False] * show_T
                        sel0 = select_pos[0][:show_T].tolist()

                        def _sym(m, v):
                            # M = masked position; u = unmasked valid token; _ = padding
                            if v:
                                return "M" if m else "u"
                            else:
                                return "_"

                        msks = " ".join(_sym(m, v) for m, v in zip(m0, attn0))
                        sels = " ".join("*" if s else "." for s in sel0)
                        toks = " ".join(toks0)
                        print("TOK:", toks)
                        print("MSK:", msks)
                        print("SEL:", sels)
                        if T > show_T:
                            print(f"... (truncated to {show_T} tokens)")

                # -----------------------------------------------------------------

                # flatten selected positions
                if hidden_states.ndim != 3:
                    raise RuntimeError(f"Expected (B,T,D) activations, got {tuple(hidden_states.shape)}")
                flat_h = hidden_states[select_pos]

                # If no positions selected (e.g., p too small & BOS removed), skip this batch
                if flat_h.numel() == 0:
                    continue

                # write into reservoir
                remaining_space = self.activation_buffer_size - current_idx
                if remaining_space <= 0:
                    break
                flat_h = flat_h[:remaining_space]

                self.activations[current_idx : current_idx + len(flat_h)] = flat_h.to(self.device)
                current_idx += len(flat_h)

        # Reset read flags
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    # ------------------------------- Introspection ------------------------------

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "io": self.io,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": str(self.device),  # JSON-friendly
            # expose DLM settings for reproducibility
            "dlm_mask_policy": self.dlm_mask_policy,
            "dlm_t_min": getattr(self, "dlm_t_min", None),
            "dlm_t_max": getattr(self, "dlm_t_max", None),
        }
