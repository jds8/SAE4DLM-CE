"""
Perplexity-based fluency scoring with batch support.

- calc_ppl(text): single-text perplexity
- calc_ppl_batch(texts): batch perplexity (much faster)

This version supports an explicit device argument (e.g. "cuda:1"),
and caches the (tokenizer, model) per device to avoid reusing a model
loaded onto a different GPU.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from config import PPL_MODEL_NAME, PPL_MAX_LENGTH, PPL_BATCH_SIZE


@lru_cache()
def _get_ppl_model(device: str = "auto") -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Load (tokenizer, model) and move the model to the requested device.

    Notes:
    - The cache key includes `device`, so "cuda:0" and "cuda:1" will maintain
      separate cached model instances.
    - device="auto" picks "cuda" if available else "cpu" (PyTorch's default CUDA device).
    """
    tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)

    # Ensure pad token exists for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME)
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)

    # Recommended: set the current CUDA device explicitly when an index is provided,
    # to avoid any implicit allocations landing on cuda:0.
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)

    model.to(torch_device)

    return tokenizer, model, str(torch_device)


def _ppl_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-example perplexity from logits with masking.

    logits: [B, L, V]
    input_ids: [B, L]
    attention_mask: [B, L] with 1 for valid tokens, 0 for padding
    """
    # Shift for causal LM loss: predict token t+1 from token t.
    shift_logits = logits[:, :-1, :].contiguous()      # [B, L-1, V]
    shift_labels = input_ids[:, 1:].contiguous()       # [B, L-1]
    shift_mask = attention_mask[:, 1:].contiguous()    # [B, L-1]

    B, Lm1 = shift_labels.shape

    # Flatten for token-level cross entropy.
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, Lm1)

    # Mask out padding tokens.
    loss_flat = loss_flat * shift_mask

    # Average loss per example over valid tokens.
    denom = shift_mask.sum(dim=1).clamp(min=1)
    loss_per_ex = loss_flat.sum(dim=1) / denom

    ppl = torch.exp(loss_per_ex)
    return ppl


def calc_ppl_batch(
    texts: List[str],
    batch_size: int = PPL_BATCH_SIZE,
    device: Optional[str] = None,
) -> List[float]:
    """
    Compute perplexity for a list of texts using batching.

    Parameters
    ----------
    texts:
        List of texts to score.
    batch_size:
        Mini-batch size for scoring.
    device:
        Explicit device, e.g. "cuda:1", "cuda:0", or "cpu".
        If None, device="auto" is used.

    Returns
    -------
    List[float]
        Perplexity values, one per input text.
    """
    if not texts:
        return []

    tokenizer, model, resolved_device = _get_ppl_model(device or "auto")

    ppls: List[float] = []

    # Process in mini-batches to avoid OOM.
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=PPL_MAX_LENGTH,
        )

        input_ids = enc["input_ids"].to(resolved_device)
        attention_mask = enc["attention_mask"].to(resolved_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        ppl_tensor = _ppl_from_logits(logits, input_ids, attention_mask)
        ppls.extend([float(x) for x in ppl_tensor.detach().cpu().tolist()])

    return ppls


def calc_ppl(text: str, device: Optional[str] = None) -> float:
    """
    Single-text perplexity (wrapper over calc_ppl_batch).
    """
    return calc_ppl_batch([text], batch_size=1, device=device)[0]
