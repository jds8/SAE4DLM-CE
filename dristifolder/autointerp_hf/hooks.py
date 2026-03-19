# autointerp_hf/hooks.py
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import torch
from torch import nn, Tensor
from sae_lens import SAE  # assumes you still use sae_lens' SAE class


def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    """
    Resolve a dotted path like "model.layers.20" into the actual submodule.
    Supports integer indexing for list-like / ModuleList layers.
    """
    cur: nn.Module = root
    parts = path.split(".")

    for p in parts:
        if p.isdigit():
            cur = cur[int(p)]  # type: ignore
        else:
            cur = getattr(cur, p)
    return cur


@torch.no_grad()
def capture_module_activations(
    model: nn.Module,
    module_path: str,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
) -> Tensor:
    """
    Run the model forward and hook the specified submodule's output.

    We assume that the submodule output has shape [B, L, D] compatible with SAE.encode.

    NOTE:
    - This is intentionally generic. The user must set module_path so that
      the forward() output of that module is indeed the residual stream
      (or hidden state) that matches how the SAE was trained.

    Returns: captured_acts [B, L, D]
    """
    captured: Dict[str, Tensor] = {}

    def fwd_hook(mod, inp, out):
        # Many HF blocks return just hidden_states,
        # but sometimes they return tuples. We'll handle both.
        if isinstance(out, (tuple, list)):
            captured["acts"] = out[0].detach()
        else:
            captured["acts"] = out.detach()
        return out

    target_module = get_module_by_path(model, module_path)
    handle = target_module.register_forward_hook(fwd_hook)

    # Try common forward signatures.
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        # Some models complain if you don't pass use_cache, etc.
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    handle.remove()

    assert "acts" in captured, (
        "Forward hook did not capture activations. "
        "Check that module_path is correct and that the module outputs a tensor."
    )

    return captured["acts"]


@torch.no_grad()
def get_non_special_mask(
    input_ids: Tensor,
    tokenizer,
) -> Tensor:
    """
    Return a boolean mask [B, L], True where token is *not* a special token
    like BOS / EOS / PAD.

    We'll ignore these positions for sparsity / activation sampling.
    """
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    special_ids = []
    if getattr(tokenizer, "pad_token_id", None) is not None:
        special_ids.append(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        special_ids.append(tokenizer.eos_token_id)
    if getattr(tokenizer, "bos_token_id", None) is not None:
        special_ids.append(tokenizer.bos_token_id)

    for sid in special_ids:
        mask &= (input_ids != sid)

    return mask


@torch.no_grad()
def collect_sae_activations_hf(
    input_ids: Tensor,              # [N, L]
    attention_mask: Tensor,         # [N, L]
    model: nn.Module,
    sae: SAE,
    batch_size: int,
    hook_module_path: str,
    tokenizer,
    mask_special_tokens: bool = True,
    selected_latents: Optional[List[int]] = None,
    activation_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Tensor:
    """
    Compute SAE activations for all tokens in `input_ids`, batching
    and hooking the HuggingFace model. Do NOT save activations to disk.

    Returns:
        sae_acts_all: [N, L, d_sae_selected]
    """
    N, L = input_ids.shape
    device = next(model.parameters()).device
    sae_acts_batches: List[Tensor] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_input_ids = input_ids[start:end].to(device)
        batch_attn_mask = attention_mask[start:end].to(device)

        # Grab hidden states from the chosen layer
        resid_BLD = capture_module_activations(
            model,
            hook_module_path,
            batch_input_ids,
            batch_attn_mask,
        )  # [B, L, D]

        # SAE encode
        sae_act_BLF = sae.encode(resid_BLD)  # [B, L, d_sae]

        # Optionally restrict to subset of latents
        if selected_latents is not None:
            sae_act_BLF = sae_act_BLF[:, :, selected_latents]

        # Mask out BOS/PAD/EOS etc so we don't treat them as semantic activations
        if mask_special_tokens:
            ns_mask_BL = get_non_special_mask(batch_input_ids, tokenizer).to(
                sae_act_BLF.device
            )
        else:
            ns_mask_BL = torch.ones_like(batch_input_ids, dtype=torch.bool).to(
                sae_act_BLF.device
            )

        sae_act_BLF = sae_act_BLF * ns_mask_BL[:, :, None]

        # Cast down for memory if requested
        if activation_dtype is not None:
            sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)

        sae_acts_batches.append(sae_act_BLF)

    sae_acts_all = torch.cat(sae_acts_batches, dim=0)
    return sae_acts_all  # [N, L, d_selected]


@torch.no_grad()
def get_feature_activation_sparsity_hf(
    input_ids: Tensor,              # [N, L]
    attention_mask: Tensor,         # [N, L]
    model: nn.Module,
    sae: SAE,
    batch_size: int,
    hook_module_path: str,
    tokenizer,
) -> Tensor:
    """
    Estimate sparsity for each SAE feature:
    fraction of tokens (excluding specials) where that feature fires (> 0).

    Returns:
        sparsity_F: [d_sae] (float32)

    We will use this to filter out "dead" latents and only interpret "alive" ones.
    """
    device = next(model.parameters()).device
    n_features = sae.W_dec.shape[0]  # sae_lens convention: output dim of SAE

    running_sum_F = torch.zeros(
        n_features, dtype=torch.float32, device=device
    )
    total_non_special_tokens = 0

    N = input_ids.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_input_ids = input_ids[start:end].to(device)
        batch_attn_mask = attention_mask[start:end].to(device)

        resid_BLD = capture_module_activations(
            model,
            hook_module_path,
            batch_input_ids,
            batch_attn_mask,
        )  # [B, L, D]

        sae_act_BLF = sae.encode(resid_BLD)  # [B, L, F]

        # Binarize nonzero activations
        sae_act_mask_BLF = (sae_act_BLF > 0).to(torch.float32)

        # Only count real tokens (exclude BOS/PAD/EOS etc)
        ns_mask_BL = get_non_special_mask(batch_input_ids, tokenizer).to(
            sae_act_mask_BLF.device
        )
        sae_act_mask_BLF = sae_act_mask_BLF * ns_mask_BL[:, :, None]

        running_sum_F += torch.sum(sae_act_mask_BLF, dim=(0, 1))  # sum over B,L
        total_non_special_tokens += ns_mask_BL.sum().item()

    sparsity_F = running_sum_F / max(total_non_special_tokens, 1)
    return sparsity_F  # [F]
