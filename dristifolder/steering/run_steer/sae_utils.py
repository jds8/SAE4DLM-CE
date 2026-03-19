# sae_utils.py
# -*- coding: utf-8 -*-
import torch
from torch import nn
from contextlib import contextmanager
from accelerate.hooks import ModelHook
from typing import Any


@contextmanager
def _disable_hooks(sae: nn.Module):
    """
    Placeholder context manager for parity with environments where SAE might
    have its own hooks; no-op here.
    """
    yield


def _resolve_layers_container(model: nn.Module) -> Any:
    """
    Try multiple common locations to find the transformer blocks (layers) container.
    Works for many decoder-only architectures (Qwen/Gemma/GPT-like).
    """
    # Most HF decoder-only models expose .model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Some expose .layers directly
    if hasattr(model, "layers"):
        return model.layers
    # GPT-J style: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not locate transformer layers. Tried model.model.layers / model.layers / model.transformer.h"
    )


# # last token
# class AmlifySAEHook(ModelHook):
#     """
#     Forward hook that amplifies selected SAE features on the last token and
#     writes the reconstructed delta back into the layer output.

#     Robust to both [B, D] and [B, T, D] layer outputs:
#     - If output is [B, D], it is temporarily expanded to [B, 1, D] and then
#       squeezed back to [B, D] before returning.
#     - If the original module returns a tuple, the first element is assumed to be
#       the hidden states tensor and the rest are passed through unchanged.

#     math:
#       A       = SAE.encode(H)
#       A'      = A; A'[:, -1, s] += amp_factor * max_f A[:, -1, f] for s in features
#       H'      = SAE.decode(A')
#       H_tilde = H + (H' - SAE.decode(A))      # residual error compensation
#     """
#     def __init__(self, layer: int, sae: nn.Module, features, amp_factor: float, device: str) -> None:
#         super().__init__()
#         self.amp_factor = float(amp_factor)
#         self.sae = sae
#         self.device = torch.device(device)
#         self.layer = int(layer)
#         self.features = [int(f) for f in features]

#     def __call__(self, module, args, output):
#         # Unpack output: Tensor or tuple where first element is hidden states.
#         if isinstance(output, (tuple, list)):
#             hidden = output[0]
#             others = list(output[1:])
#         else:
#             hidden = output
#             others = None

#         if not torch.is_tensor(hidden):
#             raise RuntimeError(f"Expected tensor from layer, got {type(hidden)}")

#         # Save original shape/type/device to restore later
#         orig_device = hidden.device
#         orig_dtype = hidden.dtype
#         squeezed = False

#         # Normalize to [B, T, D]
#         if hidden.ndim == 2:  # [B, D] -> [B, 1, D]
#             hidden = hidden.unsqueeze(1)
#             squeezed = True
#         elif hidden.ndim != 3:
#             raise RuntimeError(f"Expected 2D/3D tensor, got {hidden.ndim}D")

#         # Move to SAE device if needed
#         if hidden.device != self.device:
#             hidden = hidden.to(self.device)

#         # SAE encode (shape: [B, T, F])
#         feature_acts = self.sae.encode(hidden)

#         # Clean reconstruction (no amplification) for residual compensation
#         with torch.no_grad():
#             with _disable_hooks(self.sae):
#                 feature_acts_clean = self.sae.encode(hidden)
#                 x_reconstruct_clean = self.sae.decode(feature_acts_clean)

#         # Amplify selected features on the last token
#         last_feats = feature_acts[:, -1, :]  # [B, F]
#         F = last_feats.shape[-1]
#         idx = torch.as_tensor(self.features, device=last_feats.device, dtype=torch.long)
#         idx = idx[(idx >= 0) & (idx < F)]
#         if idx.numel() > 0:
#             max_val = last_feats.amax(dim=-1, keepdim=True)  # [B, 1]
#             src = max_val.expand(last_feats.size(0), idx.numel()) * self.amp_factor  # [B, K]
#             last_feats = last_feats.scatter_add(
#                 dim=-1,
#                 index=idx.unsqueeze(0).expand(last_feats.size(0), -1),
#                 src=src
#             )
#             feature_acts[:, -1, :] = last_feats

#         # Decode amplified activations to get updated hidden states
#         sae_out = self.sae.decode(feature_acts)

#         # Residual error compensation; ensure dtype alignment
#         sae_out = sae_out + (hidden.to(torch.float32) - x_reconstruct_clean.to(torch.float32))
#         sae_out = sae_out.to(orig_dtype)

#         # Restore original rank and device
#         if squeezed:
#             sae_out = sae_out.squeeze(1)
#         if sae_out.device != orig_device:
#             sae_out = sae_out.to(orig_device)

#         # Repack return to match original output type
#         if others is not None:
#             return tuple([sae_out] + others)
#         else:
#             return sae_out

# all tokens

class AmlifySAEHook(ModelHook):
    """
    Forward hook that amplifies selected SAE features on **all tokens** and
    writes the reconstructed delta back into the layer output.

    Robust to both [B, D] and [B, T, D] layer outputs:
    - If output is [B, D], it is temporarily expanded to [B, 1, D] and then
      squeezed back to [B, D] before returning.
    - If the original module returns a tuple, the first element is assumed to be
      the hidden states tensor and the rest are passed through unchanged.

    math (per forward pass):
      H        = layer hidden states, shape [B, T, D]
      A        = SAE.encode(H)           # [B, T, F]
      For each token position t and sample b:
        let m_{b,t} = max_f A[b, t, f]   # max activation across features
        for each selected feature s in `features`:
          A'[b, t, s] = A[b, t, s] + amp_factor * m_{b,t}

      H'       = SAE.decode(A')          # [B, T, D]
      H_clean  = SAE.decode(A_clean)     # reconstruction from unmodified A
      H_tilde  = H' + (H - H_clean)      # residual error compensation
    """

    def __init__(self, layer: int, sae: nn.Module, features, amp_factor: float, device: str) -> None:
        super().__init__()
        self.amp_factor = float(amp_factor)
        self.sae = sae
        self.device = torch.device(device)
        self.layer = int(layer)
        # store selected feature indices as a Python list of ints
        self.features = [int(f) for f in features]

    def __call__(self, module, args, output):
        # Unpack output: Tensor or tuple where first element is hidden states.
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            others = list(output[1:])
        else:
            hidden = output
            others = None

        if not torch.is_tensor(hidden):
            raise RuntimeError(f"Expected tensor from layer, got {type(hidden)}")

        # Save original shape/type/device to restore later
        orig_device = hidden.device
        orig_dtype = hidden.dtype
        squeezed = False

        # Normalize to [B, T, D]
        if hidden.ndim == 2:  # [B, D] -> [B, 1, D]
            hidden = hidden.unsqueeze(1)
            squeezed = True
        elif hidden.ndim != 3:
            raise RuntimeError(f"Expected 2D/3D tensor, got {hidden.ndim}D")

        # Move to SAE device if needed
        if hidden.device != self.device:
            hidden = hidden.to(self.device)

        # SAE encode (shape: [B, T, F])
        feature_acts = self.sae.encode(hidden)

        # Clean reconstruction (no amplification) for residual compensation
        with torch.no_grad():
            with _disable_hooks(self.sae):
                feature_acts_clean = self.sae.encode(hidden)
                x_reconstruct_clean = self.sae.decode(feature_acts_clean)

        # Amplify selected features on **all tokens**
        B, T, F = feature_acts.shape
        idx = torch.as_tensor(self.features, device=feature_acts.device, dtype=torch.long)
        # keep only indices within [0, F)
        idx = idx[(idx >= 0) & (idx < F)]

        if idx.numel() > 0:
            # max activation across features for each (batch, token)
            # shape: [B, T, 1]
            max_val = feature_acts.amax(dim=-1, keepdim=True)

            # src to add: amp_factor * max_val for each selected feature index
            # shape: [B, T, K]
            src = max_val.expand(B, T, idx.numel()) * self.amp_factor

            # expand feature indices to shape [B, T, K] for scatter_add
            idx_expanded = idx.view(1, 1, -1).expand(B, T, -1)

            # add src to the selected feature dimensions across **all tokens**
            feature_acts = feature_acts.scatter_add(
                dim=-1,
                index=idx_expanded,
                src=src,
            )

        # Decode amplified activations to get updated hidden states
        sae_out = self.sae.decode(feature_acts)

        # Residual error compensation; ensure dtype alignment
        sae_out = sae_out + (hidden.to(torch.float32) - x_reconstruct_clean.to(torch.float32))
        sae_out = sae_out.to(orig_dtype)

        # Restore original rank and device
        if squeezed:
            sae_out = sae_out.squeeze(1)
        if sae_out.device != orig_device:
            sae_out = sae_out.to(orig_device)

        # Repack return to match original output type
        if others is not None:
            return tuple([sae_out] + others)
        else:
            return sae_out

def init_hook(pipeline, sae: nn.Module, layer: int, feature: int, device: str, args):
    """
    Register the forward hook on the specified decoder layer.
    This resolves the correct container for 'layers' across different model families.
    """
    layers_container = _resolve_layers_container(pipeline.model)
    if layer < 0 or layer >= len(layers_container):
        raise IndexError(f"Layer index {layer} out of range (0..{len(layers_container)-1})")

    sae_hook = AmlifySAEHook(layer, sae, [feature], args.amp_factor, device)
    handle = layers_container[layer].register_forward_hook(sae_hook, always_call=True)
    return handle
