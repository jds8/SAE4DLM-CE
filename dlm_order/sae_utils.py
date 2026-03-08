# sae_utils.py
# -*- coding: utf-8 -*-
"""
Utilities for:
- Loading local Dictionary-Learning SAEs (DL-SAE) trained on Dream DLM residual streams.
- Resolving the transformer blocks container.
- Capturing per-layer hidden states at selected token positions via forward hooks.
- Encoding hidden states with SAE and extracting top-k / top-1 features and activations.

This file is intentionally self-contained to minimize dependencies across the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# 0) Model layers resolver
# =============================================================================

def resolve_layers_container(model: nn.Module) -> Any:
    """
    Try multiple common locations to find the transformer blocks container.
    Works for many decoder-only architectures (Qwen/Gemma/GPT-like) and often Dream.

    Returns something indexable: layers_container[layer_idx].
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


# =============================================================================
# 1) Local DL-SAE loader (minimal and robust)
# =============================================================================

def _flatten_state_dict(sd: dict) -> dict:
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def _extract_bias_vector(state_dict: dict, names: List[str]) -> Optional[torch.Tensor]:
    # First try exact names
    for n in names:
        if n in state_dict and isinstance(state_dict[n], torch.Tensor):
            v = state_dict[n]
            if v.ndim == 1:
                return v
    # Then fuzzy match
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 1:
            continue
        kl = k.lower()
        if any(n in kl for n in names):
            return v
    return None


def _ensure_FxD(mat: torch.Tensor) -> torch.Tensor:
    """
    Ensure matrix is [F, D]. If it looks like [D, F], transpose.
    """
    if mat.ndim != 2:
        raise ValueError("Decoder/Encoder matrix must be 2D.")
    f, d = mat.shape
    if f < d:
        mat = mat.t()
    return mat


def _pick_best_decoder_key(state_dict: dict) -> str:
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue
        name = k.lower()
        score = 0
        if "decoder" in name:
            score += 100
        if "w_dec" in name or "decoder.weight" in name or ("dec" in name and "weight" in name):
            score += 50
        if "weight" in name:
            score += 10
        candidates.append((score, v.numel(), k))
    if not candidates:
        raise ValueError("No 2D decoder-like tensor found in state dict.")
    candidates.sort(reverse=True)
    return candidates[0][2]


def _pick_best_encoder_key(state_dict: dict) -> Optional[str]:
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue
        name = k.lower()
        score = 0
        if "encoder" in name:
            score += 100
        if "w_enc" in name or "encoder.weight" in name or ("enc" in name and "weight" in name):
            score += 50
        if "weight" in name:
            score += 10
        if "decoder" in name or "w_dec" in name:
            score -= 200
        candidates.append((score, v.numel(), k))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if candidates[0][0] <= 0:
        return None
    return candidates[0][2]


def _extract_decoder_FxD(state_dict: dict, decoder_key: str) -> torch.Tensor:
    W = state_dict[decoder_key]
    if not isinstance(W, torch.Tensor) or W.ndim != 2:
        raise ValueError(f"Decoder weight at {decoder_key} must be a 2D tensor.")
    W = W.to(torch.float32).contiguous()
    return _ensure_FxD(W)


def _extract_encoder_FxD(state_dict: dict, encoder_key: str) -> torch.Tensor:
    W = state_dict[encoder_key]
    if not isinstance(W, torch.Tensor) or W.ndim != 2:
        raise ValueError(f"Encoder weight at {encoder_key} must be a 2D tensor.")
    W = W.to(torch.float32).contiguous()
    return _ensure_FxD(W)


def _get_k_from_config(cfg: dict) -> Optional[int]:
    tr = cfg.get("trainer", {}) if isinstance(cfg.get("trainer"), dict) else {}
    k = tr.get("k", None)
    try:
        return int(k) if k is not None else None
    except Exception:
        return None


class LocalSAE(nn.Module):
    """
    A minimal SAE wrapper for common DL-SAE variants:
    - Standard (ReLU)
    - TopK / BatchTopK
    - JumpReLU
    - Gated

    encode(x): returns activations a [..., F]
    decode(a): returns reconstruction x_hat [..., D]

    NOTE:
    - W_dec must be [F, D]
    - W_enc must be [F, D] (optional; if missing, fallback to W_dec)
    """
    def __init__(
        self,
        W_dec_FD: torch.Tensor,
        W_enc_FD: Optional[torch.Tensor],
        b_dec_D: Optional[torch.Tensor],
        b_enc_F: Optional[torch.Tensor],
        trainer_class_name: str,
        threshold_scalar: Optional[float],
        threshold_vector_F: Optional[torch.Tensor],
        k_topk: Optional[int],
        gate_bias_F: Optional[torch.Tensor],
        r_mag_F: Optional[torch.Tensor],
        mag_bias_F: Optional[torch.Tensor],
        device: str = "cpu",
    ):
        super().__init__()
        self.trainer_class_name = (trainer_class_name or "").lower()

        self.register_buffer("W_dec", W_dec_FD.to(torch.float32).to(device))

        if W_enc_FD is not None:
            self.register_buffer("W_enc", W_enc_FD.to(torch.float32).to(device))
        if b_dec_D is not None:
            self.register_buffer("b_dec", b_dec_D.to(torch.float32).to(device))
        if b_enc_F is not None:
            self.register_buffer("b_enc", b_enc_F.to(torch.float32).to(device))
        if threshold_vector_F is not None:
            self.register_buffer("threshold_vector", threshold_vector_F.to(torch.float32).to(device))
        if gate_bias_F is not None:
            self.register_buffer("gate_bias", gate_bias_F.to(torch.float32).to(device))
        if r_mag_F is not None:
            self.register_buffer("r_mag", r_mag_F.to(torch.float32).to(device))
        if mag_bias_F is not None:
            self.register_buffer("mag_bias", mag_bias_F.to(torch.float32).to(device))

        self.threshold_scalar = threshold_scalar
        self.k_topk = k_topk

    def _preact(self, x: torch.Tensor) -> torch.Tensor:
        W_enc = getattr(self, "W_enc", None)
        if W_enc is None:
            W_enc = self.W_dec
        x = x.to(W_enc.dtype)
        pre = torch.einsum("...d,fd->...f", x, W_enc)

        if self.trainer_class_name.startswith("gated"):
            gate_bias = getattr(self, "gate_bias", None)
            if gate_bias is not None:
                pre = pre + gate_bias
        else:
            b_enc = getattr(self, "b_enc", None)
            if b_enc is not None:
                pre = pre + b_enc
        return pre

    def _apply_activation(self, pre: torch.Tensor) -> torch.Tensor:
        name = self.trainer_class_name

        if "jumprelu" in name:
            thr_vec = getattr(self, "threshold_vector", None)
            thr = thr_vec if thr_vec is not None else (self.threshold_scalar if self.threshold_scalar is not None else 0.0)
            return torch.relu(pre - thr)

        if ("topk" in name) or ("batchtopk" in name):
            thr_vec = getattr(self, "threshold_vector", None)
            out = pre
            if thr_vec is not None:
                out = out - thr_vec
            elif self.threshold_scalar is not None:
                out = out - self.threshold_scalar
            out = torch.relu(out)

            # If k_topk exists, keep only top-k entries
            if self.k_topk is not None and 0 < self.k_topk < out.shape[-1]:
                topk_vals, topk_idx = torch.topk(out, k=self.k_topk, dim=-1)
                zeros = torch.zeros_like(out)
                out = zeros.scatter(-1, topk_idx, topk_vals)
            return out

        if "gated" in name:
            base = torch.relu(pre)
            r_mag = getattr(self, "r_mag", None)
            mag_bias = getattr(self, "mag_bias", None)
            if (r_mag is not None) or (mag_bias is not None):
                r = r_mag if r_mag is not None else 1.0
                m = mag_bias if mag_bias is not None else 0.0
                gate = torch.sigmoid(r * pre + m)
                base = base * gate
            return base

        return torch.relu(pre)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._preact(x)
        a = self._apply_activation(pre)
        return a

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        z = a.to(self.W_dec.dtype)
        x_hat = torch.einsum("...f,fd->...d", z, self.W_dec)
        b_dec = getattr(self, "b_dec", None)
        if b_dec is not None:
            x_hat = x_hat + b_dec
        return x_hat


def _load_dictionary_learning_sae_from_folder(trainer_folder: str, device: str) -> LocalSAE:
    """
    Read config.json + ae.pt under a trainer folder and build a LocalSAE.
    """
    cfg_path = os.path.join(trainer_folder, "config.json")
    wt_path = os.path.join(trainer_folder, "ae.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(wt_path):
        raise FileNotFoundError(f"Missing config.json or ae.pt under {trainer_folder}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    trainer_class = (cfg.get("trainer", {}) or {}).get("trainer_class", "")

    ckpt = torch.load(wt_path, map_location="cpu")
    sd = _flatten_state_dict(ckpt)

    dec_key = _pick_best_decoder_key(sd)
    W_dec_FD = _extract_decoder_FxD(sd, dec_key)

    b_dec = _extract_bias_vector(sd, names=["decoder.bias", "decoder_bias", "b_dec", "dec_bias"])

    enc_key = _pick_best_encoder_key(sd)
    W_enc_FD = _extract_encoder_FxD(sd, enc_key) if enc_key is not None else None

    b_enc = _extract_bias_vector(sd, names=["encoder.bias", "b_enc", "enc_bias", "bias_enc"])
    gate_bias = _extract_bias_vector(sd, names=["gate_bias"])

    threshold_vec = _extract_bias_vector(sd, names=["threshold", "thresholds"])

    r_mag = _extract_bias_vector(sd, names=["r_mag"])
    mag_bias = _extract_bias_vector(sd, names=["mag_bias"])

    k_topk = _get_k_from_config(cfg)

    sae = LocalSAE(
        W_dec_FD=W_dec_FD,
        W_enc_FD=W_enc_FD,
        b_dec_D=b_dec,
        b_enc_F=b_enc,
        trainer_class_name=trainer_class,
        threshold_scalar=None,
        threshold_vector_F=threshold_vec,
        k_topk=k_topk,
        gate_bias_F=gate_bias,
        r_mag_F=r_mag,
        mag_bias_F=mag_bias,
        device=device,
    )
    return sae


def _trainer_dirs_for_layer(base_dir: str, layer: int) -> List[str]:
    """
    List trainer_* subdirectories under resid_post_layer_{layer}.
    """
    layer_dir = os.path.join(base_dir, f"resid_post_layer_{layer}")
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")
    subdirs = []
    for name in sorted(os.listdir(layer_dir)):
        p = os.path.join(layer_dir, name)
        if os.path.isdir(p) and re.match(r"^trainer_\d+$", name):
            subdirs.append(p)
    if not subdirs:
        raise FileNotFoundError(f"No trainer_* subdirectories under: {layer_dir}")
    return subdirs


def _pick_trainer_dir(base_dir: str, layer: int, trainer_name: Optional[str], k_topk: Optional[int]) -> str:
    """
    Select the appropriate trainer directory for a given layer.
    Priority:
      1) Explicit trainer_name if exists.
      2) First trainer whose config.json has trainer.k == k_topk.
      3) Fallback to trainer_0 if exists, else the first trainer_*.
    """
    trainer_dirs = _trainer_dirs_for_layer(base_dir, layer)

    if trainer_name is not None:
        cand = os.path.join(base_dir, f"resid_post_layer_{layer}", trainer_name)
        if os.path.isdir(cand):
            return cand
        raise FileNotFoundError(f"Requested trainer '{trainer_name}' not found for layer {layer}: {cand}")

    if k_topk is not None:
        for d in trainer_dirs:
            cfg_path = os.path.join(d, "config.json")
            if not os.path.exists(cfg_path):
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if _get_k_from_config(cfg) == int(k_topk):
                    return d
            except Exception:
                pass

    t0 = os.path.join(base_dir, f"resid_post_layer_{layer}", "trainer_0")
    if os.path.isdir(t0):
        return t0
    return trainer_dirs[0]


def load_saes(
    sae_root_dir: str,
    layers: List[int],
    device: str,
    trainer_name: Optional[str] = None,
    k_topk: Optional[int] = None,
) -> Dict[int, nn.Module]:
    """
    Load (and return) SAEs for requested layers.

    The expected folder structure:
      sae_root_dir/
        resid_post_layer_{L}/
          trainer_0/ (or trainer_{i}/)
            ae.pt
            config.json
    """
    saes: Dict[int, nn.Module] = {}
    for layer in layers:
        trainer_dir = _pick_trainer_dir(sae_root_dir, layer, trainer_name=trainer_name, k_topk=k_topk)
        sae = _load_dictionary_learning_sae_from_folder(trainer_dir, device=device)
        sae.eval()
        saes[layer] = sae
    return saes


# =============================================================================
# 2) Capture per-layer hidden states at selected positions
# =============================================================================

@dataclass
class CaptureResult:
    """
    Captured hidden states for one forward pass.
    - positions: the positions used for capture
    - by_layer: layer_idx -> hidden tensor [P, D] on the same device as the model outputs
    """
    positions: List[int]
    by_layer: Dict[int, torch.Tensor]


class _LayerCaptureHook:
    """
    A forward hook that captures hidden states at selected positions from a given block output.
    """
    def __init__(self, layer_idx: int):
        self.layer_idx = int(layer_idx)
        self.positions: List[int] = []
        self.hidden_PxD: Optional[torch.Tensor] = None

    def set_positions(self, positions: List[int]) -> None:
        self.positions = [int(p) for p in noting_out_of_range(positions)]

    def clear(self) -> None:
        self.hidden_PxD = None

    def __call__(self, module, args, output):
        # Unpack output: Tensor or tuple/list where first element is hidden states.
        if isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            hidden = output

        if not torch.is_tensor(hidden):
            return

        # Normalize to [B, T, D]
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)  # [B,1,D]
        if hidden.ndim != 3:
            return

        if hidden.shape[0] < 1:
            return

        if not self.positions:
            return

        # Clamp positions safely
        T = hidden.shape[1]
        pos = [p for p in self.positions if 0 <= p < T]
        if not pos:
            return

        # Capture only batch item 0
        # Shape: [P, D]
        self.hidden_PxD = hidden[0, pos, :].detach()


def noting_out_of_range(xs: List[int]) -> List[int]:
    # Small helper: sanitize to ints, no extra semantics.
    out = []
    for x in xs:
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


class LayerCaptureManager:
    """
    Register capture hooks for a set of layers once, then for each step:
    - set positions
    - run forward
    - read captured [P, D] per layer
    """
    def __init__(self, model: nn.Module, layers: List[int]):
        self.model = model
        self.layers = [int(l) for l in layers]
        self.layers_container = resolve_layers_container(model)

        self.hooks: Dict[int, _LayerCaptureHook] = {}
        self.handles: List[Any] = []

        for layer in self.layers:
            if layer < 0 or layer >= len(self.layers_container):
                raise IndexError(f"Layer index {layer} out of range (0..{len(self.layers_container)-1})")
            hook = _LayerCaptureHook(layer_idx=layer)
            handle = self.layers_container[layer].register_forward_hook(hook, always_call=True)
            self.hooks[layer] = hook
            self.handles.append(handle)

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def set_positions(self, positions: List[int]) -> None:
        positions = [int(p) for p in positions]
        for hook in self.hooks.values():
            hook.set_positions(positions)

    def clear(self) -> None:
        for hook in self.hooks.values():
            hook.clear()

    def get_capture(self) -> CaptureResult:
        by_layer: Dict[int, torch.Tensor] = {}
        for layer, hook in self.hooks.items():
            if hook.hidden_PxD is None:
                raise RuntimeError(f"Missing capture for layer={layer}. Did the forward pass run?")
            by_layer[layer] = hook.hidden_PxD
        # All hooks share same positions (after clamping inside hook they may differ),
        # but we keep the intended positions list as metadata.
        # The user should rely on their own positions list for indexing.
        positions = list(self.hooks[self.layers[0]].positions) if self.layers else []
        return CaptureResult(positions=positions, by_layer=by_layer)


# =============================================================================
# 3) SAE encode helpers: top-k and top-1 extraction
# =============================================================================

@torch.inference_mode()
def encode_topk(
    sae: nn.Module,
    hidden_PxD: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode hidden vectors with SAE and return:
    - topk_ids:  [P, k] (int64)
    - topk_vals: [P, k] (float32)
    - top1_id:   [P]    (int64)
    - top1_val:  [P]    (float32)

    Notes:
    - We cast activations to float32 for stable similarity metrics.
    """
    if hidden_PxD.ndim != 2:
        raise ValueError(f"hidden_PxD must be 2D [P,D], got shape={tuple(hidden_PxD.shape)}")

    acts_PxF = sae.encode(hidden_PxD)  # [P, F]
    acts_PxF = acts_PxF.to(torch.float32)

    if k <= 0:
        raise ValueError("k must be > 0")

    # top-k
    topk_vals, topk_idx = torch.topk(acts_PxF, k=min(k, acts_PxF.shape[-1]), dim=-1)
    topk_ids = topk_idx.to(torch.int64)

    # top-1
    top1_val = topk_vals[:, 0].contiguous()
    top1_id = topk_ids[:, 0].contiguous()

    return topk_ids, topk_vals, top1_id, top1_val


# =============================================================================
# 4) Similarity/distance utilities for top-k features
# =============================================================================

def jaccard_ids(a_ids: List[int], b_ids: List[int]) -> float:
    """
    Jaccard similarity between two sets of feature IDs.
    """
    sa = set(int(x) for x in a_ids)
    sb = set(int(x) for x in b_ids)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / max(1, union))


def weighted_jaccard(a: Dict[int, float], b: Dict[int, float]) -> float:
    """
    Weighted Jaccard similarity between two sparse vectors:
      sum_f min(a_f, b_f) / sum_f max(a_f, b_f)
    """
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    num = 0.0
    den = 0.0
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        num += min(av, bv)
        den += max(av, bv)
    return float(num / (den + 1e-12))


def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    """
    Cosine similarity between two sparse vectors represented as dicts.
    """
    if not a and not b:
        return 1.0
    # dot
    dot = 0.0
    for k, av in a.items():
        dot += float(av) * float(b.get(k, 0.0))
    # norms
    na = 0.0
    for av in a.values():
        na += float(av) * float(av)
    nb = 0.0
    for bv in b.values():
        nb += float(bv) * float(bv)
    denom = (na ** 0.5) * (nb ** 0.5) + 1e-12
    return float(dot / denom)


def topk_to_sparse(ids_1d: List[int], vals_1d: List[float]) -> Dict[int, float]:
    """
    Convert top-k (ids, vals) lists into a sparse dict fid -> activation.
    """
    out: Dict[int, float] = {}
    for fid, v in zip(ids_1d, vals_1d):
        fid_i = int(fid)
        vv = float(v)
        # Keep max if duplicates occur
        if fid_i not in out or vv > out[fid_i]:
            out[fid_i] = vv
    return out
