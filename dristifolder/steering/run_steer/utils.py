# utils.py
# -*- coding: utf-8 -*-
import json
import os
import re
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple


def get_features_by_layers(features_file: str) -> Dict[int, List[int]]:
    with open(features_file, "r", encoding="utf-8") as f:
        features_by_layer = json.load(f)
        features_by_layer = {int(key): [int(v) for v in values] for key, values in features_by_layer.items()}
        return features_by_layer


# =========================
# Local DL-SAE lightweight loader & wrappers
# =========================

def _flatten_state_dict(sd: dict) -> dict:
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def _find_tensor_keys(state_dict: dict, pattern_list: List[str]) -> List[str]:
    out = []
    for k in state_dict.keys():
        kl = k.lower()
        if any(p in kl for p in pattern_list):
            out.append(k)
    return out


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


def _extract_bias_vector(state_dict: dict, names: List[str]) -> Optional[torch.Tensor]:
    for n in names:
        if n in state_dict and isinstance(state_dict[n], torch.Tensor):
            v = state_dict[n]
            if v.ndim == 1:
                return v
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 1:
            continue
        kl = k.lower()
        if any(n in kl for n in names):
            return v
    return None


def _ensure_FxD(mat: torch.Tensor) -> torch.Tensor:
    if mat.ndim != 2:
        raise ValueError("Decoder/Encoder matrix must be 2D.")
    f, d = mat.shape
    if f < d:
        mat = mat.t()
    return mat


def _extract_decoder_FxD(state_dict: dict, decoder_key: str) -> torch.Tensor:
    W = state_dict[decoder_key]
    if not isinstance(W, torch.Tensor) or W.ndim != 2:
        raise ValueError(f"Decoder weight at {decoder_key} must be a 2D tensor.")
    W = W.to(torch.float32).contiguous()
    W = _ensure_FxD(W)
    return W


def _extract_encoder_FxD(state_dict: dict, encoder_key: str) -> torch.Tensor:
    W = state_dict[encoder_key]
    if not isinstance(W, torch.Tensor) or W.ndim != 2:
        raise ValueError(f"Encoder weight at {encoder_key} must be a 2D tensor.")
    W = W.to(torch.float32).contiguous()
    W = _ensure_FxD(W)
    return W


def _get_k_from_config(cfg: dict) -> Optional[int]:
    tr = cfg.get("trainer", {}) if isinstance(cfg.get("trainer"), dict) else {}
    k = tr.get("k", None)
    try:
        return int(k) if k is not None else None
    except Exception:
        return None


class LocalSAE(nn.Module):
    """
    A minimal local SAE wrapper that supports common DL-SAE variants:
    - Standard (ReLU): a = ReLU(x W_enc^T + b_enc)
    - TopK / BatchTopK: same as ReLU then keep top-k; optionally subtract scalar/vector threshold before ReLU
    - JumpReLU: a = ReLU((x W_enc^T + b_enc) - threshold_vector)
    - Gated: ReLU(pre) * sigmoid(r_mag * pre + mag_bias) (if gate params exist)

    decode: x_hat = a @ W_dec + b_dec (when b_dec present)
    NOTE: W_dec must be [F, D]; W_enc must be [F, D].
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

    b_dec = _extract_bias_vector(sd, names=["decoder.bias", "decoder_bias", "b_dec", "bias", "dec_bias"])

    enc_key = _pick_best_encoder_key(sd)
    W_enc_FD = _extract_encoder_FxD(sd, enc_key) if enc_key is not None else None

    b_enc = _extract_bias_vector(sd, names=["encoder.bias", "b_enc", "enc_bias", "bias_enc"])
    gate_bias = _extract_bias_vector(sd, names=["gate_bias"])

    threshold_vec = _extract_bias_vector(sd, names=["threshold", "thresholds"])
    threshold_scalar = _get_k_from_config(cfg)  # Not a scalar threshold; keep for compatibility if ever used elsewhere

    # We do not read scalar threshold from cfg in most top-k trainers; keeping API parity.

    r_mag = _extract_bias_vector(sd, names=["r_mag"])
    mag_bias = _extract_bias_vector(sd, names=["mag_bias"])

    # Get k from config when available
    k_topk = _get_k_from_config(cfg)

    sae = LocalSAE(
        W_dec_FD=W_dec_FD,
        W_enc_FD=W_enc_FD,
        b_dec_D=b_dec,
        b_enc_F=b_enc,
        trainer_class_name=trainer_class,
        threshold_scalar=None,              # top-k uses threshold_vec/scalar differently; not used here
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
        else:
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

    # Fallbacks
    t0 = os.path.join(base_dir, f"resid_post_layer_{layer}", "trainer_0")
    if os.path.isdir(t0):
        return t0
    return trainer_dirs[0]


def _get_decoder_weights_any(sae_obj: nn.Module) -> torch.Tensor:
    if hasattr(sae_obj, "W_dec"):
        return sae_obj.W_dec
    if hasattr(sae_obj, "decoder") and hasattr(sae_obj.decoder, "weight"):
        W = sae_obj.decoder.weight
        if W.shape[0] < W.shape[1]:
            W = W.t()
        return W
    if hasattr(sae_obj, "W_out"):
        W = sae_obj.W_out
        if W.shape[0] < W.shape[1]:
            W = W.t()
        return W
    raise AttributeError("Decoder weights not found on SAE object.")


# =========================
# Public API used by steer.py
# =========================

def get_sae(
    model_type: str,
    layer: int,
    saes: Dict[int, nn.Module],
    backend: str = "dl_local",
    dl_local_dir: Optional[str] = None,
    device: str = "cpu",
    trainer_name: Optional[str] = None,
    k_topk: Optional[int] = None,
) -> nn.Module:
    """
    Return (and cache) a LocalSAE for a given layer.
    - backend: only 'dl_local' supported.
    - dl_local_dir: base folder containing resid_post_layer_{layer}/trainer_*/ subdirs.
    - trainer_name: e.g., 'trainer_0'
    - k_topk: choose trainer whose config.json has trainer.k == k_topk
    """
    if layer in saes:
        return saes[layer]
    if backend != "dl_local":
        raise ValueError("Only 'dl_local' backend is supported in this setup.")
    assert dl_local_dir is not None, "dl_local_dir must be provided for local DL SAEs."

    trainer_dir = _pick_trainer_dir(dl_local_dir, layer, trainer_name=trainer_name, k_topk=k_topk)
    sae = _load_dictionary_learning_sae_from_folder(trainer_dir, device=device)
    saes[layer] = sae
    return sae


def try_get_final_norm_and_lm_head(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """
    Heuristically resolve the final norm and lm_head across families.
    """
    # lm_head
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    else:
        raise AttributeError("Could not locate lm_head on model.")

    # final norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
        final_norm = model.model.final_layernorm
    elif hasattr(model, "final_layer_norm"):
        final_norm = model.final_layer_norm
    else:
        raise AttributeError("Could not locate final layer norm (model.model.norm / final_layernorm / final_layer_norm).")

    return final_norm, lm_head


def cache_logit_lens(
    layer: int,
    saes: Dict[int, nn.Module],
    model_type: str,
    final_layer_norm: nn.Module,
    lm_head: nn.Module,
    k: int,
    backend: str = "dl_local",
    dl_local_dir: Optional[str] = None,
    device: str = "cpu",
    trainer_name: Optional[str] = None,
    k_topk: Optional[int] = None,
):
    """
    Build top-k tokens per SAE feature by projecting decoder rows through final_layer_norm and lm_head.
    """
    sae = get_sae(
        model_type=model_type,
        layer=layer,
        saes=saes,
        backend=backend,
        dl_local_dir=dl_local_dir,
        device=device,
        trainer_name=trainer_name,
        k_topk=k_topk,
    )

    final_layer_norm = final_layer_norm.cpu()
    lm_head = lm_head.cpu()

    decoder_weights = _get_decoder_weights_any(sae).detach().cpu().to(torch.float32)  # [F, D]
    decoder_weights = final_layer_norm(decoder_weights)  # LN on feature vectors
    logits = lm_head(decoder_weights)  # [F, |V|]
    confidence = torch.softmax(logits, dim=1).detach().cpu()

    topk = torch.topk(confidence, dim=1, k=k)
    return topk, confidence, logits
