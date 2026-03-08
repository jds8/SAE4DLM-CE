import os
import json
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn


############################################################
# Helper functions to interpret & rebuild locally trained SAEs
#
# Context:
# - During training, each SAE is saved under a folder like:
#       /.../resid_post_layer_1/trainer_0/
#           ae.pt
#           config.json
#
# - training.py writes:
#       final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
#       torch.save(final, "<folder>/ae.pt")
#
#   ...or sometimes backup checkpoints of the form:
#       {
#         "step": step,
#         "ae": trainer.ae.state_dict(),
#         "optimizer": ...,
#         "config": ...,
#         "norm_factor": norm_factor,
#       }
#
# - We do NOT save a full nn.Module instance. We only save weights.
#   That means when we want to run interpretability, we need to
#   reconstruct an nn.Module that:
#       - knows about encoder/decoder weights
#       - can produce feature activations a = SAE.encode(x)
#
# This file does that reconstruction in a robust, heuristic way,
# without importing the original training code (StandardTrainer, etc.).
############################################################


def _load_raw_state_dict(ae_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the checkpoint file saved as ae.pt and return a bare state_dict
    mapping parameter/buffer name -> tensor.

    We handle two common formats:

    (A) Final checkpoint (most common at the end of training):
        {
            "W_dec": <Tensor[F,D]>,
            "b_dec": <Tensor[D]>,
            "W_enc": <Tensor[F,D]>,
            "b_enc": <Tensor[F]>,
            ...
        }
        i.e. a plain dict of tensors.

    (B) Backup checkpoint (during training):
        {
            "step": ...,
            "ae": { "W_dec": ..., "b_dec": ..., ... },  # <- state_dict
            "optimizer": ...,
            "config": ...,
            "norm_factor": ...,
        }

    We'll unwrap (B) so that we always return just the SAE's tensor dict.
    """
    ckpt = torch.load(ae_path, map_location="cpu")

    if isinstance(ckpt, dict):
        # Case A: all values are tensors => looks like final state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # plain state_dict

        # Case B: backup dict where "ae" is the state_dict
        if "ae" in ckpt and isinstance(ckpt["ae"], dict):
            ae_state = ckpt["ae"]
            if all(isinstance(v, torch.Tensor) for v in ae_state.values()):
                return ae_state

        # Case C: state wrapped in "state_dict"
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
            if all(isinstance(v, torch.Tensor) for v in sd.values()):
                return sd

    raise RuntimeError(
        f"[load_raw_state_dict] Unrecognized SAE checkpoint format at {ae_path}. "
        f"type(ckpt)={type(ckpt)}, keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
    )


def _pick_decoder_weight_key(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Heuristically choose which key in the state_dict corresponds to the
    decoder weight matrix. We prefer keys containing 'decoder' or 'w_dec',
    among 2D tensors, and break ties by largest number of elements.
    """
    candidates: List[Tuple[int, int, str]] = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue
        name_lower = k.lower()
        score = 0
        if "decoder" in name_lower:
            score += 100
        if "w_dec" in name_lower or "decoder.weight" in name_lower or (
            "dec" in name_lower and "weight" in name_lower
        ):
            score += 50
        if "weight" in name_lower:
            score += 10
        # We'll also store v.numel() to break ties for roughly "largest"
        candidates.append((score, v.numel(), k))

    if not candidates:
        raise ValueError(
            "[_pick_decoder_weight_key] No suitable 2D decoder-like tensor found in state_dict."
        )
    # sort descending by (score, numel)
    candidates.sort(reverse=True)
    return candidates[0][2]


def _pick_encoder_weight_key(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Heuristically choose which key looks like the encoder weight.
    We prefer keys containing 'encoder' or 'w_enc', among 2D tensors,
    but we penalize anything that also looks like 'decoder'.
    """
    candidates: List[Tuple[int, int, str]] = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue
        name_lower = k.lower()
        score = 0
        if "encoder" in name_lower:
            score += 100
        if "w_enc" in name_lower or "encoder.weight" in name_lower or (
            "enc" in name_lower and "weight" in name_lower
        ):
            score += 50
        if "weight" in name_lower:
            score += 10
        # If this ALSO looks like decoder, penalize
        if "decoder" in name_lower or "w_dec" in name_lower:
            score -= 200

        candidates.append((score, v.numel(), k))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    if best_score <= 0:
        # No convincing encoder key. We'll treat encoder as tied to decoder.
        return None

    return candidates[0][2]


def _maybe_transpose_to_FxD(mat: torch.Tensor) -> torch.Tensor:
    """
    We want all weight matrices in [F, D] shape:
    - F = number of SAE features (latents)
    - D = model hidden dim at that layer

    Some checkpoints might store [D, F] instead, so we transpose if needed.
    Heuristic rule: if first dim < second dim, we assume it's [D, F] and we transpose.
    """
    if mat.ndim != 2:
        raise ValueError("[_maybe_transpose_to_FxD] Expected a 2D tensor.")
    f, d = mat.shape
    if f < d:
        return mat.t().contiguous()
    return mat.contiguous()


def _extract_bias_vector(
    state_dict: Dict[str, torch.Tensor],
    candidate_names: List[str],
) -> Optional[torch.Tensor]:
    """
    Try to retrieve a 1D bias-like vector. We first try exact keys,
    then do a fuzzy match on name substrings.
    """
    # exact try
    for name in candidate_names:
        if name in state_dict and isinstance(state_dict[name], torch.Tensor):
            tensor = state_dict[name]
            if tensor.ndim == 1:
                return tensor.clone().to(torch.float32).contiguous()

    # fuzzy try
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 1:
            continue
        k_lower = k.lower()
        if any(n in k_lower for n in candidate_names):
            return v.clone().to(torch.float32).contiguous()

    return None


def _get_threshold_scalar_from_config(cfg: Dict[str, Any]) -> Optional[float]:
    """
    Some trainers store a global scalar threshold (e.g. TopK, JumpReLU) in the
    trainer config under something like trainer["threshold"].
    We'll parse it if it exists and is numeric / numeric-string.
    """
    trainer_cfg = cfg.get("trainer", {})
    thr_val = trainer_cfg.get("threshold", None)

    if thr_val is None:
        return None

    # direct float
    if isinstance(thr_val, (int, float)):
        return float(thr_val)

    # sometimes serialized as a string
    try:
        return float(str(thr_val))
    except Exception:
        return None


def _get_topk_from_config(cfg: Dict[str, Any]) -> Optional[int]:
    """
    Some sparsity trainers store 'k' (top-k) in trainer["k"].
    We'll parse it if present.
    """
    trainer_cfg = cfg.get("trainer", {})
    k_val = trainer_cfg.get("k", None)

    if k_val is None:
        return None

    try:
        return int(k_val)
    except Exception:
        return None


class LocalSAE(nn.Module):
    """
    A minimal self-contained SAE module for inference / interpretability.

    This is designed to approximate the behavior of several SAE variants
    we might have trained (Standard ReLU, TopK, JumpReLU, Gated...).

    It supports:
      - encode(x): returns feature activations a in shape [..., F]
        where x has shape [..., D] (e.g. [batch, seq_len, d_model])

    It does NOT need to support training, optimizer state, etc.
    It only needs to expose .encode() because our auto-interpretability
    pipeline relies on SAE.encode(hidden_states) -> activations.

    Internals:
    - We register key tensors (decoder/encoder weights, biases, gates, etc.)
      as buffers so they move with .to(device, dtype).
    - We store scalar / int hyperparams (threshold_scalar, k_topk) as Python
      attributes.

    Convention:
    - W_dec is [F, D] (rows = features, cols = hidden dim).
    - W_enc is [F, D] (rows = features, cols = hidden dim). If not present,
      we'll default to using W_dec as the encoder weights (tied weights).
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

        # Normalize the classifier name (e.g. "StandardTrainer", "JumpReLUTrainer", etc.)
        self.trainer_class_name = (trainer_class_name or "").lower()

        # Decoder weights are mandatory
        self.register_buffer("W_dec", W_dec_FD.to(torch.float32).to(device))

        # Encoder weights optional; if absent, we will reuse decoder weights
        if W_enc_FD is not None:
            self.register_buffer("W_enc", W_enc_FD.to(torch.float32).to(device))

        # Decoder bias (vector in hidden dimension D)
        if b_dec_D is not None:
            self.register_buffer("b_dec", b_dec_D.to(torch.float32).to(device))

        # Encoder bias (vector in feature dimension F)
        if b_enc_F is not None:
            self.register_buffer("b_enc", b_enc_F.to(torch.float32).to(device))

        # Threshold vector for JumpReLU / TopK etc. (length F)
        if threshold_vector_F is not None:
            self.register_buffer(
                "threshold_vector", threshold_vector_F.to(torch.float32).to(device)
            )

        # Gating parameters
        if gate_bias_F is not None:
            self.register_buffer(
                "gate_bias", gate_bias_F.to(torch.float32).to(device)
            )
        if r_mag_F is not None:
            self.register_buffer("r_mag", r_mag_F.to(torch.float32).to(device))
        if mag_bias_F is not None:
            self.register_buffer(
                "mag_bias", mag_bias_F.to(torch.float32).to(device)
            )

        # These do not need to be registered as buffers
        self.threshold_scalar = threshold_scalar
        self.k_topk = k_topk

    def _linear_preact(self, x_BTD: torch.Tensor) -> torch.Tensor:
        """
        Compute the raw pre-activation in feature space, shape [B, T, F].

        If we are missing a dedicated encoder matrix, we fall back to
        using the decoder matrix as a tied weight (common in standard SAEs).
        """
        W_enc = getattr(self, "W_enc", None)
        if W_enc is None:
            W_enc = self.W_dec  # weight tying fallback

        # einsum("...d,fd->...f") multiplies last dim of x by rows of W_enc
        # x_BTD is [..., D], W_enc is [F, D]
        x_cast = x_BTD.to(W_enc.dtype)
        pre_BTF = torch.einsum("...d,fd->...f", x_cast, W_enc)

        # Many trainer variants add a bias in feature space
        # Standard / TopK / JumpReLU uses b_enc
        # Gated variants may instead use gate_bias
        if "gated" in self.trainer_class_name:
            gate_bias = getattr(self, "gate_bias", None)
            if gate_bias is not None:
                pre_BTF = pre_BTF + gate_bias
        else:
            b_enc = getattr(self, "b_enc", None)
            if b_enc is not None:
                pre_BTF = pre_BTF + b_enc

        return pre_BTF

    def _nonlinear_activation(self, pre_BTF: torch.Tensor) -> torch.Tensor:
        """
        Apply the SAE's sparsifying nonlinearity to produce features a_BTF.
        We try to emulate:
        - Standard ReLU
        - TopK / BatchTopK (+optional global threshold)
        - JumpReLU (subtract threshold then ReLU)
        - Gated ReLU variants
        """
        name = self.trainer_class_name

        # JumpReLU: ReLU(pre - threshold)
        if "jumprelu" in name:
            thr_vec = getattr(self, "threshold_vector", None)
            if thr_vec is not None:
                shifted = pre_BTF - thr_vec
            else:
                scalar_thr = (
                    self.threshold_scalar if self.threshold_scalar is not None else 0.0
                )
                shifted = pre_BTF - scalar_thr
            return torch.relu(shifted)

        # TopK / BatchTopK: (pre - thr) -> ReLU -> keep only top-k per position
        if ("topk" in name) or ("batchtopk" in name):
            out = pre_BTF

            thr_vec = getattr(self, "threshold_vector", None)
            if thr_vec is not None:
                out = out - thr_vec
            elif self.threshold_scalar is not None:
                out = out - self.threshold_scalar

            out = torch.relu(out)

            # Retain only k largest features per token position
            if self.k_topk is not None and 0 < self.k_topk < out.shape[-1]:
                k = self.k_topk
                topk_vals, topk_idx = torch.topk(out, k=k, dim=-1)
                zeros = torch.zeros_like(out)
                out = zeros.scatter(-1, topk_idx, topk_vals)
            return out

        # Gated variants: ReLU(pre) * sigmoid(r_mag * pre + mag_bias)
        if "gated" in name:
            base = torch.relu(pre_BTF)
            r_mag = getattr(self, "r_mag", None)
            mag_bias = getattr(self, "mag_bias", None)

            if (r_mag is not None) or (mag_bias is not None):
                # Broadcast-friendly form
                r = r_mag if r_mag is not None else 1.0
                m = mag_bias if mag_bias is not None else 0.0
                gate = torch.sigmoid(r * pre_BTF + m)
                base = base * gate
            return base

        # Default: standard ReLU
        return torch.relu(pre_BTF)

    def encode(self, x_BTD: torch.Tensor) -> torch.Tensor:
        """
        Core public API used by interpretability code.
        Input:  x_BTD with shape [batch, seq_len, d_model]
        Output: a_BTF with shape [batch, seq_len, n_features]
        """
        pre_BTF = self._linear_preact(x_BTD)
        a_BTF = self._nonlinear_activation(pre_BTF)
        return a_BTF

    def decode(self, a_BTF: torch.Tensor) -> torch.Tensor:
        """
        (Optional, not strictly required by autointerp.)
        Reconstruct x_hat in hidden space from feature activations.
        x_hat_BTD = a_BTF @ W_dec + b_dec, with broadcasting.
        """
        z = a_BTF.to(self.W_dec.dtype)
        x_hat_BTD = torch.einsum("...f,fd->...d", z, self.W_dec)
        b_dec = getattr(self, "b_dec", None)
        if b_dec is not None:
            x_hat_BTD = x_hat_BTD + b_dec
        return x_hat_BTD


def _build_local_sae_from_folder(folder: str, device: str) -> LocalSAE:
    """
    High-level loader that:
      - reads <folder>/config.json
      - reads <folder>/ae.pt
      - heuristically extracts the SAE architecture & parameters
      - returns a LocalSAE (nn.Module)

    This is the main constructor we will call from `load_sae`.
    """
    cfg_path = os.path.join(folder, "config.json")
    ae_path = os.path.join(folder, "ae.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"[build_local_sae_from_folder] Missing config.json at {cfg_path}"
        )
    if not os.path.exists(ae_path):
        raise FileNotFoundError(
            f"[build_local_sae_from_folder] Missing ae.pt at {ae_path}"
        )

    # Load trainer config (JSON saved at training time)
    with open(cfg_path, "r") as f:
        cfg_json = json.load(f)

    trainer_cfg = cfg_json.get("trainer", {}) or {}
    trainer_class_name = str(trainer_cfg.get("trainer_class", ""))

    # Load raw SAE state dict from ae.pt
    raw_state = _load_raw_state_dict(ae_path)

    # -------- decoder weights --------
    dec_key = _pick_decoder_weight_key(raw_state)
    W_dec = raw_state[dec_key].clone().to(torch.float32).contiguous()
    W_dec = _maybe_transpose_to_FxD(W_dec)  # shape [F, D] after this

    # Optional decoder bias in hidden-dim space
    b_dec = _extract_bias_vector(
        raw_state,
        candidate_names=[
            "decoder.bias",
            "decoder_bias",
            "b_dec",
            "bias",
            "dec_bias",
        ],
    )

    # -------- encoder weights --------
    enc_key = _pick_encoder_weight_key(raw_state)
    if enc_key is not None:
        W_enc = raw_state[enc_key].clone().to(torch.float32).contiguous()
        W_enc = _maybe_transpose_to_FxD(W_enc)  # shape [F, D]
    else:
        W_enc = None

    # Possible encoder bias (bias in feature space F)
    b_enc = _extract_bias_vector(
        raw_state,
        candidate_names=[
            "encoder.bias",
            "b_enc",
            "enc_bias",
            "bias_enc",
        ],
    )

    # "gate_bias" for gated variants
    gate_bias = _extract_bias_vector(raw_state, candidate_names=["gate_bias"])

    # Threshold vector for JumpReLU / TopK etc.
    threshold_vec = _extract_bias_vector(
        raw_state,
        candidate_names=["threshold", "thresholds"],
    )

    # Scalar threshold and top-k from trainer config
    threshold_scalar = _get_threshold_scalar_from_config(cfg_json)
    k_topk = _get_topk_from_config(cfg_json)

    # Gated magnitude parameters (optional)
    r_mag = _extract_bias_vector(raw_state, candidate_names=["r_mag"])
    mag_bias = _extract_bias_vector(raw_state, candidate_names=["mag_bias"])

    # Build the LocalSAE as an nn.Module
    sae = LocalSAE(
        W_dec_FD=W_dec,
        W_enc_FD=W_enc,
        b_dec_D=b_dec,
        b_enc_F=b_enc,
        trainer_class_name=trainer_class_name,
        threshold_scalar=threshold_scalar,
        threshold_vector_F=threshold_vec,
        k_topk=k_topk,
        gate_bias_F=gate_bias,
        r_mag_F=r_mag,
        mag_bias_F=mag_bias,
        device=device,
    )

    return sae


def load_sae(sae_path: str, device: str, dtype: torch.dtype) -> nn.Module:
    """
    Public entry point used by run_eval.py.

    Arguments:
        sae_path: Full path to the 'ae.pt' file, e.g.
            "/.../resid_post_layer_1/trainer_0/ae.pt"
        device:  Device string ("cuda", "cuda:0", "cpu", etc.)
        dtype:   Torch dtype (torch.float32, torch.bfloat16, etc.)

    Behavior:
        - Derive the containing folder from sae_path.
        - Rebuild a LocalSAE(nn.Module) using that folder's config.json + ae.pt.
        - Move it to (device, dtype).
        - Put it in eval() mode.
        - Return it.

    Returned object MUST expose:
        .encode(hidden_states_BLD) -> feature_acts_BLF
    which is exactly what the autointerp pipeline expects.
    """
    sae_dir = os.path.dirname(sae_path)

    # Build LocalSAE in the specified device
    sae = _build_local_sae_from_folder(sae_dir, device=device)

    # Cast to final device / dtype for inference
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()

    return sae
