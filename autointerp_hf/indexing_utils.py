# autointerp_hf/indexing_utils.py
from __future__ import annotations
import torch
from torch import Tensor
import random
import numpy as np

@torch.no_grad()
def index_with_buffer(
    tensor_2d: Tensor,
    indices: Tensor,
    buffer: int,
) -> Tensor:
    """
    Extract a fixed-width window around each (row_idx, center_pos) pair in `indices`.

    tensor_2d: [N, L]
    indices:   [K, 2] where [:,0] = row_idx, [:,1] = center_pos
    buffer:    how many tokens we include to left/right of center_pos

    Returns: [K, 2*buffer + 1], i.e. each row is the slice from
        [center_pos-buffer ... center_pos+buffer]
    We assume caller only passes indices with valid buffer margins.
    """
    windows = []
    for (row_idx, center_pos) in indices.tolist():
        start = center_pos - buffer
        end = center_pos + buffer + 1
        win = tensor_2d[row_idx, start:end]
        windows.append(win)
    return torch.stack(windows, dim=0)


@torch.no_grad()
def get_k_largest_indices(
    acts_2d: Tensor,
    k: int,
    buffer: int,
    no_overlap: bool,
) -> Tensor:
    """
    Find up to k positions (row_idx, pos) with highest activation in acts_2d.
    Enforce no overlapping windows within the same row if no_overlap=True.

    acts_2d: [N, L]
    Returns: [K, 2] with dtype long, where each row = [row_idx, pos]
    """

    N, L = acts_2d.shape
    flat_vals = acts_2d.view(-1)
    # sorted indices in descending order
    sorted_idx = torch.argsort(flat_vals, descending=True)

    chosen = []
    for flat_i in sorted_idx.tolist():
        if len(chosen) >= k:
            break
        row_idx = flat_i // L
        pos_idx = flat_i % L

        # ensure buffer boundaries are valid
        if pos_idx < buffer or pos_idx >= (L - buffer):
            continue

        if no_overlap:
            conflict = False
            for (r_existing, p_existing) in chosen:
                if r_existing == row_idx and abs(p_existing - pos_idx) <= buffer:
                    conflict = True
                    break
            if conflict:
                continue

        chosen.append((row_idx, pos_idx))

    if len(chosen) == 0:
        return torch.zeros((0, 2), dtype=torch.long)

    return torch.tensor(chosen, dtype=torch.long, device=acts_2d.device)

@torch.no_grad()
def get_iw_sample_indices(
    acts: torch.Tensor,
    k: int,
    buffer: int,
) -> torch.Tensor:
    """
    Importance-weighted sampling of k (sequence_idx, position_idx) pairs.

    Args:
        acts:   [N, L] tensor of activation magnitudes for ONE latent
                after thresholding out top-k peaks.
                N = number of sequences in the dataset
                L = sequence length (context_length)
        k:      how many samples to draw
        buffer: how many tokens of context we later want left/right,
                so we should avoid sampling very near the edges. i.e.
                valid positions are [buffer, L-buffer).

    Returns:
        indices: LongTensor of shape [k, 2]
                 each row is [seq_idx, pos_idx]
                 on CPU
    """

    # Ensure 2D
    assert acts.ndim == 2, f"Expected acts shape [N, L], got {acts.shape}"

    N, L = acts.shape

    # 1) Mask out the first/last `buffer` tokens so later we can safely
    #    grab `buffer` tokens of left/right context without going OOB.
    #    We also clamp negatives to 0.
    weight = acts.clone()  # [N, L]
    if buffer > 0:
        weight[:, :buffer] = 0.0
        weight[:, L - buffer :] = 0.0
    weight = torch.clamp(weight, min=0.0)

    # 2) Flatten to 1D importance scores.
    #    We upcast to float64 for stable normalization.
    flat_w = weight.reshape(-1).to(torch.float64)  # [N*L]

    total_mass = flat_w.sum().item()

    if total_mass <= 0.0 or not np.isfinite(total_mass):
        # Edge case: no positive mass (can happen if this latent basically
        # didn't fire at all after top-activations were zeroed out).
        # Fallback to uniform sampling over all valid (non-edge) positions.
        valid_mask = torch.ones_like(weight, dtype=torch.bool)
        if buffer > 0:
            valid_mask[:, :buffer] = False
            valid_mask[:, L - buffer :] = False
        valid_flat = valid_mask.reshape(-1)

        valid_indices_flat = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
        if valid_indices_flat.numel() == 0:
            # As a last-ditch fallback, allow *any* position
            valid_indices_flat = torch.arange(N * L, device=weight.device)

        num_to_sample = min(k, valid_indices_flat.numel())
        # torch.randperm is uniform without replacement
        perm = torch.randperm(valid_indices_flat.numel(), device=weight.device)[:num_to_sample]
        sampled_flat_idx = valid_indices_flat[perm]
    else:
        # 3) Normalize into a proper probability distribution in torch.
        probs = flat_w / flat_w.sum()

        # torch.multinomial can sample without replacement using these probs.
        # We also cap k so we don't ask for more than we have.
        num_candidates = probs.shape[0]
        num_to_sample = min(k, num_candidates)

        # When probs may have tiny negative due to numeric noise after clamp,
        # .clamp(min=0) above already handled; still we renormalize after sum.
        probs = probs / probs.sum()

        sampled_flat_idx = torch.multinomial(
            probs,
            num_samples=num_to_sample,
            replacement=False,
        )  # [num_to_sample]

    # 4) Map flat indices back to (seq_idx, pos_idx)
    seq_idx = sampled_flat_idx // L  # integer division
    pos_idx = sampled_flat_idx % L

    out = torch.stack([seq_idx.to(torch.long), pos_idx.to(torch.long)], dim=-1)
    # shape [num_to_sample, 2] on current device

    # We return on CPU to match downstream index_with_buffer(...) expectations
    return out.cpu()