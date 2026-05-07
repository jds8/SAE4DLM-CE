"""
visualize_trajectory.py

For each (layer, feature) pair, loads that feature's own top activating text
from interp_results/ and visualizes the denoising trajectory.

Produces per feature:
  1. trajectory_L{l}_F{f}.pdf   — activation heatmap (M=masked, *=just committed)
  2. commit_order_L{l}_F{f}.pdf — when each token commits + activation at that moment

Usage:
    # Recommended: each feature uses its own top text from interp_results/
    python visualize_trajectory.py \
        --pairs 23:7116 23:6193 10:12343 10:3806 1:2788 \
        --interp_dir interp_results \
        --out_dir trajectory_plots

    # Or specify one shared text manually
    python visualize_trajectory.py \
        --pairs 23:7116 \
        --text "We prove that the algorithm converges, hence comprising a unique solution." \
        --out_dir trajectory_plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME   = "Dream-org/Dream-v0-Base-7B"
SAE_REPO     = "AwesomeInterpretability/dlm-mask-topk-sae"
SAE_MODEL_ID = "Dream-org_Dream-v0-Base-7B"
TRAINER      = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, nargs="+", required=True,
                        help="layer:feature pairs e.g. 23:7116 10:12343")
    parser.add_argument("--interp_dir", type=str, default=None,
                        help="Directory with results_L{l}_F{f}_{alg}.json. "
                             "Each feature uses its own top activating text.")
    parser.add_argument("--text", type=str, default=None,
                        help="Single shared text for all features.")
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--prompt_max_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--load_in_4bit", action="store_false", default=False)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="trajectory_plots")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top texts per feature to show in one plot")
    return parser.parse_args()


def resolve_pairs(pairs_arg) -> Dict[int, List[int]]:
    d = defaultdict(list)
    for p in pairs_arg:
        l, f = p.split(":")
        d[int(l)].append(int(f))
    return dict(d)


def get_top_texts_for_feature(interp_dir: str, layer: int, feature: int, alg: str, top_k: int = 3) -> List[str]:
    """Load top-k activating texts for a specific feature from its JSON file."""
    path = os.path.join(interp_dir, f"results_L{layer}_F{feature}_{alg}.json")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t["text"] for t in data["top_texts"][:top_k]]


def load_model_and_tokenizer(args):
    print("Loading Dream model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if quant_config:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = args.device
    model = AutoModel.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_sae(layer, device):
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    ae_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"saes_mask_{SAE_MODEL_ID}_top_k/resid_post_layer_{layer}/trainer_{TRAINER}/ae.pt",
    )
    sae = AutoEncoderTopK.from_pretrained(ae_path, device=device)
    sae.eval()
    return sae


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Cannot find layers")


def collect_trajectory(text, model, tokenizer, sae, layer, features, mask_token_id, args):
    """Collect denoising trajectory for one text, one layer, multiple features."""
    device = next(model.parameters()).device

    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=args.prompt_max_len, padding=False, add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device).bool()
    input_len = input_ids.shape[1]

    captured = {}

    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(h):
            captured["h"] = h.detach()

    handle = get_layers(model)[layer].register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            out = model.diffusion_generate(
                inputs=input_ids, attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens, steps=args.steps,
                temperature=1.0, top_p=0.95, alg=args.alg, alg_temp=0.0,
                output_history=True, return_dict_in_generate=True, do_sample=True,
            )

        history = getattr(out, "history", None) or (out.get("history") if isinstance(out, dict) else None)
        if history is None:
            return None

        if isinstance(history, torch.Tensor):
            arr = history.detach().cpu()
            step_seqs = [arr[t, 0].tolist() if arr.dim() == 3 else arr[t].tolist()
                         for t in range(arr.shape[0])]
        else:
            step_seqs = []
            for item in history:
                a = item.detach().cpu() if isinstance(item, torch.Tensor) else torch.tensor(item)
                step_seqs.append(a[0].tolist() if a.dim() == 2 else a.tolist())

        T = len(step_seqs)
        S = len(step_seqs[0]) - input_len

        activations = {f: np.zeros((T, S), dtype=np.float32) for f in features}
        step_ids_all, step_tokens_all = [], []

        for t_idx, step_ids in enumerate(step_seqs):
            ids = torch.tensor(step_ids, dtype=torch.long, device=device).unsqueeze(0)
            am  = torch.ones_like(ids, dtype=torch.bool)
            with torch.inference_mode():
                _ = model(input_ids=ids, attention_mask=am)
            hidden = captured.get("h")
            if hidden is not None:
                hidden_cast = hidden.to(dtype=sae.encoder.weight.dtype)
                with torch.no_grad():
                    feat_acts = sae.encode(hidden_cast)
                for f in features:
                    activations[f][t_idx] = feat_acts[0, input_len:, f].cpu().float().numpy()
            gen_ids = ids[0, input_len:].cpu().tolist()
            step_ids_all.append(gen_ids)
            step_tokens_all.append([tokenizer.decode([i]) for i in gen_ids])

    finally:
        handle.remove()

    return {
        "text": text,
        "step_ids": step_ids_all,
        "step_tokens": step_tokens_all,
        "activations": activations,
        "mask_token_id": mask_token_id,
    }


def plot_activation_heatmap(traj, layer, feature, out_dir, dpi=150):
    acts = traj["activations"][feature]  # [T, S]
    step_ids = traj["step_ids"]
    step_tokens = traj["step_tokens"]
    mask_id = traj["mask_token_id"]
    T, S = acts.shape

    fig, ax = plt.subplots(figsize=(max(10, T * 0.18), max(6, S * 0.35)))
    im = ax.imshow(acts.T, aspect="auto", origin="upper", cmap="Reds", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Feature Activation")

    for t in range(T):
        for s in range(S):
            tok_id = step_ids[t][s] if t < len(step_ids) and s < len(step_ids[t]) else -1
            if tok_id == mask_id:
                ax.text(t, s, "M", ha="center", va="center", fontsize=5, color="steelblue", alpha=0.7)
            elif t > 0 and s < len(step_ids[t-1]) and step_ids[t-1][s] == mask_id:
                ax.text(t, s, "*", ha="center", va="center", fontsize=7, color="black", fontweight="bold")

    final_tokens = step_tokens[-1] if step_tokens else []
    ax.set_yticks(range(S))
    ax.set_yticklabels([repr(tok)[:8] for tok in final_tokens], fontsize=7)
    ax.set_xlabel("Diffusion Timestep  (0=fully masked → 63=decoded)")
    ax.set_ylabel("Token Position (generated)")
    ax.set_title(
        f"L{layer}-F{feature} | M=masked  *=just committed\n"
        f'"{traj["text"][:80]}..."', fontsize=9,
    )
    plt.tight_layout()
    fname = os.path.join(out_dir, f"trajectory_L{layer}_F{feature}.pdf")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_commit_order(traj, layer, feature, out_dir, dpi=150):
    acts = traj["activations"][feature]  # [T, S]
    step_ids = traj["step_ids"]
    step_tokens = traj["step_tokens"]
    mask_id = traj["mask_token_id"]
    T, S = acts.shape
    final_tokens = step_tokens[-1] if step_tokens else [""] * S

    commit_times, commit_acts, token_labels = [], [], []
    for s in range(S):
        committed_at = T - 1
        for t in range(T):
            if s < len(step_ids[t]) and step_ids[t][s] != mask_id:
                if t == 0 or step_ids[t-1][s] == mask_id:
                    committed_at = t
                    break
        commit_times.append(committed_at)
        commit_acts.append(float(acts[committed_at, s]))
        token_labels.append(repr(final_tokens[s])[:10] if s < len(final_tokens) else "?")

    colors = ["tomato" if ct < T // 3 else ("gold" if ct < 2 * T // 3 else "steelblue")
              for ct in commit_times]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, S * 0.4), 8))

    ax = axes[0]
    ax.bar(range(S), commit_times, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(T // 3,     linestyle="--", color="tomato",    alpha=0.6, label="Early/Mid")
    ax.axhline(2 * T // 3, linestyle="--", color="steelblue", alpha=0.6, label="Mid/Late")
    ax.set_xticks(range(S))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Timestep committed")
    ax.set_title(f"L{layer}-F{feature}: Token commitment order\nRed=Early  Gold=Mid  Blue=Late")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(range(S), commit_acts, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(S))
    ax2.set_xticklabels(token_labels, rotation=90, fontsize=7)
    ax2.set_ylabel("Feature activation at commit moment")
    ax2.set_title("Feature activation at the moment each token is committed")

    plt.suptitle(f'L{layer}-F{feature}  |  "{traj["text"][:80]}..."', fontsize=9)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"commit_order_L{layer}_F{feature}.pdf")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_top_k_commit_order(trajs: list, layer: int, feature: int, out_dir: str, dpi: int = 150):
    """
    One plot with top-k texts stacked as rows.
    Each row = one text, showing commit time bars + activation line.
    """
    k = len(trajs)
    if k == 0:
        return

    fig, axes = plt.subplots(k, 1, figsize=(max(12, 64 * 0.18), 5 * k), squeeze=False)

    for row, traj in enumerate(trajs):
        acts = traj["activations"][feature]  # [T, S]
        step_ids = traj["step_ids"]
        step_tokens = traj["step_tokens"]
        mask_id = traj["mask_token_id"]
        T, S = acts.shape
        final_tokens = step_tokens[-1] if step_tokens else [""] * S

        commit_times, commit_acts, token_labels = [], [], []
        for s in range(S):
            committed_at = T - 1
            for t in range(T):
                if s < len(step_ids[t]) and step_ids[t][s] != mask_id:
                    if t == 0 or step_ids[t-1][s] == mask_id:
                        committed_at = t
                        break
            commit_times.append(committed_at)
            commit_acts.append(float(acts[committed_at, s]))
            token_labels.append(repr(final_tokens[s])[:8] if s < len(final_tokens) else "?")

        colors = ["tomato" if ct < T // 3 else ("gold" if ct < 2 * T // 3 else "steelblue")
                  for ct in commit_times]

        ax = axes[row][0]
        ax2 = ax.twinx()

        ax.bar(range(S), commit_times, color=colors, alpha=0.6, edgecolor="white", linewidth=0.3)
        ax.axhline(T // 3,     linestyle="--", color="tomato",    alpha=0.5)
        ax.axhline(2 * T // 3, linestyle="--", color="steelblue", alpha=0.5)
        ax2.plot(range(S), commit_acts, color="black", linewidth=1.5,
                 marker="o", markersize=3, label="activation at commit")

        ax.set_xticks(range(S))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=6)
        ax.set_ylabel("Timestep committed", fontsize=8)
        ax2.set_ylabel("Activation", fontsize=8)
        ax.set_title(
            f'[Text {row+1}] "{traj["text"][:80]}..."',
            fontsize=8,
        )
        if row == 0:
            ax.legend(
                handles=[
                    plt.Rectangle((0,0),1,1, color="tomato",    alpha=0.6, label="Early"),
                    plt.Rectangle((0,0),1,1, color="gold",      alpha=0.6, label="Mid"),
                    plt.Rectangle((0,0),1,1, color="steelblue", alpha=0.6, label="Late"),
                ],
                fontsize=7, loc="upper right",
            )

    plt.suptitle(
        f"L{layer}-F{feature} | Top-{k} texts: commit order (bars) + activation at commit (line)\nRed=Early  Gold=Mid  Blue=Late",
        fontsize=10,
    )
    plt.tight_layout()
    fname = os.path.join(out_dir, f"top{k}_commit_L{layer}_F{feature}.pdf")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_top_k_heatmaps(trajs: list, layer: int, feature: int, out_dir: str, dpi: int = 150):
    """
    One plot with top-k activation heatmaps stacked as rows.
    """
    k = len(trajs)
    if k == 0:
        return

    T_max = max(len(t["step_ids"]) for t in trajs)
    S_max = max(len(t["step_ids"][0]) for t in trajs if t["step_ids"])

    fig, axes = plt.subplots(k, 1, figsize=(max(10, T_max * 0.18), max(5, S_max * 0.3) * k), squeeze=False)

    for row, traj in enumerate(trajs):
        acts = traj["activations"][feature]  # [T, S]
        step_ids = traj["step_ids"]
        step_tokens = traj["step_tokens"]
        mask_id = traj["mask_token_id"]
        T, S = acts.shape
        ax = axes[row][0]

        vmax = acts.max() if acts.max() > 0 else 1.0
        im = ax.imshow(acts.T, aspect="auto", origin="upper", cmap="Reds",
                       interpolation="nearest", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Activation", shrink=0.8)

        for t in range(T):
            for s in range(S):
                tok_id = step_ids[t][s] if t < len(step_ids) and s < len(step_ids[t]) else -1
                if tok_id == mask_id:
                    ax.text(t, s, "M", ha="center", va="center", fontsize=4, color="steelblue", alpha=0.6)
                elif t > 0 and s < len(step_ids[t-1]) and step_ids[t-1][s] == mask_id:
                    ax.text(t, s, "*", ha="center", va="center", fontsize=6, color="black", fontweight="bold")

        final_tokens = step_tokens[-1] if step_tokens else []
        ax.set_yticks(range(S))
        ax.set_yticklabels([repr(tok)[:6] for tok in final_tokens], fontsize=6)
        ax.set_xlabel("Timestep", fontsize=7)
        ax.set_title(f'[Text {row+1}] "{traj["text"][:80]}..."', fontsize=8)

    plt.suptitle(
        f"L{layer}-F{feature} | Top-{k} texts activation heatmap\n"
        "M=masked  *=just committed",
        fontsize=10,
    )
    plt.tight_layout()
    fname = os.path.join(out_dir, f"top{k}_heatmap_L{layer}_F{feature}.pdf")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_dlm_sae"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_dlm_sae", "dictionary_learning"))

    layer_to_features = resolve_pairs(args.pairs)
    model, tokenizer = load_model_and_tokenizer(args)
    mask_token_id = tokenizer.mask_token_id
    model_device = str(next(model.parameters()).device)

    # Process each layer separately (each has its own SAE)
    for layer, features in layer_to_features.items():
        print(f"\n{'='*60}")
        print(f"Layer {layer} | Features: {features}")
        print(f"{'='*60}")

        sae = load_sae(layer, model_device)

        for feature in features:
            # Get texts for this feature
            if args.interp_dir:
                texts = get_top_texts_for_feature(args.interp_dir, layer, feature, args.alg, args.top_k)
                if not texts:
                    continue
            elif args.text:
                texts = [args.text]
            else:
                print("Provide --interp_dir or --text")
                return

            # Collect trajectory for each text
            trajs = []
            for i, text in enumerate(texts):
                print(f"\nL{layer}-F{feature} [text {i+1}/{len(texts)}]: {text[:60]}...")
                traj = collect_trajectory(
                    text, model, tokenizer, sae, layer,
                    [feature], mask_token_id, args,
                )
                if traj is not None:
                    trajs.append(traj)

            if not trajs:
                print(f"  No valid trajectories for L{layer}-F{feature}")
                continue

            # Plot top-k combined plots
            plot_top_k_heatmaps(trajs, layer, feature, args.out_dir)
            plot_top_k_commit_order(trajs, layer, feature, args.out_dir)

    print(f"\nDone! All plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
