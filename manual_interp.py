"""
manual_interp.py

Runs text sequences through Dream + SAE for one or more (layer, feature) pairs.
Texts are loaded ONCE and reused across all features in the same layer.

Output per feature:
  - results_L{layer}_F{feature}_{alg}.txt   (human readable)
  - results_L{layer}_F{feature}_{alg}.json  (LLM input ready)

Usage:
    # Single feature
    python manual_interp.py --layer 23 --features 7116

    # Multiple features, same layer (SAE loaded once)
    python manual_interp.py --layer 23 --features 7116 1311 6193

    # Multiple layers (use --pairs layer:feat layer:feat ...)
    python manual_interp.py --pairs 23:7116 23:1311 10:12343 10:3806 1:2788

    # Justice's approximate features
    python manual_interp.py --pairs 1:4000 1:5900 1:9800 1:14200 5:4000 5:5900 10:6900 10:11800 23:1900 23:4300 23:12800
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME   = "Dream-org/Dream-v0-Base-7B"
SAE_REPO     = "AwesomeInterpretability/dlm-mask-topk-sae"
SAE_MODEL_ID = "Dream-org_Dream-v0-Base-7B"
TRAINER      = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual feature interpretation for DLM SAE.")

    # Option A: single layer, multiple features
    parser.add_argument("--layer",    type=int, default=None, help="Layer index")
    parser.add_argument("--features", type=int, nargs="+", default=None, help="One or more feature indices")

    # Option B: arbitrary (layer, feature) pairs
    parser.add_argument(
        "--pairs", type=str, nargs="+", default=None,
        help="layer:feature pairs e.g. 23:7116 10:12343",
    )

    parser.add_argument("--num_texts",      type=int, default=100)
    parser.add_argument("--prompt_max_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps",          type=int, default=64)
    parser.add_argument("--top_k_texts",    type=int, default=20)
    parser.add_argument("--top_k_tokens",   type=int, default=10)
    parser.add_argument("--alg",  type=str, default="entropy", choices=["entropy", "origin"])
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir",  type=str, default="interp_results")
    parser.add_argument("--dataset",  type=str, default="common-pile/comma_v0.1_training_dataset")
    return parser.parse_args()


def resolve_pairs(args) -> Dict[int, List[int]]:
    """Returns {layer: [feature, ...]} dict."""
    layer_to_features: Dict[int, List[int]] = defaultdict(list)

    if args.pairs:
        for p in args.pairs:
            layer_str, feat_str = p.split(":")
            layer_to_features[int(layer_str)].append(int(feat_str))

    if args.layer is not None and args.features:
        for f in args.features:
            layer_to_features[args.layer].append(f)

    if not layer_to_features:
        raise ValueError("Provide --layer + --features or --pairs.")

    return dict(layer_to_features)


# ---------------------------------------------------------------------------
# MODEL / SAE
# ---------------------------------------------------------------------------

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


def load_sae(layer: int, device: str):
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


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def stream_texts(dataset_name: str, num_texts: int) -> List[str]:
    ds = load_dataset(dataset_name, split="train", streaming=True)
    texts = []
    for ex in ds:
        for key in ["text", "content", "raw_content"]:
            if key in ex and isinstance(ex[key], str) and ex[key].strip():
                texts.append(ex[key])
                break
        if len(texts) >= num_texts:
            break
    return texts


# ---------------------------------------------------------------------------
# ACTIVATION COLLECTION (multi-feature per forward pass)
# ---------------------------------------------------------------------------

@dataclass
class TextResult:
    text: str
    feature: int
    max_activation: float
    mean_activation: float
    timestep_data: Dict = field(default_factory=dict)


def collect_all_features_for_text(
    text: str,
    model,
    tokenizer,
    sae,
    layers_container,
    layer: int,
    features: List[int],
    mask_token_id: int,
    args,
) -> Dict[int, Optional[TextResult]]:
    """
    Single diffusion generate + replay for one text,
    collecting activations for ALL requested features simultaneously.
    Returns {feature: TextResult or None}.
    """
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

    handle = layers_container[layer].register_forward_hook(hook_fn)

    # {feature: {timestep: {tokens, activations, is_masked}}}
    feature_timestep_data: Dict[int, Dict] = {f: {} for f in features}

    try:
        with torch.inference_mode():
            out = model.diffusion_generate(
                inputs=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=1.0,
                top_p=0.95,
                alg=args.alg,
                alg_temp=0.0,
                output_history=True,
                return_dict_in_generate=True,
                do_sample=True,
            )

        history = getattr(out, "history", None) or (out.get("history") if isinstance(out, dict) else None)
        if history is None:
            return {f: None for f in features}

        if isinstance(history, torch.Tensor):
            arr = history.detach().cpu()
            step_seqs = [arr[t, 0].tolist() if arr.dim() == 3 else arr[t].tolist()
                         for t in range(arr.shape[0])]
        else:
            step_seqs = []
            for item in history:
                a = item.detach().cpu() if isinstance(item, torch.Tensor) else torch.tensor(item)
                step_seqs.append(a[0].tolist() if a.dim() == 2 else a.tolist())

        for t_idx, step_ids in enumerate(step_seqs):
            ids = torch.tensor(step_ids, dtype=torch.long, device=device).unsqueeze(0)
            am  = torch.ones_like(ids, dtype=torch.bool)

            with torch.inference_mode():
                _ = model(input_ids=ids, attention_mask=am)

            hidden = captured.get("h")
            if hidden is None:
                continue

            hidden_cast = hidden.to(dtype=sae.encoder.weight.dtype)
            with torch.no_grad():
                feats_all = sae.encode(hidden_cast)  # [1, S, F_total]

            gen_ids = ids[0, input_len:].cpu().tolist()
            is_masked = [tok == mask_token_id for tok in gen_ids]

            for feat in features:
                gen_feats = feats_all[0, input_len:, feat].cpu().float().tolist()
                feature_timestep_data[feat][t_idx] = {
                    "tokens":      gen_ids,
                    "activations": gen_feats,
                    "is_masked":   is_masked,
                }

    finally:
        handle.remove()

    results = {}
    for feat in features:
        td = feature_timestep_data[feat]
        all_acts = [a for t_data in td.values() for a in t_data["activations"]]
        if not all_acts:
            results[feat] = None
        else:
            results[feat] = TextResult(
                text=text,
                feature=feat,
                max_activation=float(max(all_acts)),
                mean_activation=float(np.mean(all_acts)),
                timestep_data=td,
            )
    return results


# ---------------------------------------------------------------------------
# ANALYSIS + OUTPUT (unchanged from original)
# ---------------------------------------------------------------------------

def top_tokens_in_stage(results, stage_ts, masked, tokenizer, top_k):
    token_act = {}
    for tr in results:
        for t in stage_ts:
            if t not in tr.timestep_data:
                continue
            for tok_id, act, is_m in zip(
                tr.timestep_data[t]["tokens"],
                tr.timestep_data[t]["activations"],
                tr.timestep_data[t]["is_masked"],
            ):
                if is_m != masked:
                    continue
                tok_str = tokenizer.decode([tok_id])
                token_act.setdefault(tok_str, []).append(act)
    ranked = sorted(token_act.items(), key=lambda x: np.mean(x[1]), reverse=True)
    return [(tok, float(np.mean(acts)), len(acts)) for tok, acts in ranked[:top_k]]


def mean_activation_per_timestep(results, masked, steps):
    vals = []
    for t in range(steps):
        bucket = []
        for tr in results:
            if t not in tr.timestep_data:
                continue
            for act, is_m in zip(tr.timestep_data[t]["activations"], tr.timestep_data[t]["is_masked"]):
                if is_m == masked:
                    bucket.append(act)
        vals.append(float(np.mean(bucket)) if bucket else 0.0)
    return vals


def save_txt(path, args, feature, results, tokenizer):
    T = args.steps
    EARLY = list(range(0, T // 3))
    MID   = list(range(T // 3, 2 * T // 3))
    LATE  = list(range(2 * T // 3, T))
    masked_ts   = mean_activation_per_timestep(results, True,  T)
    unmasked_ts = mean_activation_per_timestep(results, False, T)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Manual Interpretation: Layer {args.layer or 'multi'} — Feature {feature}\n")
        f.write(f"Algorithm: {args.alg} | Texts: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        f.write("ACTIVATION OVER DIFFUSION TIMESTEPS\n")
        f.write(f"  Masked   — early: {np.mean(masked_ts[:T//3]):.4f}, late: {np.mean(masked_ts[2*T//3:]):.4f}\n")
        f.write(f"  Unmasked — early: {np.mean(unmasked_ts[:T//3]):.4f}, late: {np.mean(unmasked_ts[2*T//3:]):.4f}\n\n")
        f.write(f"TOP {args.top_k_texts} TEXTS BY MAX ACTIVATION\n")
        f.write("-" * 80 + "\n")
        for rank, tr in enumerate(results[:args.top_k_texts], 1):
            f.write(f"\n[{rank}] max={tr.max_activation:.4f}  mean={tr.mean_activation:.4f}\n")
            f.write(tr.text[:500].replace("\n", " ") + "\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP ACTIVATING TOKENS BY STAGE\n")
        for stage_name, stage_ts in [("EARLY", EARLY), ("MID", MID), ("LATE", LATE)]:
            f.write(f"\n{stage_name} (t={stage_ts[0]}–{stage_ts[-1]})\n")
            for label, masked in [("MASKED", True), ("UNMASKED", False)]:
                tops = top_tokens_in_stage(results, stage_ts, masked, tokenizer, args.top_k_tokens)
                f.write(f"  [{label}]\n")
                for i, (tok, mean_act, count) in enumerate(tops, 1):
                    f.write(f"    {i:2d}. {repr(tok):25s}  mean={mean_act:.4f}  count={count}\n")
    print(f"  Saved TXT: {path}")


def save_json(path, args, feature, layer, results, tokenizer):
    T = args.steps
    EARLY = list(range(0, T // 3))
    MID   = list(range(T // 3, 2 * T // 3))
    LATE  = list(range(2 * T // 3, T))
    masked_ts   = mean_activation_per_timestep(results, True,  T)
    unmasked_ts = mean_activation_per_timestep(results, False, T)

    top_texts = [
        {"rank": i+1, "max_activation": tr.max_activation,
         "mean_activation": tr.mean_activation, "text": tr.text[:1000]}
        for i, tr in enumerate(results[:args.top_k_texts])
    ]
    stage_tokens = {}
    for stage_name, stage_ts in [("early", EARLY), ("mid", MID), ("late", LATE)]:
        stage_tokens[stage_name] = {}
        for label, masked in [("masked", True), ("unmasked", False)]:
            tops = top_tokens_in_stage(results, stage_ts, masked, tokenizer, args.top_k_tokens)
            stage_tokens[stage_name][label] = [
                {"token": tok, "mean_activation": mean_act, "count": count}
                for tok, mean_act, count in tops
            ]

    output = {
        "meta": {"layer": layer, "feature": feature, "algorithm": args.alg, "num_texts": len(results)},
        "activation_summary": {
            "masked":   {"early_mean": float(np.mean(masked_ts[:T//3])),   "late_mean": float(np.mean(masked_ts[2*T//3:]))},
            "unmasked": {"early_mean": float(np.mean(unmasked_ts[:T//3])), "late_mean": float(np.mean(unmasked_ts[2*T//3:]))},
        },
        "top_texts": top_texts,
        "top_tokens_by_stage": stage_tokens,
        "llm_prompt": (
            f"I am analyzing SAE feature {feature} at layer {layer} of a diffusion language model. "
            f"This feature activates strongly on masked tokens early in diffusion and on unmasked tokens late. "
            f"Below are the top {args.top_k_texts} text sequences where this feature activates most strongly. "
            f"Please identify any semantic, syntactic, or thematic patterns these texts share. "
            f"What concept might this feature be detecting?"
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON: {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_dlm_sae"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_dlm_sae", "dictionary_learning"))

    layer_to_features = resolve_pairs(args)
    print(f"Plan: {dict(layer_to_features)}")

    model, tokenizer = load_model_and_tokenizer(args)
    mask_token_id = tokenizer.mask_token_id
    layers_container = get_layers(model)
    model_device = str(next(model.parameters()).device)

    print(f"Loading {args.num_texts} texts...")
    texts = stream_texts(args.dataset, args.num_texts)
    print(f"Loaded {len(texts)} texts\n")

    for layer, features in layer_to_features.items():
        print(f"{'='*60}")
        print(f"Layer {layer} | Features: {features}")
        print(f"{'='*60}")

        sae = load_sae(layer, model_device)

        # {feature: [TextResult]}
        all_results: Dict[int, List[TextResult]] = {f: [] for f in features}

        for i, text in enumerate(texts):
            per_feat = collect_all_features_for_text(
                text, model, tokenizer, sae, layers_container,
                layer, features, mask_token_id, args,
            )
            for feat, tr in per_feat.items():
                if tr is not None:
                    all_results[feat].append(tr)

            if (i + 1) % 10 == 0:
                counts = {f: len(all_results[f]) for f in features}
                print(f"  [{i+1}/{len(texts)}] collected: {counts}")

        # Save per feature
        for feat in features:
            results = sorted(all_results[feat], key=lambda x: x.max_activation, reverse=True)
            if not results:
                print(f"  L{layer}-F{feat}: no results, skipping")
                continue
            print(f"\nL{layer}-F{feat}: top max activation = {results[0].max_activation:.4f}")
            base = os.path.join(args.out_dir, f"results_L{layer}_F{feat}_{args.alg}")
            save_txt(base + ".txt", args, feat, results, tokenizer)
            save_json(base + ".json", args, feat, layer, results, tokenizer)

    print(f"\nAll done! Outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
