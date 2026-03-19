from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_ROOT, "train_dlm_sae"))
sys.path.append(os.path.join(REPO_ROOT, "train_dlm_sae", "dictionary_learning"))

from dictionary_learning.trainers.top_k import AutoEncoderTopK


DEFAULT_MODEL_NAME = "Dream-org/Dream-v0-Base-7B"
DEFAULT_SAE_REPO = "AwesomeInterpretability/dlm-mask-topk-sae"
DEFAULT_DATASET = "common-pile/comma_v0.1_training_dataset"
DEFAULT_LAYERS = [5, 14, 27]


@dataclass
class SequenceHistory:
    """Stores the prompt length and the per-diffusion-step token sequences."""
    input_len: int
    step_seqs: List[List[int]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Dream SAE feature statistics over diffusion timesteps."
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sae_repo", type=str, default=DEFAULT_SAE_REPO)
    parser.add_argument(
        "--sae_model_id",
        type=str,
        default="Dream-org_Dream-v0-Base-7B",
        help="Path prefix used inside the SAE HF repo.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layer indices to attach SAE hooks to.",
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=0,
        help="Trainer index inside the SAE repo path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Expected SAE sparsity level K. Verified against the loaded checkpoint.",
    )
    parser.add_argument(
        "--top_m",
        type=int,
        required=True,
        help="Top-M token activations averaged for statistic 2.",
    )
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--num_texts",
        type=int,
        default=16,
        help="Number of CommonPile examples to process (use ~39000 for 5M tokens at length 128).",
    )
    parser.add_argument(
        "--prompt_max_len",
        type=int,
        default=128,
        help="Maximum prompt token length fed to Dream before generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Number of tokens Dream generates per example.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of diffusion timesteps (T dimension in the output tensor).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--alg",
        type=str,
        default="entropy",
        choices=["entropy", "origin", "topk_margin"],
        help="Dream decoder strategy. Experiment 1 uses entropy.",
    )
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load Dream in 4-bit quantized mode to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--token_scope",
        type=str,
        default="generated",
        choices=["generated", "full"],
        help=(
            "'generated': compute statistics only over newly generated tokens. "
            "'full': include prompt tokens as well."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_exp1_stats",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_layers_container(model: torch.nn.Module):
    """Return the list-like container that holds transformer decoder layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find a transformer layer container in the model.")


def extract_hidden_tensor(output: Any) -> torch.Tensor:
    """Pull the hidden-state tensor out of whatever a layer returns."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        if "hidden_states" in output and isinstance(output["hidden_states"], torch.Tensor):
            return output["hidden_states"]
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported module output type: {type(output)}")


def find_text_field(example: Dict[str, Any]) -> str:
    """Return the first non-empty string field from a dataset example."""
    for key in ["text", "content", "raw_content", "contents"]:
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key]
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return str(example)


def stream_common_pile_texts(dataset_name: str, split: str, num_texts: int) -> Iterable[str]:
    """Lazily yield up to num_texts non-empty text strings from CommonPile."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    yielded = 0
    for example in dataset:
        text = find_text_field(example)
        if not text.strip():
            continue
        yield text
        yielded += 1
        if yielded >= num_texts:
            return


# ---------------------------------------------------------------------------
# Model / SAE loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any]:
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=get_torch_dtype(args.dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            torch_dtype=get_torch_dtype(args.dtype),
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=get_torch_dtype(args.dtype),
            trust_remote_code=True,
        ).to(args.device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_sae(
    sae_repo: str,
    sae_model_id: str,
    layer: int,
    trainer: int,
    expected_k: int,
    device: torch.device,
) -> AutoEncoderTopK:
    """Download and verify an SAE checkpoint for a single layer."""
    ae_path = hf_hub_download(
        repo_id=sae_repo,
        filename=(
            f"saes_mask_{sae_model_id}_top_k/"
            f"resid_post_layer_{layer}/trainer_{trainer}/ae.pt"
        ),
    )
    sae = AutoEncoderTopK.from_pretrained(ae_path, device=str(device))
    loaded_k = int(sae.k.item()) if isinstance(sae.k, torch.Tensor) else int(sae.k)
    if loaded_k != expected_k:
        raise ValueError(
            f"SAE at layer {layer} has k={loaded_k}, but --k={expected_k}. "
            "Use the correct trainer index or k value."
        )
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Diffusion history helpers
# ---------------------------------------------------------------------------

def extract_history(output: Any) -> Optional[Any]:
    """Return the per-step history object from Dream's generate output."""
    if hasattr(output, "history"):
        return output.history
    if hasattr(output, "sequences_history"):
        return output.sequences_history
    if isinstance(output, dict) and "history" in output:
        return output["history"]
    return None


def normalize_step_seqs(history: Any) -> List[List[int]]:
    """Convert Dream's history (various formats) into a list of token-id lists, one per step."""
    step_seqs: List[List[int]] = []
    if history is None:
        return step_seqs

    if isinstance(history, (list, tuple)):
        for item in history:
            arr = item.detach().cpu() if isinstance(item, torch.Tensor) else torch.tensor(item)
            if arr.dim() == 2:
                step_seqs.append(arr[0].tolist())
            elif arr.dim() == 1:
                step_seqs.append(arr.tolist())
            else:
                raise ValueError(f"Unexpected history item shape: {tuple(arr.shape)}")
        return step_seqs

    if isinstance(history, torch.Tensor):
        arr = history.detach().cpu()
        if arr.dim() == 3:       # [T, B, S]
            for t in range(arr.shape[0]):
                step_seqs.append(arr[t, 0].tolist())
        elif arr.dim() == 2:     # [T, S]
            for t in range(arr.shape[0]):
                step_seqs.append(arr[t].tolist())
        else:
            raise ValueError(f"Unexpected history tensor shape: {tuple(arr.shape)}")
        return step_seqs

    raise ValueError(f"Unsupported history type: {type(history)}")


def tokenize_prompt(
    tokenizer,
    text: str,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device).bool()
    return input_ids, attention_mask


def generate_history(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
) -> SequenceHistory:
    """
    Run Dream's diffusion_generate with output_history=True.
    Returns the prompt length and all per-step token sequences.
    """
    device = get_model_device(model)
    input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, args.prompt_max_len, device)
    input_len = int(input_ids.shape[1])

    with torch.inference_mode():
        output = model.diffusion_generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            steps=int(args.steps),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            alg=str(args.alg),
            alg_temp=float(args.alg_temp),
            output_history=True,
            return_dict_in_generate=True,
            do_sample=bool(args.do_sample),
        )

    history = extract_history(output)
    step_seqs = normalize_step_seqs(history)
    if not step_seqs:
        raise RuntimeError(
            "Dream returned no diffusion history. "
            "Ensure output_history=True is supported by the installed Dream version."
        )

    return SequenceHistory(input_len=input_len, step_seqs=step_seqs)


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats_for_batch(
    feats: torch.Tensor,
    attention_mask: torch.Tensor,
    top_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three per-sequence statistics over the token dimension.

    Args:
        feats:           [B, S, F]  SAE feature activations
        attention_mask:  [B, S]     bool mask (True = valid token)
        top_m:           int        number of top tokens for statistic 2

    Returns (each shape [B, F]):
        stat1 - mean activation across all valid tokens
        stat2 - mean activation across the top-M tokens by value
        stat3 - fraction of valid tokens where the feature is active (> 0)
    """
    if feats.ndim != 3:
        raise ValueError(f"Expected feats shape [B, S, F], got {tuple(feats.shape)}")

    mask = attention_mask.bool()                                   # [B, S]
    valid_counts = mask.sum(dim=1).clamp(min=1).to(feats.dtype)   # [B]

    # Stat 1: average over all valid tokens
    masked_feats = feats * mask.unsqueeze(-1)
    stat1 = masked_feats.sum(dim=1) / valid_counts.unsqueeze(-1)  # [B, F]

    # Stat 2: average over the top-M valid tokens per feature
    feats_bfs = feats.transpose(1, 2)                             # [B, F, S]
    mask_bfs = mask.unsqueeze(1).expand_as(feats_bfs)
    neg_inf = torch.full_like(feats_bfs, float("-inf"))
    feats_for_topk = torch.where(mask_bfs, feats_bfs, neg_inf)
    k = min(int(top_m), feats_for_topk.shape[-1])
    top_vals = feats_for_topk.topk(k=k, dim=-1).values            # [B, F, k]
    finite_mask = torch.isfinite(top_vals)
    top_vals = torch.where(finite_mask, top_vals, torch.zeros_like(top_vals))
    denom_top = finite_mask.sum(dim=-1).clamp(min=1).to(feats.dtype)
    stat2 = top_vals.sum(dim=-1) / denom_top                      # [B, F]

    # Stat 3: activation frequency (fraction of tokens where feature > 0)
    stat3 = ((feats > 0) & mask.unsqueeze(-1)).float().sum(dim=1) / valid_counts.unsqueeze(-1)

    return stat1, stat2, stat3


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class LayerStatsAccumulator:
    """
    Accumulates the three statistics across all text sequences and diffusion timesteps.

    Internal buffers have shape [F, T] per layer per statistic.
    Finalize() returns a tensor of shape [L, 3, F, T].
    """

    def __init__(self, layers: List[int], feature_dim: int, num_steps: int, top_m: int):
        self.layers = list(layers)
        self.feature_dim = int(feature_dim)
        self.num_steps = int(num_steps)
        self.top_m = int(top_m)

        # State set before each forward pass by replay_history_and_collect
        self.current_timestep: Optional[int] = None
        self.current_attention_mask: Optional[torch.Tensor] = None
        self.current_region_start: int = 0
        self.current_region_end: Optional[int] = None

        # Running sums: {layer: [F, T]}
        self.sum_stat1: Dict[int, torch.Tensor] = {
            l: torch.zeros(feature_dim, num_steps, dtype=torch.float32) for l in self.layers
        }
        self.sum_stat2: Dict[int, torch.Tensor] = {
            l: torch.zeros(feature_dim, num_steps, dtype=torch.float32) for l in self.layers
        }
        self.sum_stat3: Dict[int, torch.Tensor] = {
            l: torch.zeros(feature_dim, num_steps, dtype=torch.float32) for l in self.layers
        }
        # Number of sequences that contributed to each timestep column
        self.seq_count_by_timestep = torch.zeros(num_steps, dtype=torch.long)

    def start_timestep(
        self,
        timestep: int,
        attention_mask: torch.Tensor,
        region_start: int,
        region_end: Optional[int] = None,
    ) -> None:
        """Called once per (text, timestep) pair before the forward pass."""
        self.current_timestep = int(timestep)
        self.current_attention_mask = attention_mask
        self.current_region_start = int(region_start)
        self.current_region_end = None if region_end is None else int(region_end)

    def update(self, layer: int, feats: torch.Tensor) -> None:
        """
        Called inside the forward hook with raw SAE features [B, S, F].
        Slices to the requested token region and accumulates statistics.
        """
        if self.current_timestep is None or self.current_attention_mask is None:
            raise RuntimeError("start_timestep() must be called before the forward pass.")

        start = self.current_region_start
        end = self.current_region_end if self.current_region_end is not None else feats.shape[1]
        if end <= start:
            return

        feats_region = feats[:, start:end, :].float()
        mask_region = self.current_attention_mask[:, start:end].bool()

        # Skip if no valid tokens in the region (e.g. very short prompt)
        if mask_region.sum().item() == 0:
            return

        s1, s2, s3 = compute_stats_for_batch(feats_region, mask_region, self.top_m)

        t = self.current_timestep
        # Sum over the batch dimension; finalize() divides by seq count later
        self.sum_stat1[layer][:, t] += s1.sum(dim=0).cpu()
        self.sum_stat2[layer][:, t] += s2.sum(dim=0).cpu()
        self.sum_stat3[layer][:, t] += s3.sum(dim=0).cpu()
        self.seq_count_by_timestep[t] += feats_region.shape[0]

    def finalize(self) -> torch.Tensor:
        """
        Divide running sums by sequence counts to get per-timestep means.
        Returns a tensor of shape [L, 3, F, T].
        """
        num_layers = len(self.layers)
        out = torch.zeros(num_layers, 3, self.feature_dim, self.num_steps, dtype=torch.float32)
        counts = self.seq_count_by_timestep.clamp(min=1).float()  # [T]

        for i, layer in enumerate(self.layers):
            out[i, 0] = self.sum_stat1[layer] / counts.unsqueeze(0)
            out[i, 1] = self.sum_stat2[layer] / counts.unsqueeze(0)
            out[i, 2] = self.sum_stat3[layer] / counts.unsqueeze(0)

        return out


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_stats_hooks(
    model: torch.nn.Module,
    saes: Dict[int, AutoEncoderTopK],
    accumulator: LayerStatsAccumulator,
) -> List[Any]:
    """
    Attach one forward hook per layer.

    The hook encodes the hidden state with the SAE and records the feature
    activations into the accumulator, then returns the original output
    unchanged so downstream layers are not affected.
    """
    handles = []
    layers_container = resolve_layers_container(model)

    def make_hook(layer_idx: int, sae: AutoEncoderTopK):
        def hook_fn(module, inputs, output):
            with torch.no_grad():
                hidden = extract_hidden_tensor(output)
                hidden = hidden.to(dtype=sae.encoder.weight.dtype)
                feats = sae.encode(hidden)
                accumulator.update(layer_idx, feats)
            # Return the original output to avoid any downstream side effects
            return output
        return hook_fn

    for layer in saes:
        handle = layers_container[layer].register_forward_hook(make_hook(layer, saes[layer]))
        handles.append(handle)

    return handles


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def replay_history_and_collect(
    model: torch.nn.Module,
    history: SequenceHistory,
    accumulator: LayerStatsAccumulator,
    args: argparse.Namespace,
) -> None:
    """
    Re-run the model once per diffusion timestep using the token sequences
    stored in history. The forward hooks collect SAE features at each step.

    token_scope:
        'generated' - statistics computed only over the newly generated tokens
        'full'      - statistics computed over the entire sequence
    """
    device = get_model_device(model)
    max_steps = min(args.steps, len(history.step_seqs))

    for timestep in range(max_steps):
        step_ids = history.step_seqs[timestep]
        input_ids = torch.tensor(step_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]

        # All positions are valid in the replayed sequence (no padding added here)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        region_start = history.input_len if args.token_scope == "generated" else 0
        region_end = input_ids.shape[1]

        accumulator.start_timestep(
            timestep=timestep,
            attention_mask=attention_mask,
            region_start=region_start,
            region_end=region_end,
        )

        with torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(
    out_dir: str,
    args: argparse.Namespace,
    layers: List[int],
    tensor: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Save the [L, 3, F, T] tensor plus metadata to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    # Raw tensor for fast loading
    torch.save(tensor, os.path.join(out_dir, "layer_stats_Lx3xFxT.pt"))

    # Full bundle with config for reproducibility
    bundle = {
        "tensor": tensor,
        "layers": layers,
        "stats_order": [
            "all_token_average",
            f"top_{args.top_m}_token_average",
            "activation_frequency",
        ],
        "count_by_timestep": counts,
        "config": vars(args),
    }
    torch.save(bundle, os.path.join(out_dir, "layer_stats_bundle.pt"))

    # Human-readable metadata
    metadata = {
        "layers": layers,
        "tensor_shape": list(tensor.shape),
        "stats_order": [
            "all_token_average",
            f"top_{args.top_m}_token_average",
            "activation_frequency",
        ],
        "count_by_timestep": counts.tolist(),
        "config": vars(args),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Outputs saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.alg != "entropy":
        print(f"[Warning] Experiment 1 expects alg='entropy', but got alg='{args.alg}'.")

    print("Loading Dream model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args)
    model_device = get_model_device(model)
    print(f"Model device: {model_device}")

    print("Loading SAE checkpoints...")
    saes: Dict[int, AutoEncoderTopK] = {}
    for layer in args.layers:
        saes[layer] = load_sae(
            sae_repo=args.sae_repo,
            sae_model_id=args.sae_model_id,
            layer=layer,
            trainer=args.trainer,
            expected_k=args.k,
            device=model_device,
        )
    print(f"Loaded SAEs for layers: {args.layers}")

    feature_dim = next(iter(saes.values())).dict_size
    accumulator = LayerStatsAccumulator(
        layers=args.layers,
        feature_dim=feature_dim,
        num_steps=args.steps,
        top_m=args.top_m,
    )

    processed = 0
    for text in stream_common_pile_texts(args.dataset_name, args.dataset_split, args.num_texts):
        # Generate diffusion history first with NO hooks attached,
        # so the internal forward passes inside diffusion_generate are not intercepted.
        history = generate_history(model, tokenizer, text, args)

        # Attach hooks only for the replay loop where start_timestep() is called.
        handles = register_stats_hooks(model, saes, accumulator)
        try:
            replay_history_and_collect(model, history, accumulator, args)
        finally:
            # Always remove hooks after replay to keep the model clean.
            for handle in handles:
                handle.remove()

        processed += 1
        print(
            f"[{processed}/{args.num_texts}] "
            f"history steps: {len(history.step_seqs)} | "
            f"prompt len: {history.input_len}"
        )

    if processed == 0:
        raise RuntimeError("No CommonPile texts were processed. Check dataset config.")

    final_tensor = accumulator.finalize()
    save_outputs(
        out_dir=args.out_dir,
        args=args,
        layers=args.layers,
        tensor=final_tensor,
        counts=accumulator.seq_count_by_timestep,
    )

    print("Done.")
    print(f"Processed texts : {processed}")
    print(f"Final tensor shape : {tuple(final_tensor.shape)}  (L x 3 x F x T)")
    print(f"Outputs written to : {args.out_dir}")


if __name__ == "__main__":
    main()
