import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_ROOT, "train_dlm_sae"))
sys.path.append(os.path.join(REPO_ROOT, "train_dlm_sae", "eval_sae"))

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from eval_delta_dlm_loss import register_sae_splice_hook


MODEL_NAME = "Dream-org/Dream-v0-Base-7B"
SAE_REPO = "AwesomeInterpretability/dlm-mask-topk-sae"

# Start with one layer for a smoke test.
LAYERS = [1, 5, 10, 14, 23, 27]
TRAINER = 0

BATCH_SIZE = 1
MAX_LENGTH = 128
NUM_TEXTS = 2

OUTPUT_DIR = "outputs_smoke"


def resolve_layers_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find a transformer layer container.")


def extract_hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported output type: {type(output)}")


def find_text_field(example):
    for key in ["text", "content", "raw_content", "contents"]:
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key]
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return str(example)


def load_dream():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_sae(layer, device):
    ae_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=(
            f"saes_mask_Dream-org_Dream-v0-Base-7B_top_k/"
            f"resid_post_layer_{layer}/trainer_{TRAINER}/ae.pt"
        ),
    )
    sae = AutoEncoderTopK.from_pretrained(ae_path, device=str(device))
    sae.eval()
    return sae


class RunningAverageStore:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, key, feats):
        if feats.ndim == 3:
            reduce_dims = (0, 1)
            count = feats.shape[0] * feats.shape[1]
        elif feats.ndim == 2:
            reduce_dims = (0,)
            count = feats.shape[0]
        else:
            raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")

        feats_sum = feats.sum(dim=reduce_dims).detach().cpu()

        if key not in self.sums:
            self.sums[key] = feats_sum
            self.counts[key] = count
        else:
            self.sums[key] += feats_sum
            self.counts[key] += count

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        metadata = {}

        for key in self.sums:
            mean_tensor = self.sums[key] / max(self.counts[key], 1)
            safe_key = key.replace("/", "_")
            save_path = os.path.join(out_dir, f"{safe_key}.pt")
            torch.save(mean_tensor, save_path)
            metadata[key] = {
                "count": int(self.counts[key]),
                "shape": list(mean_tensor.shape),
                "path": save_path,
            }

        torch.save(metadata, os.path.join(out_dir, "metadata.pt"))
        return metadata


def main():
    print("Loading Dream model...")
    model, tokenizer = load_dream()
    print("Dream model loaded.")

    layers_container = resolve_layers_container(model)
    model_device = next(
        (p.device for p in model.parameters() if p.requires_grad),
        torch.device("cpu"),
    )

    print(f"Using device: {model_device}")
    print(f"Target layers: {LAYERS}")

    print("Loading SAE checkpoints...")
    saes = {layer: load_sae(layer, model_device) for layer in LAYERS}
    print("SAE checkpoints loaded.")

    store = RunningAverageStore()
    splice_handles = []
    stats_handles = []

    # This hook records encoded SAE features without modifying the model output.
    def make_stats_hook(layer_idx, sae_module):
        def hook_fn(module, inputs, output):
            with torch.no_grad():
                hidden = extract_hidden_tensor(output)
                hidden = hidden.to(dtype=sae_module.encoder.weight.dtype)
                feats = sae_module.encode(hidden)
                store.update(f"layer_{layer_idx}/encoded_features", feats)
            return None
        return hook_fn

    for layer in LAYERS:
        submodule = layers_container[layer]

        # This is the actual SAE splice into the model.
        splice_handle = register_sae_splice_hook(
            submodule=submodule,
            dictionary=saes[layer],
            io="out",
        )
        splice_handles.append(splice_handle)

        # This separate hook records feature activations.
        stats_handle = submodule.register_forward_hook(
            make_stats_hook(layer, saes[layer])
        )
        stats_handles.append(stats_handle)

    print("Loading CommonPile stream...")
    dataset = load_dataset(
        "common-pile/comma_v0.1_training_dataset",
        split="train",
        streaming=True,
    )

    texts = []
    for example in dataset:
        text = find_text_field(example)
        if text and text.strip():
            texts.append(text)
        if len(texts) >= NUM_TEXTS:
            break

    print(f"Collected {len(texts)} text samples.")

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        tokens = tokenizer(
    		batch_texts,
    		return_tensors="pt",
    		padding=True,
    		truncation=True,
    		max_length=MAX_LENGTH,
	)
        tokens = {k: v.to(model_device) for k, v in tokens.items()}

	# Dream's attention expects a bool or float attention mask for SDPA.
        if "attention_mask" in tokens:
    	    tokens["attention_mask"] = tokens["attention_mask"].bool()

        with torch.no_grad():
            _ = model(**tokens)

        print(f"Processed batch {start // BATCH_SIZE + 1}")

    metadata = store.save(OUTPUT_DIR)
    print("Saved outputs:")
    for key, value in metadata.items():
        print(key, value)

    for handle in splice_handles:
        handle.remove()
    for handle in stats_handles:
        handle.remove()

    print("Done.")


if __name__ == "__main__":
    main()
