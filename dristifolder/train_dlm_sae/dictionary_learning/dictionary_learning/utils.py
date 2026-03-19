from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
from transformers import AutoModelForCausalLM
from fractions import Fraction
import random
from transformers import AutoTokenizer
import torch as t

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    """
    Create a Python generator over a Hugging Face dataset that yields plain text
    from the 'text' field. Streaming mode keeps memory usage small.
    """
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file and yield the 'text' field of each JSONL row.
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)["text"]

    return generator()


def randomly_remove_system_prompt(
    text: str, freq: float, system_prompt: str | None = None
) -> str:
    """
    Optionally remove a known system prompt string with probability `freq`.
    Useful to avoid overexposing a constant prompt in mixed chat data.
    """
    if system_prompt and random.random() < freq:
        assert system_prompt in text
        text = text.replace(system_prompt, "")
    return text


def hf_mixed_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    chat_dataset: str = "lmsys/lmsys-chat-1m",
    min_chars: int = 1,
    pretrain_frac: float = 0.9,  # 0.9 → 90% pretrain, 10% chat
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    chat_key: str = "conversation",
    sequence_pack_pretrain: bool = True,
    sequence_pack_chat: bool = False,
    system_prompt_to_remove: str | None = None,
    system_prompt_removal_freq: float = 0.9,
):
    """
    Yield a mixture of pretrain and chat data at a specified ratio.

    - `min_chars`: minimal character count to perform naive sequence packing.
    - By default, we pack pretrain data (joining with EOS) but *not* chat data,
      because chat packing can distort conversation boundaries unless intended.
    - For chat samples, we use `tokenizer.apply_chat_template(..., tokenize=False)`.

    Note: You may need to request access for certain datasets on the Hub.
    """
    if not 0 < pretrain_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    assert min_chars > 0

    # Load both datasets as iterable streams
    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    chat_ds = iter(load_dataset(chat_dataset, split=split, streaming=streaming))

    # Convert the fraction to two small integers (e.g., 0.9 → 9/10)
    frac = Fraction(pretrain_frac).limit_denominator()
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain

    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token if tokenizer.bos_token else eos_token

    def gen():
        while True:
            # Pretrain portion
            for _ in range(n_pretrain):
                if sequence_pack_pretrain:
                    length = 0
                    samples = []
                    while length < min_chars:
                        sample = next(pretrain_ds)[pretrain_key]
                        samples.append(sample)
                        length += len(sample)
                    samples = bos_token + eos_token.join(samples)
                    yield samples
                else:
                    sample = bos_token + next(pretrain_ds)[pretrain_key]
                    yield sample
            # Chat portion
            for _ in range(n_chat):
                if sequence_pack_chat:
                    length = 0
                    samples = []
                    while length < min_chars:
                        conv = next(chat_ds)[chat_key]
                        formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                        formatted = randomly_remove_system_prompt(
                            formatted, system_prompt_removal_freq, system_prompt_to_remove
                        )
                        samples.append(formatted)
                        length += len(formatted)
                    samples = "".join(samples)
                    yield samples
                else:
                    conv = next(chat_ds)[chat_key]
                    formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                    formatted = randomly_remove_system_prompt(
                        formatted, system_prompt_removal_freq, system_prompt_to_remove
                    )
                    yield formatted

    return gen()


def hf_sequence_packing_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "common-pile/comma_v0.1_training_dataset",
    min_chars: int = 1,
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    sequence_pack_pretrain: bool = True,
):
    """
    Simple generator that yields pretraining text, optionally sequence-packed by concatenating
    samples with EOS until `min_chars` is reached. We add BOS once at the beginning.
    """
    assert min_chars > 0

    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token if tokenizer.bos_token else eos_token

    def gen():
        while True:
            if sequence_pack_pretrain:
                length = 0
                samples = []
                while length < min_chars:
                    sample = next(pretrain_ds)[pretrain_key]
                    samples.append(sample)
                    length += len(sample)
                samples = bos_token + eos_token.join(samples)
                yield samples
            else:
                sample = bos_token + next(pretrain_ds)[pretrain_key]
                yield sample

    return gen()


def simple_hf_mixed_dataset_to_generator(
    main_name: str,
    aux_name: str,
    main_frac: float = 0.9,  # 0.9 → 90% main, 10% aux
    split: str = "train",
    streaming: bool = True,
    main_key: str = "text",
    aux_key: str = "text",
):
    """
    Alternate simple mixer that yields `main_frac` from a main dataset and the rest from an aux dataset.
    """
    if not 0 < main_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    main_ds = iter(load_dataset(main_name, split=split, streaming=streaming))
    aux_ds = iter(load_dataset(aux_name, split=split, streaming=streaming))

    frac = Fraction(main_frac).limit_denominator()
    n_main = frac.numerator
    n_aux = frac.denominator - n_main

    def gen():
        while True:
            for _ in range(n_main):
                yield next(main_ds)[main_key]
            for _ in range(n_aux):
                yield next(aux_ds)[aux_key]

    return gen()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively collect folders that contain an 'ae.pt' file, starting from `path`.
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


def load_dictionary(base_path: str, device: str) -> tuple:
    """
    Load a saved dictionary (SAE weights) and its JSON config, based on the recorded dict_class.
    """
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config


def _arch_name(model: AutoModelForCausalLM) -> str:
    """
    Convenience: return the first architecture string in the config, lowercased.
    """
    try:
        return str(model.config.architectures[0]).lower()
    except Exception:
        return ""


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """
    Return the residual-stream-level submodule (decoder block) at `layer` index.

    For Dream-7B (DLM), the modules are under `model.layers.*`, same as Qwen/Gemma.
    We therefore return `model.model.layers[layer]`.
    """
    arch = _arch_name(model)

    if arch == "gptneoxforcausallm":
        return model.gpt_neox.layers[layer]

    # Qwen2, Gemma2, Dream-7B (DLM) all expose `model.layers`
    if arch in ("qwen2forcausallm", "gemma2forcausallm", "dreamforcausallm") or "dream" in arch:
        return model.model.layers[layer]

    # Fall back with a helpful error message
    raise ValueError(
        f"Please add submodule mapping for model {model.name_or_path} (arch='{model.config.architectures}')."
    )


def truncate_model(model: AutoModelForCausalLM, layer: int):
    """
    Truncate the model to keep only up to and including the specified `layer`.
    This can save a lot of memory when we only need intermediate activations.

    Implementation mirrors tilde-research/activault approach:
    - Drop all higher layers.
    - Replace the LM head with Identity to avoid holding logits projection weights.

    Supported families:
    - Qwen2ForCausalLM / Gemma2ForCausalLM / DreamForCausalLM (decoder under `model.layers`, head `lm_head`)
    - GPTNeoXForCausalLM (decoder under `gpt_neox.layers`, head `embed_out`)
    """
    import gc

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Model parameters before truncation: {total_params_before:,}")

    arch = _arch_name(model)

    if arch in ("qwen2forcausallm", "gemma2forcausallm", "dreamforcausallm") or "dream" in arch:
        # Decoder layers live in model.model.layers
        removed_layers = model.model.layers[layer + 1 :]
        model.model.layers = model.model.layers[: layer + 1]
        del removed_layers

        # Replace lm_head with Identity to avoid holding output projection
        if hasattr(model, "lm_head"):
            del model.lm_head
            model.lm_head = t.nn.Identity()

    elif arch == "gptneoxforcausallm":
        removed_layers = model.gpt_neox.layers[layer + 1 :]
        model.gpt_neox.layers = model.gpt_neox.layers[: layer + 1]
        del removed_layers

        if hasattr(model, "embed_out"):
            del model.embed_out
            model.embed_out = t.nn.Identity()

    else:
        raise ValueError(
            f"Please add truncation mapping for model {model.name_or_path} (arch='{model.config.architectures}')."
        )

    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"Model parameters after truncation: {total_params_after:,}")

    gc.collect()
    t.cuda.empty_cache()

    return model
