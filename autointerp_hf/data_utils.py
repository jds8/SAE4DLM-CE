# autointerp_hf/data_utils.py
from typing import Tuple
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

@torch.no_grad()
def load_and_tokenize_dataset(
    dataset_name: str,
    context_length: int,
    total_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stream text from a HuggingFace dataset WITHOUT fully downloading
    all shards to local disk first.

    We iterate over examples from the dataset in streaming mode, tokenize
    each example independently, break the token sequence into fixed-length
    chunks of size `context_length`, and keep appending those chunks until
    we have collected at least `total_tokens` tokens worth of data.

    Returns:
        input_ids:      [num_seqs, context_length]   on `device`
        attention_mask: [num_seqs, context_length]   on `device`
    """

    # 1) Streaming iterator (no full local materialization)
    ds_iter = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    chunk_list = []
    seen_tokens = 0

    # Progress bar: total is the target number of tokens we want to collect.
    # We'll "update" it by context_length every time we add a new chunk.
    pbar = tqdm(
        total=total_tokens,
        desc="Streaming+tokenizing dataset",
        unit="tok",
        dynamic_ncols=True,
    )

    for row in ds_iter:
        # Get raw text
        text = row.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            continue

        # Tokenize this single document
        toks = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids_long = toks["input_ids"][0]  # shape [T_total]
        T = input_ids_long.shape[0]

        if T < context_length:
            # Not enough tokens to make even one full chunk
            continue

        # Slice this doc into contiguous fixed-length chunks
        for start in range(0, T, context_length):
            end = start + context_length
            if end > T:
                break  # drop the short tail

            seg_ids = input_ids_long[start:end]  # [context_length]
            seg_mask = torch.ones_like(seg_ids, dtype=torch.long)

            chunk_list.append((seg_ids, seg_mask))

            seen_tokens += context_length
            pbar.update(context_length)  # NEW: show progress

            if seen_tokens >= total_tokens:
                break

        if seen_tokens >= total_tokens:
            break

    pbar.close()

    if len(chunk_list) == 0:
        raise RuntimeError(
            "load_and_tokenize_dataset(streaming=True): collected 0 chunks. "
            "Dataset may be empty or network is too slow."
        )

    # Stack into [num_seqs, context_length]
    input_id_chunks = torch.stack([c[0] for c in chunk_list], dim=0)      # [N, L]
    attn_mask_chunks = torch.stack([c[1] for c in chunk_list], dim=0)     # [N, L]

    # Move to device
    input_id_chunks = input_id_chunks.to(device)
    attn_mask_chunks = attn_mask_chunks.to(device)

    return input_id_chunks, attn_mask_chunks
