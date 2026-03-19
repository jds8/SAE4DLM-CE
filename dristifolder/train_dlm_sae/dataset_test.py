from tqdm import tqdm
from typing import Callable
from transformers import AutoTokenizer
from dictionary_learning.dictionary_learning.utils import (
    hf_mixed_dataset_to_generator,
)


def exhaustiveness_test(
    make_gen: Callable[[], iter],  # e.g. lambda: hf_mixed_dataset_to_generator(tok, …)
    tokenizer: AutoTokenizer,
    target_tokens: int = 100_000_000,  # how many you need for the whole run
    ctx_len: int = 1024,  # max_length you’ll use in training
    report_every: int = 10_000,  # print progress every N samples
) -> None:
    """
    Streams until `target_tokens` are seen or the generator ends early.
    Raises RuntimeError if the generator exhausts too soon.
    """
    gen = make_gen()
    seen_tokens = 0
    seen_samples = 0

    while seen_tokens < target_tokens:
        try:
            sample = next(gen)
        except StopIteration:
            raise RuntimeError(
                f"Generator exhausted after {seen_tokens:,} tokens; "
                f"needed {target_tokens:,}."
            )

        # Fast tokenizer call; no attention mask needed for the count.
        ids = tokenizer(
            sample,
            truncation=True,
            max_length=ctx_len,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        seen_tokens += len(ids)
        seen_samples += 1

        if seen_samples % report_every == 0:
            print(f"{seen_samples:,} samples – {seen_tokens:,} tokens")

    print(
        f"✅ Success: generated {seen_tokens:,} tokens "
        f"across {seen_samples:,} samples without exhaustion."
    )


# ------------------------------------------------------------
# Example use
# ------------------------------------------------------------
if __name__ == "__main__":
    # Useful test to make sure you don't run out of tokens
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    # Build a *factory* so the test can restart the stream cleanly
    gen_factory = lambda: hf_mixed_dataset_to_generator(
        tokenizer=tok,
        pretrain_frac=0.9,
        sequence_pack_pretrain=True,
        sequence_pack_chat=False,
        system_prompt_to_remove=None,
    )

    exhaustiveness_test(
        make_gen=gen_factory,
        tokenizer=tok,
        target_tokens=500_000_000,  # 100 M tokens
        ctx_len=2048,
        report_every=5_000,
    )
