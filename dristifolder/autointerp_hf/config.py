# autointerp_hf/config.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AutoInterpEvalConfig:
    """
    Configuration controlling the end-to-end auto-interpretability pipeline.

    This is inspired by the original AutoInterpEvalConfig from saebench, but modified
    for a pure-HuggingFace / custom-hook setup.

    Key responsibilities:
    - controls which latents we try to interpret (n_latents / override_latents)
    - controls how we sample examples for generation & scoring
    - controls prompt sizes for the LLM judge
    - controls which model layer/module we hook

    USER MUST ensure that `hook_module_path` matches where the SAE was trained.
    For example, for a Qwen2.5 block you might use "model.layers.20".
    """

    # -------------------------
    # Model / SAE / hooking config
    # -------------------------
    model_name_or_path: str           # e.g. "Qwen/Qwen2.5-7B"
    hook_module_path: str             # e.g. "model.layers.20"
    device: str = "cuda"              # e.g. "cuda" or "cpu"
    torch_dtype: str = "auto"         # "float32","bfloat16","float16","auto"

    # -------------------------
    # Latent selection
    # -------------------------
    n_latents: Optional[int] = 100     # how many latents to evaluate
    override_latents: Optional[List[int]] = None
    dead_latent_threshold: float = 15  # minimum activation count for a latent to be considered "alive"
    random_seed: int = 42

    # -------------------------
    # Dataset / tokenization
    # -------------------------
    dataset_name: str = "common-pile/comma_v0.1_training_dataset"
    total_tokens: int = 2_000_000          # total tokens to gather from dataset
    llm_context_size: int = 128            # context length for each row in the tokenized dataset
    batch_size: Optional[int] = None       # batch size for collecting activations; user can override
    # NOTE: batch_size can be auto-resolved by you at runtime if None.

    # -------------------------
    # Activation processing
    # -------------------------
    buffer: int = 10                       # number of tokens to the left/right of each "center" token
    no_overlap: bool = True                # avoid overlapping windows for top-k picks
    act_threshold_frac: float = 0.01       # fraction * max_activation used to mark which tokens are "active"

    # -------------------------
    # Prompt generation / scoring toggles
    # -------------------------
    scoring: bool = True
    max_tokens_in_explanation: int = 30    # max new tokens judge can generate for explanation
    use_demos_in_explanation: bool = True  # include demo explanations in the system prompt

    # -------------------------
    # Sampling strategy sizes
    # -------------------------
    n_top_ex_for_generation: int = 10      # top activations used for explanation prompt
    n_iw_sampled_ex_for_generation: int = 5  # importance-weighted "medium" activations for explanation
    n_top_ex_for_scoring: int = 2          # leftover top acts for scoring
    n_random_ex_for_scoring: int = 10      # random windows for negative examples
    n_iw_sampled_ex_for_scoring: int = 2   # leftover medium activations for scoring

    # internal field, set in __post_init__
    latents: Optional[List[int]] = field(default=None, init=False)

    def __post_init__(self):
        """
        Resolve which latents we plan to interpret:
        - If override_latents is provided, use exactly those (and set n_latents accordingly).
        - Otherwise, n_latents stays as-is, we will later randomly sample that many alive latents.
        """
        if self.n_latents is None:
            assert self.override_latents is not None, (
                "If n_latents is None you must supply override_latents."
            )
            self.latents = self.override_latents
            self.n_latents = len(self.latents)
        else:
            assert self.override_latents is None, (
                "You cannot set both n_latents and override_latents."
            )
            self.latents = None

    # ------------- convenience properties -------------

    @property
    def n_top_ex(self) -> int:
        """
        Total number of top examples we will pull per-latent, before splitting them
        across 'generation' vs 'scoring'.
        """
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def n_ex_for_generation(self) -> int:
        """
        How many examples we feed to the judge in the explanation (generation) phase.
        """
        return self.n_top_ex_for_generation + self.n_iw_sampled_ex_for_generation

    @property
    def n_ex_for_scoring(self) -> int:
        """
        How many examples total we're going to ask the judge to classify during scoring.
        This list will be shuffled and includes:
          - leftover top activations
          - leftover IW activations
          - random windows
        """
        return (
            self.n_top_ex_for_scoring
            + self.n_random_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_iw_sampled_ex(self) -> int:
        """
        Total number of medium-activation (importance-weighted) examples per latent.
        """
        return (
            self.n_iw_sampled_ex_for_generation
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_correct_for_scoring(self) -> int:
        """
        During scoring we show n_ex_for_scoring windows, some of which truly activate the latent.
        The judge must return the indices (1-based) of all examples where it thinks the latent is active.
        We'll compare to ground truth to compute accuracy.
        """
        return (
            self.n_top_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def max_tokens_in_prediction(self) -> int:
        """
        Prediction responses are short: comma-separated integers, maybe 'None'.
        We give some margin here.
        """
        return 2 * self.n_ex_for_scoring + 5
