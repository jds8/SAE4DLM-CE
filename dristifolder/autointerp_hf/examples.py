# autointerp_hf/examples.py
from __future__ import annotations
from typing import List, Iterator, Optional, Any
import random
from dataclasses import dataclass
from tabulate import tabulate


def tokens_to_display_str(
    token_ids: List[int],
    active_mask: List[bool],
    tokenizer: Any,
    mark_toks: bool,
) -> str:
    """
    Convert a list of token IDs into a human-readable string.
    Optionally wrap "active" tokens (where the latent fires) in << >>.

    We approximate transformer_lens' to_str_tokens() using the HF tokenizer:
    - tokenizer.convert_ids_to_tokens() gives subword pieces.
    - GPT-NeoX/GPT2-style BPE often uses 'Ġ' as "this token starts with a space".
    - LLaMA/SentencePiece-style often uses '▁' to mean "this token starts a new word".
    - We also replace raw newlines with '↵' so table rows stay on one line.
    """

    toks = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

    out_pieces: List[str] = []
    for idx, (tok, is_active) in enumerate(zip(toks, active_mask)):
        # Drop special tokens entirely (BOS, EOS, etc.)
        if tok in getattr(tokenizer, "all_special_tokens", []):
            continue

        prefix_space = ""
        clean_tok = tok

        # GPT-NeoX / GPT2 convention ("ĠHello" => " Hello")
        if tok.startswith("Ġ"):
            prefix_space = " " if idx > 0 else ""
            clean_tok = tok[1:]

        # LLaMA / SentencePiece convention ("▁Hello" => " Hello")
        elif tok.startswith("▁"):
            prefix_space = " " if idx > 0 else ""
            clean_tok = tok[1:]

        # Avoid raw newlines wrecking the ASCII table layout
        piece_text = (prefix_space + clean_tok).replace("\n", "↵").replace("�", "")

        # Highlight tokens that exceed the activation threshold
        if mark_toks and is_active:
            piece_text = f"<<{piece_text}>>"

        out_pieces.append(piece_text)

    return "".join(out_pieces)


@dataclass
class Example:
    """
    Represents one context window for a single latent / SAE feature.

    Fields:
        token_ids: List[int]
            The token IDs in this window (fixed context_length).
        acts: List[float]
            The per-token activation values of *this latent* on that window.
            Must be same length as token_ids.
        act_threshold: float
            Threshold deciding if a token "counts" as active for this latent.
        tokenizer: Any
            HF tokenizer used to decode / pretty-print.

    Derived:
        active_mask: List[bool]
            active_mask[i] = True if acts[i] > act_threshold.
        is_active: bool
            True if ANY token in this window is active above threshold.

    Methods:
        to_str(mark_toks: bool):
            Convert tokens to a display string. If mark_toks=True, wrap
            high-activation tokens in << >>. Otherwise return plain text.
    """

    token_ids: List[int]
    acts: List[float]
    act_threshold: float
    tokenizer: Any

    def __post_init__(self):
        self.active_mask: List[bool] = [
            (float(a) > float(self.act_threshold)) for a in self.acts
        ]
        self.is_active: bool = any(self.active_mask)

    def to_str(self, mark_toks: bool = False) -> str:
        return tokens_to_display_str(
            token_ids=self.token_ids,
            active_mask=self.active_mask,
            tokenizer=self.tokenizer,
            mark_toks=mark_toks,
        )


class Examples:
    """
    A container around a list[Example] with helper methods for:
      - sorting/shuffling on init
      - table display for logs
      - numbered list formatting for LLM prompts
      - pretty-printing the chat messages we send to / get from the judge LLM

    Usage:
      * In "generation" (explanation) phase:
            shuffle=False
            -> We sort by descending max activation, so strongest examples
               appear first to the judge.
      * In "scoring" phase:
            shuffle=True
            -> We randomize order and mix positives/negatives before asking
               the judge to guess which ones should fire.

    After init:
      self.examples is always a list[Example] in the chosen order.
    """

    def __init__(self, examples: List[Example], shuffle: bool = False) -> None:
        if shuffle:
            # Randomize example order (for scoring phase)
            rng = random.Random()
            rng.shuffle(examples)
            self.examples = examples
        else:
            # Sort by strongest activation (for explanation phase)
            self.examples = sorted(
                examples,
                key=lambda ex: max(ex.acts) if len(ex.acts) > 0 else 0.0,
                reverse=True,
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def display(self, predictions: Optional[List[int]] = None) -> str:
        """
        Build an ASCII table using tabulate().

        1) Generation / explanation mode (predictions is None):
           Columns = [Top act, Sequence]
           - "Sequence" shows << >> markers for tokens whose activation
             crosses the threshold.

        2) Scoring mode (predictions is a list of 1-based indices predicted active):
           Columns = [Top act, Active?, Predicted?, Sequence]
           - "Active?"    is ground truth for this latent on that window.
           - "Predicted?" is judge guess ("Y" if this index is in predictions).
           - "Sequence"   is shown WITHOUT << >> markers (mark_toks=False),
             because in scoring we shouldn't leak the ground truth trigger.
        """
        rows: List[List[str]] = []

        for i, ex in enumerate(self.examples):
            top_act = max(ex.acts) if len(ex.acts) > 0 else 0.0

            if predictions is None:
                # Explanation view
                row = [
                    f"{top_act:.3f}",
                    ex.to_str(mark_toks=True),
                ]
                rows.append(row)
            else:
                # Scoring view
                predicted_active = (i + 1) in (predictions or [])
                row = [
                    f"{top_act:.3f}",
                    "Y" if ex.is_active else "",
                    "Y" if predicted_active else "",
                    ex.to_str(mark_toks=False),
                ]
                rows.append(row)

        headers = (
            ["Top act", "Sequence"]
            if predictions is None
            else ["Top act", "Active?", "Predicted?", "Sequence"]
        )

        return tabulate(
            rows,
            headers=headers,
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def to_table_string(self, predictions: Optional[List[int]] = None) -> str:
        """
        Thin wrapper so that autointerp.py can call:
            gen_examples.to_table_string()
        or:
            score_examples.to_table_string(pred_indices)

        Internally this just calls self.display(...).
        """
        return self.display(predictions=predictions)

    def to_numbered_string(self, highlight: bool) -> str:
        """
        Render the examples as a numbered list string for LLM prompts.

        The judge expects text like:
            "1. We<< know>> how important it is...
             2. I<< understand>> now that you are...
             3. I do know it was the right thing to do<< given>> ..."

        Rules:
        - Indexing starts at 1.
        - Use current ordering (sorted or shuffled, depending on phase).
        - If highlight=True:
              We call ex.to_str(mark_toks=True), so active tokens get << >>.
              This is used in the EXPLANATION prompt, where we WANT to show
              what actually fired so the judge can infer the concept.
        - If highlight=False:
              We call ex.to_str(mark_toks=False), i.e. plain text without
              << >> markers. This is used in the SCORING prompt, where we
              ASK the judge to guess which sequences will activate, so we
              CANNOT leak ground truth highlights.

        Returns:
            A single string:
            "1. some text  2. some other text  3. yet another text"
            (joined with two spaces).
        """
        parts: List[str] = []
        for idx, ex in enumerate(self.examples, start=1):
            seq_txt = ex.to_str(mark_toks=highlight)
            seq_txt = str(seq_txt).strip()
            parts.append(f"{idx}. {seq_txt}")
        return "  ".join(parts)

    def messages_to_table(self, messages: List[dict]) -> str:
        """
        Pretty-print the conversation we send to / receive from the judge LLM
        as an ASCII table, so it can be embedded into the final logs.

        `messages` is expected to be a list of dicts like:
            {"role": "system", "content": "..."}
            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "model reply"}

        We replace newlines in content with spaces or '↵' so each row
        fits on one line in the ASCII table.
        """
        rows: List[List[str]] = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            # flatten newlines for prettier single-row display
            content = content.replace("\n", " ").replace("�", "").strip()
            rows.append([role, content])

        return tabulate(
            rows,
            headers=["role", "content"],
            tablefmt="simple_outline",
        )

    def to_scoring_table_string(
        self,
        correct_indices: Optional[List[int]] = None,
        pred_indices: Optional[List[int]] = None,
    ) -> str:
        """
        Produce the ASCII table used in the scoring-phase logs:
        columns are [Top act, Active?, Predicted?, Sequence]

        Arguments:
            correct_indices:
                Optional list of 1-based indices that are truly active.
                If provided, we will use it to mark the "Active?" column.
                If not provided, we fall back to Example.is_active.
            pred_indices:
                Optional list of 1-based indices the judge claimed
                would activate. Used to mark the "Predicted?" column.

        Note:
        - We NEVER highlight tokens in scoring, because the LLM judge
          is supposed to infer which sequences fire *without* seeing
          ground-truth << >> markers.
        """
        rows: List[List[str]] = []
        for idx, ex in enumerate(self.examples, start=1):
            # top activation magnitude within this window
            top_act = max(ex.acts) if len(ex.acts) > 0 else 0.0

            # ground truth active?
            if correct_indices is not None:
                gt_active = (idx in correct_indices)
            else:
                gt_active = ex.is_active

            # judge predicted active?
            pred_active = (idx in (pred_indices or []))

            rows.append([
                f"{top_act:.3f}",
                "Y" if gt_active else "",
                "Y" if pred_active else "",
                ex.to_str(mark_toks=False),
            ])

        return tabulate(
            rows,
            headers=["Top act", "Active?", "Predicted?", "Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )
