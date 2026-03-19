# autointerp_hf/autointerp.py
from __future__ import annotations
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor

from .config import AutoInterpEvalConfig
from .examples import Example, Examples
from .indexing_utils import (
    get_k_largest_indices,
    get_iw_sample_indices,
    index_with_buffer,
)
from .hooks import (
    collect_sae_activations_hf,
    get_feature_activation_sparsity_hf,
)
from .judge import AsyncOpenAIJudge
from tqdm import tqdm


@dataclass
class AutoInterpResultSingleLatent:
    """
    Container for the final result for one latent.
    """
    latent_id: int
    explanation: str
    predictions: Optional[List[int]]
    correct_seqs: Optional[List[int]]
    score: Optional[float]
    logs: str


class AutoInterpRunner:
    """
    End-to-end runner for:
    - Step 1: gather_data()  -> build generation_examples / scoring_examples
    - Step 2: generate explanation via judge
    - Step 3: scoring: judge predicts which windows are active, we compute accuracy

    We DO NOT persist activations to disk. We do everything in-memory.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model,
        sae,
        tokenizer,
        input_ids: Tensor,          # [N, L]
        attention_mask: Tensor,     # [N, L]
        sparsity: Tensor,           # [F], from get_feature_activation_sparsity_hf
        judge: AsyncOpenAIJudge,
    ):
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.judge = judge

        # Select which latents we will interpret
        if cfg.latents is not None:
            # user gave override_latents explicitly
            self.latents = cfg.latents
        else:
            assert cfg.n_latents is not None

            # scale sparsity by total_tokens to get approx. activation counts
            # and pick "alive" latents with enough activations
            activation_counts = sparsity * cfg.total_tokens
            alive_latents = (
                torch.nonzero(
                    activation_counts > cfg.dead_latent_threshold
                ).squeeze(1).tolist()
            )

            if len(alive_latents) < cfg.n_latents:
                print(
                    f"[WARN] only {len(alive_latents)} alive latents "
                    f"(threshold={cfg.dead_latent_threshold}), "
                    f"less than requested {cfg.n_latents}"
                )
                self.latents = alive_latents
            else:
                random.seed(cfg.random_seed)
                self.latents = random.sample(alive_latents, k=cfg.n_latents)

        self.n_latents = len(self.latents)

    def _create_examples_from_windows(
        self,
        toks_slice_2d: Tensor,   # [num_windows, win_len]
        acts_slice_2d: Optional[Tensor],
        act_threshold: float,
    ) -> List[Example]:
        """
        Turn each window of tokens (and optional per-token activations) into
        an Example object usable downstream.
        """
        examples_list: List[Example] = []
        if acts_slice_2d is None:
            # If acts not provided, treat all activations as zeros
            acts_slice_2d = torch.zeros_like(toks_slice_2d, dtype=torch.float32)

        for win_tokens, win_acts in zip(toks_slice_2d.tolist(), acts_slice_2d.tolist()):
            ex = Example(
                token_ids=win_tokens,
                acts=win_acts,
                act_threshold=act_threshold,
                tokenizer=self.tokenizer,
            )
            examples_list.append(ex)

        return examples_list

    def _split_top_and_iw(
        self,
        top_toks: Tensor,
        top_vals: Tensor,
        iw_toks: Tensor,
        iw_vals: Tensor,
    ) -> Tuple[List[Example], List[Example]]:
        cfg = self.cfg

        # >>> NEW: use actual counts (after possible filtering)
        top_count = top_toks.shape[0]
        iw_count  = iw_toks.shape[0]

        # how many go to generation set (cap by available)
        n_top_gen = min(cfg.n_top_ex_for_generation, top_count)
        n_iw_gen  = min(cfg.n_iw_sampled_ex_for_generation, iw_count)

        # random permutations based on actual counts
        top_perm = torch.randperm(top_count, device=top_toks.device)
        iw_perm  = torch.randperm(iw_count,  device=iw_toks.device)

        top_gen_idx   = top_perm[: n_top_gen]
        top_score_idx = top_perm[n_top_gen:]  # rest go to scoring

        iw_gen_idx    = iw_perm[: n_iw_gen]
        iw_score_idx  = iw_perm[n_iw_gen:]    # rest go to scoring
        # <<< NEW

        # slice
        top_gen_toks = top_toks[top_gen_idx]
        top_gen_vals = top_vals[top_gen_idx]
        top_score_toks = top_toks[top_score_idx]
        top_score_vals = top_vals[top_score_idx]

        iw_gen_toks = iw_toks[iw_gen_idx]
        iw_gen_vals = iw_vals[iw_gen_idx]
        iw_score_toks = iw_toks[iw_score_idx]
        iw_score_vals = iw_vals[iw_score_idx]

        return (
            self._create_examples_from_windows(
                torch.cat([top_gen_toks, iw_gen_toks], dim=0),
                torch.cat([top_gen_vals, iw_gen_vals], dim=0),
                act_threshold=0.0,
            ),
            self._create_examples_from_windows(
                torch.cat([top_score_toks, iw_score_toks], dim=0),
                torch.cat([top_score_vals, iw_score_vals], dim=0),
                act_threshold=0.0,
            ),
        )


    def gather_data(
        self,
    ) -> Tuple[Dict[int, Examples], Dict[int, Examples]]:
        """
        STEP 1:
        Build per-latent generation_examples & scoring_examples.

        This:
        - runs the model with hooks to collect latent activations via the SAE
        - picks strongest activations (top_k)
        - picks "medium" activations via importance-weighted sampling
        - picks random windows as negatives
        - constructs `Examples` objects for generation & scoring sets
        """

        cfg = self.cfg
        device = self.input_ids.device
        N, L = self.input_ids.shape

        assert cfg.batch_size is not None, "cfg.batch_size must be set"

        # 1. Collect SAE activations for *selected* latents only
        #    acts_all: [N, L, len(self.latents)]
        acts_all = collect_sae_activations_hf(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            model=self.model,
            sae=self.sae,
            batch_size=cfg.batch_size,
            hook_module_path=cfg.hook_module_path,
            tokenizer=self.tokenizer,
            mask_special_tokens=True,
            selected_latents=self.latents,
            activation_dtype=torch.bfloat16,
        )

        generation_examples: Dict[int, Examples] = {}
        scoring_examples: Dict[int, Examples] = {}

        # 2. For each latent index into self.latents, pick examples
        for i, latent_id in enumerate(self.latents):
            latent_acts_2d = acts_all[:, :, i]  # [N, L]

            # (a) random negatives
            rand_rows = torch.randint(
                low=0,
                high=N,
                size=(cfg.n_random_ex_for_scoring,),
                device=device,
            )
            rand_cols = torch.randint(
                low=cfg.buffer,
                high=L - cfg.buffer,
                size=(cfg.n_random_ex_for_scoring,),
                device=device,
            )
            rand_indices = torch.stack([rand_rows, rand_cols], dim=-1)  # [R, 2]
            rand_toks = index_with_buffer(self.input_ids, rand_indices, cfg.buffer)

            # We'll treat random acts as all zeros later.

            # (b) top activations
            top_indices = get_k_largest_indices(
                latent_acts_2d,
                k=cfg.n_top_ex,
                buffer=cfg.buffer,
                no_overlap=cfg.no_overlap,
            )
            if top_indices.shape[0] == 0:
                # No strong activations => consider latent dead for now
                continue

            valid_top_mask = (top_indices[:, 1] >= cfg.buffer) & (top_indices[:, 1] < L - cfg.buffer)
            top_indices = top_indices[valid_top_mask]
            if top_indices.shape[0] == 0:
                # after filtering, nothing left
                continue

            top_toks = index_with_buffer(self.input_ids, top_indices, cfg.buffer)
            top_vals = index_with_buffer(latent_acts_2d, top_indices, cfg.buffer)

            # activation threshold for highlighting: frac * max activation
            act_threshold = cfg.act_threshold_frac * top_vals.max().item()

            # (c) importance-weighted (medium) activations
            # We'll block out the top peaks so IW sampling doesn't duplicate them
            center_vals = top_vals[:, cfg.buffer]  # center position's act for each top window
            threshold_for_top = center_vals.min().item()

            # zero out acts >= threshold_for_top so we focus on medium activations
            acts_thresholded = torch.where(
                latent_acts_2d >= threshold_for_top,
                torch.zeros_like(latent_acts_2d),
                latent_acts_2d,
            )

            if acts_thresholded[
                :, cfg.buffer : (L - cfg.buffer)
            ].max() < 1e-6:
                # no meaningful medium activations => skip
                continue

            iw_indices = get_iw_sample_indices(
                acts_thresholded,
                k=cfg.n_iw_sampled_ex,
                buffer=cfg.buffer,
            )

            valid_iw_mask = (iw_indices[:, 1] >= cfg.buffer) & (iw_indices[:, 1] < L - cfg.buffer)
            iw_indices = iw_indices[valid_iw_mask]
            if iw_indices.shape[0] == 0:
                # after filtering, nothing left
                continue

            iw_toks = index_with_buffer(self.input_ids, iw_indices, cfg.buffer)
            iw_vals = index_with_buffer(latent_acts_2d, iw_indices, cfg.buffer)

            # (d) split top/iw into generation vs scoring sets
            gen_list, scoring_topiw_list = self._split_top_and_iw(
                top_toks,
                top_vals,
                iw_toks,
                iw_vals,
            )

            # (e) set the correct threshold on each Example in these lists
            for ex in gen_list:
                ex.act_threshold = act_threshold
                ex.__post_init__()
            for ex in scoring_topiw_list:
                ex.act_threshold = act_threshold
                ex.__post_init__()

            # (f) random negatives -> Examples with zero activations
            rand_list = self._create_examples_from_windows(
                toks_slice_2d=rand_toks,
                acts_slice_2d=torch.zeros_like(rand_toks, dtype=torch.float32),
                act_threshold=act_threshold,
            )
            for ex in rand_list:
                ex.__post_init__()

            # generation_examples[latent] = sorted by top activation (no shuffle)
            generation_examples[latent_id] = Examples(
                gen_list,
                shuffle=False,
            )

            # scoring_examples[latent] = mix of scoring_topiw_list + rand_list, then shuffle
            scoring_examples[latent_id] = Examples(
                scoring_topiw_list + rand_list,
                shuffle=True,
            )

        return generation_examples, scoring_examples

    # ------------------------------------------------------------------
    # (older prompt-builder helpers used by run_single_latent; we keep
    # them around for completeness/debug, but run() below now calls
    # judge.generate_explanation / judge.score_latent directly.)
    # ------------------------------------------------------------------

    def get_generation_prompts(self, generation_examples: Examples) -> List[dict]:
        """
        Build the messages for explanation generation (Step 2).
        """
        assert len(generation_examples) > 0, "No generation examples found"

        examples_as_str = "\n".join(
            [
                f"{i+1}. {ex.to_str(mark_toks=True)}"
                for i, ex in enumerate(generation_examples)
            ]
        )

        system_prompt = (
            "We're studying neurons in a neural network. "
            "Each neuron activates on some particular word/words/substring/concept "
            "in a short document. The activating words in each document are "
            "indicated with << ... >>. We will give you a list of documents on "
            "which the neuron activates, in order from most strongly activating "
            "to least strongly activating. Look at the parts of the document "
            "the neuron activates for and summarize in a single sentence what "
            "the neuron is activating on. Try not to be overly specific in "
            "your explanation. Note that some neurons will activate only on "
            "specific words or substrings, but others will activate on most/all "
            "words in a sentence provided that sentence contains some particular "
            "concept. Your explanation should cover most or all activating "
            "words. Pay attention to capitalization & punctuation if relevant. "
            "Keep the explanation as short and simple as possible, limited "
            "to 20 words or less. Avoid long lists of words."
        )

        if self.cfg.use_demos_in_explanation:
            system_prompt += (
                " Some examples: \"This neuron activates on the word 'knows' in "
                "rhetorical questions\", and \"This neuron activates on verbs "
                "related to decision-making and preferences\", and \"This neuron "
                "activates on the substring 'Ent' at the start of words\", and "
                "\"This neuron activates on text about government economic policy\"."
            )
        else:
            system_prompt += (
                " Your response should be in the form "
                "\"This neuron activates on...\""
            )

        user_prompt = (
            f"The activating documents are given below:\n\n{examples_as_str}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def get_scoring_prompts(
        self,
        explanation: str,
        scoring_examples: Examples,
    ) -> List[dict]:
        """
        Build the messages for scoring (Step 3).
        """
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join(
            [
                f"{i+1}. {ex.to_str(mark_toks=False)}"
                for i, ex in enumerate(scoring_examples)
            ]
        )

        # fabricate a sample style like "2, 5, 7"
        idxs = random.sample(
            range(1, 1 + self.cfg.n_ex_for_scoring),
            k=min(self.cfg.n_correct_for_scoring, self.cfg.n_ex_for_scoring),
        )
        idxs.sort()
        example_response_str = ", ".join([str(i) for i in idxs])

        system_prompt = (
            "We're studying neurons in a neural network. Each neuron activates "
            "on some particular word/words/substring/concept in a short document. "
            "You will be given a short explanation of what this neuron activates for, "
            f"and then be shown {self.cfg.n_ex_for_scoring} example sequences in random order. "
            "You will have to return a comma-separated list of the examples where "
            "you think the neuron should activate at least once, on ANY of the words "
            "or substrings in the document. For example, your response might look "
            f"like \"{example_response_str}\". Try not to be overly specific in "
            "your interpretation of the explanation. If you think there are "
            "no examples where the neuron will activate, you should just respond "
            "with \"None\". You should include nothing else in your response "
            "other than comma-separated numbers or the word \"None\" - this is important."
        )

        user_prompt = (
            f"Here is the explanation: this neuron fires on {explanation}.\n\n"
            f"Here are the examples:\n\n{examples_as_str}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Small helper parsers/metrics (used by run_single_latent)
    # ------------------------------------------------------------------

    def parse_explanation(self, explanation_raw: str) -> str:
        """
        Extract the "core" explanation. We mimic the saebench heuristic by
        cutting after 'activates on', trimming punctuation.
        """
        out = explanation_raw.split("activates on")[-1].rstrip(".").strip()
        return out

    def parse_predictions(self, predictions_raw: str) -> Optional[List[int]]:
        """
        Parse a string like "2, 5, 7" or "2 and 5" or "None" into a list of ints.
        Return None if parsing fails catastrophically (judge answered incorrectly).
        """
        txt = predictions_raw.strip().rstrip(".")
        txt = txt.replace("and", ",")
        txt = txt.replace("None", "")
        parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
        if len(parts) == 0:
            return []
        if not all(p.isdigit() for p in parts):
            return None
        return [int(p) for p in parts]

    def score_predictions(
        self,
        predictions: List[int],
        scoring_examples: Examples,
    ) -> float:
        """
        Compute accuracy:
        - classification for each example i is (i+1 in predictions)
        - ground truth is ex.is_active
        - final score = mean(match)
        """
        classifications = [
            (i + 1) in predictions
            for i in range(len(scoring_examples))
        ]
        truth = [ex.is_active for ex in scoring_examples]
        correct = [
            int(c == t) for (c, t) in zip(classifications, truth)
        ]
        return sum(correct) / max(len(correct), 1)

    # ------------------------------------------------------------------
    # (Legacy) per-latent pipeline that calls self.judge.chat directly.
    # We keep it for reference; the main path now is run().
    # ------------------------------------------------------------------

    async def run_single_latent(
        self,
        latent_id: int,
        generation_examples: Examples,
        scoring_examples: Examples,
    ) -> Optional[AutoInterpResultSingleLatent]:
        """
        Legacy path. Not used by run() anymore but kept for reference.
        """
        # Step 2: explanation
        gen_messages = self.get_generation_prompts(generation_examples)
        expl_raw, gen_logs = await self.judge.chat(
            messages=gen_messages,
            max_tokens=self.cfg.max_tokens_in_explanation,
            n_completions=1,
        )
        explanation = self.parse_explanation(expl_raw)

        logs_accum = (
            "Generation phase\n"
            + gen_logs
            + "\n"
            + generation_examples.display(predictions=None)
        )

        predictions_list: Optional[List[int]] = None
        correct_list: Optional[List[int]] = None
        score_val: Optional[float] = None

        # Step 3: scoring
        if self.cfg.scoring:
            score_messages = self.get_scoring_prompts(
                explanation=explanation,
                scoring_examples=scoring_examples,
            )
            preds_raw, score_logs = await self.judge.chat(
                messages=score_messages,
                max_tokens=self.cfg.max_tokens_in_prediction,
                n_completions=1,
            )

            predictions_parsed = self.parse_predictions(preds_raw)
            if predictions_parsed is None:
                # judge didn't follow format, skip
                return None

            predictions_list = predictions_parsed
            score_val = self.score_predictions(
                predictions=predictions_parsed,
                scoring_examples=scoring_examples,
            )
            correct_list = [
                i
                for i, ex in enumerate(scoring_examples, start=1)
                if ex.is_active
            ]

            logs_accum += (
                "\nScoring phase\n"
                + score_logs
                + "\n"
                + scoring_examples.display(predictions=predictions_list)
            )

        return AutoInterpResultSingleLatent(
            latent_id=latent_id,
            explanation=explanation,
            predictions=predictions_list,
            correct_seqs=correct_list,
            score=score_val,
            logs=logs_accum,
        )

    # ------------------------------------------------------------------
    # Main path used by run_eval.py
    # ------------------------------------------------------------------

    async def run(self) -> Dict[int, dict]:
        """
        Main entry point for the pipeline:
        1. Collect per-latent examples (top activations for explanation, mixed activations for scoring).
        2. For each latent:
           - Ask judge LLM for a natural language explanation.
           - Ask judge LLM to predict which held-out sequences fire.
           - Compute accuracy.
        3. Return a dict {latent_id: {...result info...}}.

        We also wrap the per-latent loop in a tqdm progress bar so we
        can see progress live in the terminal, e.g.:
            "LLM judge per-latent:  40%|████ ..."

        This path uses AsyncOpenAIJudge.generate_explanation() / score_latent()
        that we just fixed for debugging and robust parsing.
        """

        # 1) Gather the candidate examples up front (one dict per latent)
        generation_examples_dict, scoring_examples_dict = self.gather_data()

        # Latent IDs we will iterate over
        latent_ids = sorted(generation_examples_dict.keys())

        results: Dict[int, dict] = {}

        # 2) Loop over latents with a tqdm progress bar
        for latent_id in tqdm(latent_ids, desc="LLM judge per-latent", unit="latent"):
            gen_examples = generation_examples_dict[latent_id]
            scoring_examples = scoring_examples_dict[latent_id]

            # -------------------------------------------------
            # (A) Ask judge LLM to generate a natural language explanation
            # -------------------------------------------------
            explanation_text, gen_messages = await self.judge.generate_explanation(
                gen_examples=gen_examples,
                max_tokens_in_explanation=self.cfg.max_tokens_in_explanation,
                # FIX HERE: use_demos instead of use_demos_in_explanation
                use_demos=self.cfg.use_demos_in_explanation,
            )

            # Pretty "Top act | Sequence" table for logging
            gen_table_str = gen_examples.to_table_string()

            # Start building a per-latent log blob (what ends up in JSON "logs")
            latent_log_parts = []
            latent_log_parts.append("Generation phase")
            # dump the full prompt sent to judge (role/content table)
            latent_log_parts.append(gen_examples.messages_to_table(gen_messages))
            # dump gen examples table
            latent_log_parts.append(gen_table_str)

            # -------------------------------------------------
            # (B) Ask judge LLM to score / predict which held-out sequences fire
            # -------------------------------------------------
            if self.cfg.scoring:
                (
                    pred_indices,
                    correct_indices,
                    acc,
                    score_messages,
                    raw_answer,
                ) = await self.judge.score_latent(
                    explanation_text=explanation_text,
                    scoring_examples=scoring_examples,
                )

                # scoring table with GT vs pred vs text
                scoring_table_str = scoring_examples.to_scoring_table_string(
                    pred_indices=pred_indices
                )

                # add scoring-phase info to the log
                latent_log_parts.append("Scoring phase")
                latent_log_parts.append(scoring_examples.messages_to_table(score_messages))
                latent_log_parts.append(scoring_table_str)

                # stash final structured result for this latent
                results[latent_id] = {
                    "latent": latent_id,
                    "explanation": explanation_text,
                    "predictions": pred_indices,
                    "correct_seqs": correct_indices,
                    "score": acc,
                    "logs": "\n".join(latent_log_parts),
                }
            else:
                # scoring disabled: still record explanation/logs
                results[latent_id] = {
                    "latent": latent_id,
                    "explanation": explanation_text,
                    "predictions": [],
                    "correct_seqs": [],
                    "score": 0.0,
                    "logs": "\n".join(latent_log_parts),
                }

        # 3) return the dict of results for all latents
        return results
