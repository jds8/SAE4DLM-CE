# autointerp_hf/judge.py
from __future__ import annotations

import os
import re
import json
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel
import asyncio
import traceback

from .examples import Examples

try:
    # openai>=1.x style client
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # we'll raise a clear error in __init__


class OpenAIJudgeConfig(BaseModel):
    """
    Configuration for the async LLM judge.
    This matches how run_eval.py builds it.
    """
    model: str
    base_url: str
    timeout: float
    max_retries: int
    debug: bool = False
    debug_truncate: int = 800


class AsyncOpenAIJudge:
    """
    Asynchronous wrapper that:
    1. Builds prompts for "explanation" and "scoring".
    2. Calls a chat completion endpoint (OpenAI-compatible).
    3. Parses model outputs.
    4. Emits very verbose debug prints (when cfg.debug=True) so we can
       inspect the raw provider response and see exactly where the text lives.

    Public methods expected by AutoInterpRunner:
      - generate_explanation(gen_examples, max_tokens_in_explanation, use_demos)
      - score_latent(explanation_text, scoring_examples)
    """

    def __init__(self, cfg: OpenAIJudgeConfig):
        self.cfg = cfg

        # Hard error if we don't have the OpenAI async client available.
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai python package not found. Please `pip install openai>=1.0`."
            )

        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in environment. "
                "The judge client needs it to call the remote model."
            )

        # Async client (OpenAI-compatible). Many providers mimic this interface.
        self.client = AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key=api_key,
            timeout=self.cfg.timeout,
        )

    # ------------------------------------------------------------------
    # High-level API called by AutoInterpRunner
    # ------------------------------------------------------------------

    async def generate_explanation(
        self,
        gen_examples: Examples,
        max_tokens_in_explanation: int,
        use_demos: bool,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Ask the judge model:
        "Given top-activating slices with << >> highlights, summarize
        what this latent fires on (one sentence)."

        Returns:
          explanation_text (str),
          gen_messages (list[{'role':..., 'content':...}])  # for logs
        """

        gen_messages = self._build_generation_messages(
            gen_examples=gen_examples,
            use_demos=use_demos,
        )

        if self.cfg.debug:
            print("\n[generate_explanation] ================================")
            print("[generate_explanation] Sending messages to judge:")
            for m in gen_messages:
                content_preview = (
                    m["content"][: self.cfg.debug_truncate] + "..."
                    if len(m["content"]) > self.cfg.debug_truncate
                    else m["content"]
                )
                print(
                    f"  role={m['role']}, "
                    f"content[0:{self.cfg.debug_truncate}]:\n{content_preview}\n"
                )

        raw_text, resp_json = await self._chat_completion(gen_messages)

        if self.cfg.debug:
            print("[generate_explanation] Raw judge response string repr:")
            print(repr(raw_text))
            print("[generate_explanation] Raw judge response json keys:")
            if isinstance(resp_json, dict):
                print(list(resp_json.keys()))
            else:
                print(type(resp_json))

        explanation_text = (raw_text or "").strip()
        # limit verbosity by word-ish count
        explanation_text = self._truncate_by_wordcount(
            explanation_text,
            max_tokens_in_explanation,
        )

        if self.cfg.debug:
            print("[generate_explanation] Final explanation_text repr:")
            print(repr(explanation_text))
            print("[generate_explanation] ==================================\n")

        return explanation_text, gen_messages

    async def score_latent(
        self,
        explanation_text: str,
        scoring_examples: Examples,
    ) -> Tuple[List[int], List[int], float, List[Dict[str, str]], str]:
        """
        Ask the judge model:
        "Given the explanation, here are several shuffled sequences.
         Which SHOULD this neuron activate on? Return comma-separated indices."

        Returns:
          predictions (list[int])              e.g. [2, 5, 9]
          correct_indices (list[int])          ground-truth active seqs (1-based)
          score (float)                        agreement accuracy
          scoring_messages (list[dict])        for logs
          raw_resp_text (str)                  raw text from judge (pre-parsing)
        """

        scoring_messages = self._build_scoring_messages(
            explanation_text=explanation_text,
            scoring_examples=scoring_examples,
        )

        if self.cfg.debug:
            print("\n[score_latent] =========================================")
            print("[score_latent] Sending messages to judge:")
            for m in scoring_messages:
                content_preview = (
                    m["content"][: self.cfg.debug_truncate] + "..."
                    if len(m["content"]) > self.cfg.debug_truncate
                    else m["content"]
                )
                print(
                    f"  role={m['role']}, "
                    f"content[0:{self.cfg.debug_truncate}]:\n{content_preview}\n"
                )

        raw_text, resp_json = await self._chat_completion(scoring_messages)

        if self.cfg.debug:
            print("[score_latent] Raw judge response string repr:")
            print(repr(raw_text))
            print("[score_latent] Raw judge response json keys:")
            if isinstance(resp_json, dict):
                print(list(resp_json.keys()))
            else:
                print(type(resp_json))

        parsed_predictions = self._parse_prediction_numbers(
            raw_text,
            n_examples=len(scoring_examples),
        )

        if self.cfg.debug:
            print("[score_latent] Parsed predictions:", parsed_predictions)

        # Ground truth: which sequences in scoring_examples actually had activation
        correct_indices = [
            (i + 1)
            for i, ex in enumerate(scoring_examples)
            if ex.is_active
        ]

        # Compute simple agreement accuracy:
        # for each i in 1..N:
        #   gt_active = i in correct_indices
        #   pr_active = i in parsed_predictions
        #   agreement += (gt_active == pr_active)
        n_total = len(scoring_examples)
        agree = 0
        for i in range(n_total):
            gt = ((i + 1) in correct_indices)
            pr = ((i + 1) in parsed_predictions)
            if gt == pr:
                agree += 1
        score = float(agree) / float(n_total) if n_total > 0 else 0.0

        if self.cfg.debug:
            print("[score_latent] correct_indices:", correct_indices)
            print("[score_latent] accuracy score:", score)
            print("[score_latent] =========================================\n")

        return parsed_predictions, correct_indices, score, scoring_messages, raw_text

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_generation_messages(
        self,
        gen_examples: Examples,
        use_demos: bool,
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for explanation-generation.

        Matches what we saw in your logs under "Generation phase":
        - system: instructions
        - user:   "The activating documents are given below:  1. ...  2. ..."
        """

        system_content = (
            "We're studying neurons in a neural network. Each neuron activates on "
            "some particular word/words/substring/concept in a short document. "
            "The activating words in each document are indicated with << ... >>. "
            "We will give you a list of documents on which the neuron activates, "
            "in order from most strongly activating to least strongly activating. "
            "Look at the parts of the document the neuron activates for and "
            "summarize in a single sentence what the neuron is activating on. "
            "Try not to be overly specific in your explanation. "
            "Note that some neurons will activate only on specific words or "
            "substrings, but others will activate on most/all words in a sentence "
            "provided that sentence contains some particular concept. "
            "Your explanation should cover most or all activating words. "
            "Pay attention to capitalization and punctuation, since they might matter."
        )

        if use_demos:
            # place for optional few-shot examples if you have them
            pass

        # show examples with << >> highlights
        user_list_str = gen_examples.to_numbered_string(highlight=True)
        user_content = (
            "The activating documents are given below:  " + user_list_str
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return messages

    def _build_scoring_messages(
        self,
        explanation_text: str,
        scoring_examples: Examples,
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for scoring.

        Matches logs under "Scoring phase":
        - system: instructions that force ONLY a comma-separated list of indices or 'None'
        - user: explanation + "Here are the examples: 1. ..."
        """

        system_content = (
            "We're studying neurons in a neural network. Each neuron activates on "
            "some particular word/words/substring/concept in a short document. "
            "You will be given a short explanation of what this neuron activates for, "
            "and then be shown several example sequences in random order. "
            "You must return a comma-separated list of the examples where you think "
            "the neuron should activate at least once, on ANY of the words or "
            "substrings in the document. For example, your response might look like "
            "\"2, 9, 10, 12\". Try not to be overly specific in your interpretation "
            "of the explanation. If you think there are no examples where the neuron "
            "will activate, you should just respond with \"None\". You should include "
            "nothing else in your response other than comma-separated numbers or the "
            "word \"None\" - this is important."
        )

        # for scoring, we DO NOT show activation highlights (they'd leak ground truth);
        # so highlight=False
        examples_list_str = scoring_examples.to_numbered_string(highlight=False)

        # include whatever explanation_text we currently think describes the neuron
        user_content = (
            "Here is the explanation: "
            + explanation_text.strip()
            + ".  Here are the examples:  "
            + examples_list_str
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return messages

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_by_wordcount(text: str, max_words: int) -> str:
        """
        Hard cap on explanation length. We don't want the judge to ramble.
        We'll split text by whitespace and keep only the first max_words tokens.
        """
        if max_words is None or max_words <= 0:
            return text
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    @staticmethod
    def _parse_prediction_numbers(raw_response: str, n_examples: int) -> List[int]:
        """
        Parse judge output into a sorted unique list of 1-based indices.

        - If the model says anything like "None" (or empty), we treat that
          as predicting zero activations.
        - Otherwise we grab all integers 1..n_examples from the string.
        """
        if not raw_response:
            return []

        raw = raw_response.strip()
        low = raw.lower()
        if "none" in low:
            return []

        # Extract any integers from the text.
        tokens = re.split(r"[^0-9]+", raw)
        nums: List[int] = []
        for t in tokens:
            if not t:
                continue
            try:
                val = int(t)
            except ValueError:
                continue
            if 1 <= val <= n_examples:
                nums.append(val)

        # de-dup + sort
        return sorted(set(nums))

    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Core network call:
        - send messages to remote LLM
        - get raw response object
        - convert to python dict (resp_json)
        - robustly try to pull assistant text from common fields
        - emit extensive debug info if cfg.debug is True

        Returns:
          (assistant_text, resp_json_dict)
        """

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                # openai>=1.x style:
                # await client.chat.completions.create(
                #     model=..., messages=[...], temperature=0.0
                # )
                resp = await self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=0.0,
                )

                resp_json = self._response_to_dict(resp)

                if self.cfg.debug:
                    print(f"[AsyncOpenAIJudge] COMPLETION RAW (attempt {attempt}) dict (truncated):")
                    try:
                        pretty = json.dumps(resp_json, indent=2)
                    except Exception:
                        pretty = str(resp_json)
                    print(pretty[:2000])
                    print()

                assistant_text = self._extract_assistant_text(resp_json)

                if self.cfg.debug:
                    print("[AsyncOpenAIJudge] extracted assistant_text repr:")
                    print(repr(assistant_text))
                    print()

                return assistant_text, resp_json

            except Exception as e:
                last_exc = e
                if self.cfg.debug:
                    print("[AsyncOpenAIJudge] ERROR during chat.completions.create:")
                    traceback.print_exc()
                # short backoff, then retry
                await asyncio.sleep(0.5)

        # if we exhausted retries:
        if last_exc is not None and self.cfg.debug:
            print("[AsyncOpenAIJudge] giving up after retries, returning empty string")
        return ("", {})  # fallback empty

    @staticmethod
    def _response_to_dict(resp: Any) -> Dict[str, Any]:
        """
        Best-effort turn the OpenAI client response object into a plain dict,
        so we can inspect keys and see where the text actually lives.

        - openai>=1.x returns a pydantic model with .model_dump()
        - some providers already return a dict
        - as a last resort we str() + json.loads()
        - final fallback: {"_unparsed_response_str": "..."}
        """
        # Already a dict?
        if isinstance(resp, dict):
            return resp

        # openai>=1.x (pydantic model)
        if hasattr(resp, "model_dump"):
            try:
                return resp.model_dump()
            except Exception:
                pass

        # try a generic 'to_dict'
        if hasattr(resp, "to_dict"):
            try:
                return resp.to_dict()  # type: ignore
            except Exception:
                pass

        # Fallback: string-ify then try JSON load
        try:
            as_str = str(resp)
            return json.loads(as_str)
        except Exception:
            # final fallback: opaque
            return {"_unparsed_response_str": str(resp)}

    @staticmethod
    def _extract_assistant_text(resp_json: Dict[str, Any]) -> str:
        """
        We don't assume the provider uses *exactly* OpenAI's schema.
        We'll try multiple common paths and return the first non-empty string.

        Common possibilities:
          resp["choices"][0]["message"]["content"]    (OpenAI official 1.x)
          resp["choices"][0]["text"]                  (many OpenAI-compatible providers)
          resp["output"] / resp["response"] / resp["data"] / resp["content"]
        """

        # 1. Official OpenAI-style
        try:
            val = resp_json["choices"][0]["message"]["content"]
            if isinstance(val, str) and val.strip():
                return val
        except Exception:
            pass

        # 2. Some providers still do "choices"[0]["text"]
        try:
            val = resp_json["choices"][0]["text"]
            if isinstance(val, str) and val.strip():
                return val
        except Exception:
            pass

        # 3. Some providers put final output in a top-level field
        for k in ["output", "response", "data", "content"]:
            if k in resp_json and isinstance(resp_json[k], str):
                if resp_json[k].strip():
                    return resp_json[k]

        # 4. Nothing worked
        return ""
