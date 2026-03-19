"""
LLM-based concept scoring (LLM judge).

Fixes in this version:
- Detects "reasoning-only" completions that hit max_tokens and produce empty content.
- Automatically retries with a larger token budget (JUDGE_MAX_TOKENS_ON_EMPTY) when that happens.
- Keeps "reasoning" parameter out of requests (gateway returns 400 otherwise).
- Robust assistant content extraction (supports both str and content-part arrays).
- Optional JSON mode with fallback if gateway rejects it.
- Debug JSONL logging path is taken from config.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # will raise in get_global_judge()


# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------
try:
    from config import (
        OPENAI_BASE_URL_DEFAULT,
        MODEL_NAME,
        JUDGE_TIMEOUT,
        JUDGE_DEBUG,
        JUDGE_MAX_CONCURRENCY,
        JUDGE_MAX_RETRIES,
        JUDGE_BACKOFF_BASE,
        JUDGE_BACKOFF_JITTER,
        JUDGE_FALLBACK_SCORE,
        JUDGE_MAX_TOKENS,
        JUDGE_MAX_TOKENS_ON_EMPTY,
        JUDGE_USE_JSON_MODE,
        JUDGE_SANITIZE_TEXT,
        JUDGE_LOG_PATH,
        JUDGE_LOG_TRUNCATE_CHARS,
        OPENAI_API_KEY_CONFIG,
    )
except Exception:
    OPENAI_BASE_URL_DEFAULT = "https://api.shubiaobiao.cn/v1"
    MODEL_NAME = "gpt-4o-mini"
    JUDGE_TIMEOUT = 60.0
    JUDGE_DEBUG = False
    JUDGE_MAX_CONCURRENCY = 5
    JUDGE_MAX_RETRIES = 6
    JUDGE_BACKOFF_BASE = 0.8
    JUDGE_BACKOFF_JITTER = 0.4
    JUDGE_FALLBACK_SCORE = 50.0
    JUDGE_MAX_TOKENS = 128
    JUDGE_MAX_TOKENS_ON_EMPTY = 256
    JUDGE_USE_JSON_MODE = True
    JUDGE_SANITIZE_TEXT = True
    JUDGE_LOG_PATH = "/tmp/llm_judge_debug.jsonl"
    JUDGE_LOG_TRUNCATE_CHARS = 0
    OPENAI_API_KEY_CONFIG = ""


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _safe_json_dumps(obj: Any, limit: int = 2000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "...(truncated)"
    return s


def _env_flag(name: str, default: bool) -> bool:
    """
    Parse env var like "0/1", "false/true", "no/yes".
    If not set, return default.
    """
    v = os.environ.get(name, None)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _sanitize_text(s: str) -> str:
    """
    Conservative sanitizer:
    - Remove NUL bytes and other control chars (keep \n \t).
    - Strip extreme whitespace.
    """
    if not s:
        return ""
    s = s.replace("\x00", "")
    # Remove control chars except newline/tab
    s = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s.strip()


def _truncate_for_log(s: str, limit: int) -> str:
    if limit and limit > 0 and len(s) > limit:
        return s[:limit] + "...(truncated)"
    return s


# ---------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------
@dataclass
class OpenAIJudgeConfig:
    model: str = MODEL_NAME
    base_url: str = OPENAI_BASE_URL_DEFAULT
    timeout: float = float(JUDGE_TIMEOUT)
    debug: bool = bool(JUDGE_DEBUG)

    # Primary token budget
    max_tokens: int = int(JUDGE_MAX_TOKENS)

    # If we get empty visible output and reasoning consumed the whole budget, retry with this.
    max_tokens_on_empty: int = int(JUDGE_MAX_TOKENS_ON_EMPTY)

    max_concurrency: int = int(JUDGE_MAX_CONCURRENCY)
    max_retries: int = int(JUDGE_MAX_RETRIES)
    backoff_base: float = float(JUDGE_BACKOFF_BASE)
    backoff_jitter: float = float(JUDGE_BACKOFF_JITTER)

    fallback_score: float = float(JUDGE_FALLBACK_SCORE)

    # JSON mode helps some gateways to enforce JSON output.
    json_mode: bool = bool(JUDGE_USE_JSON_MODE)

    sanitize_text: bool = bool(JUDGE_SANITIZE_TEXT)

    # Debug log path (JSONL)
    debug_log_path: str = str(JUDGE_LOG_PATH)
    log_truncate_chars: int = int(JUDGE_LOG_TRUNCATE_CHARS)


# ---------------------------------------------------------------------
# Judge implementation
# ---------------------------------------------------------------------
class AsyncOpenAIJudge:
    """
    Async judge wrapper that scores (without_text, after_text) in a single request.
    """

    def __init__(self, cfg: OpenAIJudgeConfig):
        self.cfg = cfg

        if AsyncOpenAI is None:
            raise RuntimeError("openai package not found. Please `pip install openai>=1.0`.")

        api_key = os.environ.get("OPENAI_API_KEY", None) or (OPENAI_API_KEY_CONFIG or None)
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set (env) and OPENAI_API_KEY_CONFIG is empty.")

        self.client = AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key=api_key,
            timeout=self.cfg.timeout,
        )
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)

        # Allow overriding debug/json_mode from env (optional).
        self.cfg.debug = _env_flag("SAE_JUDGE_DEBUG", self.cfg.debug)
        self.cfg.json_mode = _env_flag("SAE_JUDGE_JSON_MODE", self.cfg.json_mode)

    # -----------------------------------------------------------------
    # Prompt builder
    # -----------------------------------------------------------------
    @staticmethod
    def _build_pair_scoring_messages(
        explanation: str,
        without_text: str,
        after_text: str,
    ) -> List[Dict[str, str]]:
        # Keep prompts short to reduce reasoning overhead.
        system_content = (
            "Return ONLY a strict JSON object with exactly two numeric fields: "
            "\"score_without\" and \"score_after\" (0..100). "
            "No other text."
        )

        user_content = (
            "[Concept]\n"
            f"{explanation}\n\n"
            "[Without]\n"
            f"{without_text}\n\n"
            "[After]\n"
            f"{after_text}\n\n"
            "Score each text for how strongly it expresses the Concept (0..100). "
            "Return ONLY:\n"
            "{\"score_without\": <0..100>, \"score_after\": <0..100>}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # -----------------------------------------------------------------
    # Response parsing helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _response_to_dict(resp: Any) -> Dict[str, Any]:
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "model_dump"):
            try:
                return resp.model_dump()
            except Exception:
                pass
        if hasattr(resp, "to_dict"):
            try:
                return resp.to_dict()  # type: ignore
            except Exception:
                pass
        try:
            return json.loads(str(resp))
        except Exception:
            return {"_unparsed_response_str": str(resp)}

    @staticmethod
    def _extract_assistant_text(resp_json: Dict[str, Any]) -> str:
        # OpenAI-style: choices[0].message.content
        try:
            content = resp_json["choices"][0]["message"]["content"]
            # content can be str or array of parts
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for p in content:
                    if isinstance(p, dict):
                        # Common formats: {"type":"text","text":"..."} or {"text":"..."}
                        if "text" in p and isinstance(p["text"], str):
                            parts.append(p["text"])
                return "".join(parts)
        except Exception:
            pass

        # Some providers: choices[0].text
        try:
            val = resp_json["choices"][0]["text"]
            if isinstance(val, str):
                return val
        except Exception:
            pass

        # Fallback: some providers use top-level fields
        for k in ["output", "response", "data", "content"]:
            if k in resp_json and isinstance(resp_json[k], str):
                return resp_json[k]

        return ""

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("Empty response text")
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or first >= last:
            raise ValueError(f"No JSON object found in response: {text!r}")
        return json.loads(text[first : last + 1])

    @staticmethod
    def _coerce_score(val: Any, fallback: float) -> float:
        try:
            x = float(val)
        except Exception:
            return float(fallback)
        if x != x:  # NaN
            return float(fallback)
        return float(max(0.0, min(100.0, x)))

    @staticmethod
    def _looks_like_reasoning_only_budget_exhaustion(resp_json: Dict[str, Any], max_tokens: int) -> bool:
        """
        Detect the pattern shown in your logs:
        - finish_reason == "length"
        - completion_tokens_details.reasoning_tokens == max_tokens
        - output_tokens == 0 or accepted_prediction_tokens == 0
        - message.content == ""
        """
        try:
            finish_reason = resp_json["choices"][0].get("finish_reason", None)
            usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
            ctd = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}

            reasoning = int(ctd.get("reasoning_tokens", 0) or 0)
            accepted_pred = ctd.get("accepted_prediction_tokens", None)
            output_tokens = usage.get("output_tokens", None)

            if finish_reason != "length":
                return False
            if reasoning < max_tokens:
                return False
            if output_tokens is not None and int(output_tokens) > 0:
                return False
            if accepted_pred is not None and int(accepted_pred) > 0:
                return False
            return True
        except Exception:
            return False

    # -----------------------------------------------------------------
    # Debug logging
    # -----------------------------------------------------------------
    def _append_debug_log(self, record: Dict[str, Any]) -> None:
        if not self.cfg.debug:
            return
        try:
            os.makedirs(os.path.dirname(self.cfg.debug_log_path), exist_ok=True)
            with open(self.cfg.debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Never crash evaluation due to logging.
            pass

    def _debug_print(self, *args: Any) -> None:
        if self.cfg.debug:
            print("[AsyncOpenAIJudge DEBUG]", *args)

    # -----------------------------------------------------------------
    # Core network call (NEVER raises out; always returns a tuple)
    # -----------------------------------------------------------------
    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
          assistant_text (may be empty if all retries fail),
          resp_json (last response dict or {}),
          meta (attempts/errors/etc.)

        This function NEVER raises to the caller (to avoid crashing the pipeline).
        """
        errors: List[str] = []
        last_resp_json: Dict[str, Any] = {}

        use_json_mode = bool(self.cfg.json_mode)

        # Dynamic token budget:
        # start with primary max_tokens, and upgrade to max_tokens_on_empty
        # only if we detect "reasoning-only budget exhaustion".
        cur_max_tokens = int(self.cfg.max_tokens)
        upgraded_budget = False

        for attempt in range(self.cfg.max_retries):
            try:
                req_kwargs: Dict[str, Any] = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": int(cur_max_tokens),
                }

                if use_json_mode:
                    req_kwargs["response_format"] = {"type": "json_object"}

                # IMPORTANT: do NOT send "reasoning" anywhere. Your gateway rejects it.

                async with self._sem:
                    resp = await self.client.chat.completions.create(**req_kwargs)

                resp_json = self._response_to_dict(resp)
                last_resp_json = resp_json

                assistant_text = (self._extract_assistant_text(resp_json) or "").strip()

                if not assistant_text:
                    # Detect your exact failure mode and upgrade budget once.
                    if (
                        (not upgraded_budget)
                        and (self.cfg.max_tokens_on_empty > cur_max_tokens)
                        and self._looks_like_reasoning_only_budget_exhaustion(resp_json, cur_max_tokens)
                    ):
                        upgraded_budget = True
                        cur_max_tokens = int(self.cfg.max_tokens_on_empty)
                        errors.append("empty_assistant_content_reasoning_budget_exhausted_upgrade_tokens")
                        if self.cfg.debug:
                            usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
                            self._debug_print(
                                f"attempt {attempt+1}: empty content due to reasoning-only exhaustion; "
                                f"upgrading max_tokens to {cur_max_tokens}; usage={_safe_json_dumps(usage, 800)}"
                            )
                        # Immediate retry (no backoff needed here, but keep it consistent)
                    else:
                        errors.append("empty_assistant_content")
                        if self.cfg.debug:
                            usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
                            self._debug_print(
                                f"attempt {attempt+1}: empty content; usage={_safe_json_dumps(usage, 800)}"
                            )

                    # As a last resort, if json_mode might contribute, disable it after a couple empties.
                    if use_json_mode and errors.count("empty_assistant_content") >= 2:
                        use_json_mode = False
                        errors.append("disabled_json_mode_after_repeated_empty")

                    if attempt < self.cfg.max_retries - 1:
                        sleep_s = (self.cfg.backoff_base * (2 ** attempt)) + random.uniform(
                            0.0, self.cfg.backoff_jitter
                        )
                        await asyncio.sleep(sleep_s)
                    continue

                meta = {
                    "attempts": attempt + 1,
                    "errors": errors,
                    "used_json_mode": use_json_mode,
                    "max_tokens": int(cur_max_tokens),
                    "upgraded_budget": bool(upgraded_budget),
                }
                return assistant_text, resp_json, meta

            except Exception as e:
                err_s = repr(e)
                errors.append(err_s)

                # If JSON mode is rejected by gateway, disable and retry.
                low = err_s.lower()
                if use_json_mode and ("response_format" in low or "json_object" in low):
                    use_json_mode = False

                if self.cfg.debug:
                    self._debug_print(f"attempt {attempt+1} failed:", err_s)
                    traceback.print_exc()

                if attempt < self.cfg.max_retries - 1:
                    sleep_s = (self.cfg.backoff_base * (2 ** attempt)) + random.uniform(
                        0.0, self.cfg.backoff_jitter
                    )
                    await asyncio.sleep(sleep_s)

        meta_final = {
            "attempts": int(self.cfg.max_retries),
            "errors": errors,
            "used_json_mode": use_json_mode,
            "max_tokens": int(cur_max_tokens),
            "upgraded_budget": bool(upgraded_budget),
            "last_resp_json_preview": _safe_json_dumps(last_resp_json, 2000) if last_resp_json else "",
        }
        return "", last_resp_json, meta_final

    # -----------------------------------------------------------------
    # Public scoring API
    # -----------------------------------------------------------------
    async def score_pair(
        self,
        explanation: str,
        without_text: str,
        after_text: str,
    ) -> Tuple[float, float, bool]:
        """
        Score (without_text, after_text) in a single request.

        Returns:
          (score_without, score_after, ok)

        ok=True only if JSON was parsed and both keys exist.
        """
        if self.cfg.sanitize_text:
            explanation = _sanitize_text(explanation)
            without_text = _sanitize_text(without_text)
            after_text = _sanitize_text(after_text)

        messages = self._build_pair_scoring_messages(explanation, without_text, after_text)
        raw_text, resp_json, meta = await self._chat_completion(messages)

        record: Dict[str, Any] = {
            "ts_utc": _utc_ts(),
            "model": self.cfg.model,
            "base_url": self.cfg.base_url,
            "timeout": float(self.cfg.timeout),
            "ok": False,
            "parse_mode": None,
            "meta": meta,
            "input": {
                "explanation": _truncate_for_log(explanation, self.cfg.log_truncate_chars),
                "without_text": _truncate_for_log(without_text, self.cfg.log_truncate_chars),
                "after_text": _truncate_for_log(after_text, self.cfg.log_truncate_chars),
                "messages": messages if self.cfg.log_truncate_chars == 0 else None,
            },
            "output": {
                "assistant_text": raw_text,
                "resp_json_keys": list(resp_json.keys()) if isinstance(resp_json, dict) else None,
                "resp_json_preview": _safe_json_dumps(resp_json, 2000) if resp_json else "",
            },
            "extracted": {
                "score_without": float(self.cfg.fallback_score),
                "score_after": float(self.cfg.fallback_score),
            },
        }

        if not raw_text.strip():
            record["parse_mode"] = "empty"
            self._append_debug_log(record)
            return float(self.cfg.fallback_score), float(self.cfg.fallback_score), False

        try:
            obj = self._extract_first_json_object(raw_text)
            s_wo = self._coerce_score(obj.get("score_without"), self.cfg.fallback_score)
            s_af = self._coerce_score(obj.get("score_after"), self.cfg.fallback_score)

            ok = ("score_without" in obj) and ("score_after" in obj)

            record["parse_mode"] = "json"
            record["ok"] = bool(ok)
            record["extracted"] = {"score_without": s_wo, "score_after": s_af}
            self._append_debug_log(record)

            return s_wo, s_af, bool(ok)

        except Exception:
            record["parse_mode"] = "json_parse_error"
            self._append_debug_log(record)
            return float(self.cfg.fallback_score), float(self.cfg.fallback_score), False

    async def score_pairs(
        self,
        explanation: str,
        pairs: List[Tuple[str, str]],
    ) -> List[Tuple[float, float, bool]]:
        """
        Score many (without_text, after_text) pairs concurrently.
        """
        tasks = [self.score_pair(explanation, wo, af) for (wo, af) in pairs]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------
_global_judge: Optional[AsyncOpenAIJudge] = None


def get_global_judge() -> AsyncOpenAIJudge:
    """
    Lazy-init a global judge instance (reused across the whole run).
    """
    global _global_judge  # noqa: PLW0603
    if _global_judge is None:
        cfg = OpenAIJudgeConfig()
        _global_judge = AsyncOpenAIJudge(cfg)
    return _global_judge
