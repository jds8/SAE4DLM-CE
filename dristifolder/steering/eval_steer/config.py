"""
Global configuration for SAE steering evaluation (no env vars required).
Edit this file only; do not rely on terminal exports.
"""

from __future__ import annotations
from pathlib import Path

# ----- Output configuration -----

EVAL_ROOT = Path("/home/dslabra5/sae4dlm/steering/eval_steering_file")

# ----- LLM judge configuration -----

OPENAI_BASE_URL_DEFAULT = "https://api.shubiaobiao.cn/v1"
MODEL_NAME = "gpt-4o-mini"
JUDGE_TIMEOUT = 60.0

# Debug prints to stdout
JUDGE_DEBUG = True

# Primary completion budget
JUDGE_MAX_TOKENS = 128

# If visible content is empty but reasoning_tokens consumed the entire budget,
# retry once with this larger budget.
JUDGE_MAX_TOKENS_ON_EMPTY = 256

# Concurrency (set to 1 to avoid 429)
JUDGE_MAX_CONCURRENCY = 1

# Retry policy
JUDGE_MAX_RETRIES = 6
JUDGE_BACKOFF_BASE = 0.8
JUDGE_BACKOFF_JITTER = 0.4

# Fallback score if judge fails / parse fails
JUDGE_FALLBACK_SCORE = 50.0

# Force JSON mode if supported by the gateway
JUDGE_USE_JSON_MODE = True

# Sanitize text to reduce content_filter errors
JUDGE_SANITIZE_TEXT = True

# ----- Judge debug logging (file) -----

JUDGE_LOG_FALLBACK = True
JUDGE_LOG_WHEN_BOTH_EQ_FALLBACK = True
JUDGE_LOG_DIR = Path("/home/dslabra5/sae4dlm/steering/eval_steer/logs")
JUDGE_LOG_PATH = JUDGE_LOG_DIR / "llm_judge_debug.jsonl"

# 0 means no truncation
JUDGE_LOG_TRUNCATE_CHARS = 0

# ----- Perplexity model configuration -----

PPL_MODEL_NAME = "gpt2"
PPL_MAX_LENGTH = 512
PPL_BATCH_SIZE = 16

# ----- Checkpointing -----

CHECKPOINT_EVERY_N_FEATURES = 5
