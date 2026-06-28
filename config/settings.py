"""
config/settings.py — Centralised configuration for the AI Store Agent.

All values are read from environment variables with safe defaults,
so the agent works out-of-the-box with no setup required.

Override any value by exporting the corresponding environment variable:

    export LLM_PROVIDER=gemini
    export GEMINI_API_KEY=your-key-here
    export LOG_LEVEL=DEBUG
"""

import os

# ─── API Keys ─────────────────────────────────────────────────────────────────
# Required only when LLM_PROVIDER="gemini".
# Obtain a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ─── LLM Provider ─────────────────────────────────────────────────────────────
# Controls which planner the agent uses for intent classification.
#
#   "deterministic"  (default) — rule-based planner, zero dependencies,
#                                always works, no API key needed.
#   "gemini"                   — Gemini-backed planner; requires GEMINI_API_KEY.
#                                Falls back to deterministic if key is missing.
#
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "deterministic")

# ─── Logging ──────────────────────────────────────────────────────────────────
# Controls logging verbosity.
# Accepted values: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
