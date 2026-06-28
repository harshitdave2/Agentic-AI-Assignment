"""
agent/agent.py — Agent orchestrator.

This module is intentionally thin. Its only job is to wire the pipeline
together and expose the public run_agent() entry point.

Execution pipeline
──────────────────

  User Question
       │
       ▼
  ┌─────────────┐
  │ Preprocessor│  [agent/planner.py]  Regex entity extraction
  └──────┬──────┘
         │ preprocessed dict (order_id, product_id, cleaned_question)
         ▼
  ┌─────────────┐
  │   Planner   │  [agent/planner.py]  Intent classification
  └──────┬──────┘  (DeterministicPlanner or GeminiProvider)
         │ AgentPlan (intent, entities, reasoning, confidence)
         ▼
  ┌───────────────┐
  │ Plan Validator│  [agent/planner.py]  Checks required entities exist
  └──────┬────────┘
         │ (True, "") or (False, customer_message)
         ▼
  ┌──────────────┐
  │   Executor   │  [agent/executor.py]  Deterministic tool pipelines
  └──────┬───────┘
         │ ExecutionContext (tool outputs, errors, timing)
         ▼
  ┌──────────────┐
  │  Responder   │  [agent/responder.py]  Customer-friendly formatting
  └──────┬───────┘
         │
         ▼
    Final Response (str)

Why each stage is separate
──────────────────────────
• Preprocessor  — LLMs should not waste tokens on regex-trivial extraction.
• Planner       — Separating WHAT from HOW enables LLM/deterministic swap.
• Validator     — Catching bad plans before tool calls improves reliability.
• Executor      — Deterministic pipelines guarantee predictable behaviour.
• Responder     — Presentation must never mix with business logic.

Backward compatibility
──────────────────────
detect_intents() and execute_plan() are preserved here so that all 25
existing tests in tests/test_agent.py continue to pass without modification.
"""

import os
import sys
import logging
import time

from agent.planner  import preprocess, get_planner, validate_plan, DeterministicPlanner
from agent.executor import execute, execute_legacy
from agent.responder import build_response

# ─── Logging Setup ────────────────────────────────────────────────────────────
# Create the logs/ directory before configuring the FileHandler so that fresh
# clones (which have no logs/ directory) never crash on startup.
#
# Windows cp1252 consoles cannot encode many Unicode characters (arrows, rupee
# symbol, box-drawing chars). We fix this at the root:
#   - FileHandler: always UTF-8 so logs on disk are complete.
#   - StreamHandler: reconfigure stdout to UTF-8 where possible; otherwise use
#     'replace' error handler so unencodable chars become '?' instead of
#     raising UnicodeEncodeError.

os.makedirs("logs", exist_ok=True)

from config.settings import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_LEVEL  = getattr(logging, LOG_LEVEL, logging.INFO)

# File handler — always UTF-8.
_file_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

# Console handler — safe on Windows.
try:
    # Python 3.7+: reconfigure stdout to UTF-8 with 'replace' so that
    # any character that cannot be encoded becomes '?' instead of crashing.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass  # Already UTF-8 or not a TextIOWrapper (e.g. redirected)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

logging.basicConfig(
    level=_LOG_LEVEL,
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger("agent.core")


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run_agent(question: str) -> str:
    """
    Main agent function — the single entry point for all customer queries.

    Parameters
    ──────────
    question : str
        A customer's natural-language question.

    Returns
    ───────
    str
        A customer-friendly response. This function never raises an exception;
        all errors are caught and converted into a graceful message.
    """
    if not question or not question.strip():
        return "Please ask me something! For example: 'Where is my order ORD-1002?'"

    start = time.perf_counter()
    logger.info("-" * 60)
    logger.info(f"[AGENT] Question: {question!r}")

    try:
        # ── Stage 1: Preprocess ────────────────────────────────────────────
        preprocessed = preprocess(question)

        # ── Stage 2: Plan ──────────────────────────────────────────────────
        planner = get_planner()
        plan    = planner.plan(question, preprocessed)
        logger.info(
            f"[AGENT] Plan  : intent={plan.intent!r} | "
            f"confidence={plan.confidence:.2f} | "
            f"tools={plan.recommended_tool_sequence}"
        )
        logger.info(f"[AGENT] Reason: {plan.reasoning}")

        # ── Stage 3: Validate ──────────────────────────────────────────────
        valid, validation_msg = validate_plan(plan)
        if not valid:
            logger.info(f"[AGENT] Plan validation failed: {validation_msg!r}")
            return validation_msg

        # ── Stage 4: Execute ───────────────────────────────────────────────
        ctx = execute(plan, preprocessed)

        # ── Stage 5: Respond ───────────────────────────────────────────────
        response = build_response(question, plan, ctx)

    except Exception as exc:
        logger.exception(f"[AGENT] Unhandled exception: {exc}")
        response = (
            "I encountered an unexpected error while processing your request. "
            "Please try again or contact our support team."
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[AGENT] Response: {len(response)} chars in {elapsed_ms:.1f}ms")
    logger.info("-" * 60)
    return response


# ─── Backward-Compatible API ──────────────────────────────────────────────────
#
# These two functions preserve the exact interface expected by the 25 existing
# tests in tests/test_agent.py. They delegate to the new modular pipeline.
#
# New code should use run_agent() or the individual pipeline functions directly.

def detect_intents(question: str) -> dict:
    """
    Legacy interface: classify intent and return the original intents dict format.

    Preserved for backward compatibility with existing tests.
    Delegates to DeterministicPlanner + AgentPlan.to_legacy_intents().

    New code should use:  get_planner().plan(question, preprocess(question))
    """
    preprocessed = preprocess(question)
    plan         = DeterministicPlanner().plan(question, preprocessed)
    intents      = plan.to_legacy_intents()
    logger.info(f"[AGENT] detect_intents (legacy): {intents}")
    return intents


def execute_plan(intents: dict) -> dict:
    """
    Legacy interface: accept old-style intents dict, return old-style ctx dict.

    Preserved for backward compatibility with existing tests.
    Delegates to execute_legacy() which bridges the old and new data models.

    New code should use:  execute(plan, preprocessed)
    """
    return execute_legacy(intents)
