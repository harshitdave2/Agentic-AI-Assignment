"""
llm/gemini_provider.py — Gemini LLM Provider.

Uses Google's Gemini model (gemini-1.5-flash) for intent classification
and entity understanding. The model produces a structured JSON AgentPlan.
Tool execution remains entirely deterministic in the executor.

Design philosophy
─────────────────
The LLM does ONLY reasoning:
  ✓ intent classification
  ✓ query understanding
  ✓ entity enrichment
  ✓ confidence scoring

The LLM does NOT:
  ✗ call tools
  ✗ access the database
  ✗ control execution flow
  ✗ produce the final customer response

This separation guarantees predictable, auditable behaviour even when
an LLM is in the loop.

Setup
─────
    pip install google-generativeai
    export GEMINI_API_KEY="your-key-here"
    export LLM_PROVIDER="gemini"

Failure handling
────────────────
If Gemini is unavailable (bad key, quota exceeded, network error), the
plan() method logs a warning and returns a safe "unknown" plan. The
calling code in get_planner() also falls back to DeterministicPlanner
if GeminiProvider cannot be initialised at all.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from llm.base import (
    AgentPlan,
    LLMProvider,
    INTENT_ORDER_STATUS,
    INTENT_CHEAPER_ALTERNATIVE,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_DETAIL,
    INTENT_UNKNOWN,
    TOOL_GET_ORDER,
    TOOL_GET_PRODUCT,
    TOOL_SEARCH_PRODUCTS,
    ALL_INTENTS,
    ALL_TOOLS,
)

logger = logging.getLogger("agent.llm.gemini")

# ─── System Prompt ────────────────────────────────────────────────────────────
#
# The system prompt instructs Gemini to act as a query understanding module,
# NOT as a conversational agent. It must return structured JSON only.

_SYSTEM_PROMPT = """
You are a query understanding module for an AI customer support agent for an online store.

Your ONLY job is to classify the customer's intent and extract entities.
You do NOT answer the question directly.
You do NOT execute any tools.
You produce a structured JSON plan that will be passed to a separate execution engine.

Available intents:
- order_status         : Customer wants to know about an order (status, delivery, tracking).
- cheaper_alternative  : Customer wants a cheaper alternative to an item in an existing order.
- product_search       : Customer wants to search for or browse products.
- product_detail       : Customer wants details about a specific product by ID (e.g. PROD-201).
- unknown              : None of the above applies.

Available tools (for context only — you do NOT call them):
- get_order       : Fetches order details by order ID.
- get_product     : Fetches product details by product ID.
- search_products  : Searches products by keyword.

Important:
- You will be given pre-extracted entities. Trust them — do NOT re-extract order/product IDs.
- Be conservative with confidence scores. Only use > 0.9 when very certain.

Respond ONLY with a valid JSON object in this exact format (no markdown, no explanation):
{
  "intent": "<one of the intent names above>",
  "entities": {
    "order_id":    "<ORD-XXXX or null>",
    "product_id":  "<PROD-XXXX or null>",
    "search_query": "<search terms or null>"
  },
  "reasoning": "<one concise sentence explaining your classification>",
  "confidence": <float between 0.0 and 1.0>,
  "recommended_tool_sequence": ["<tool1>", "<tool2>"]
}
""".strip()


class GeminiProvider(LLMProvider):
    """
    LLM planner backed by Google Gemini (gemini-1.5-flash).

    Gemini performs intent classification and entity understanding only.
    It never executes tools or controls execution flow.

    The response_mime_type="application/json" generation config ensures
    Gemini returns parseable JSON consistently.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialise the Gemini client.

        Parameters
        ──────────
        api_key : str
            Google AI Studio API key.

        Raises
        ──────
        ImportError   if google-generativeai is not installed.
        Exception     if the API key is invalid or configuration fails.
        """
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("[GEMINI] GeminiProvider initialised (gemini-1.5-flash).")

    def plan(self, question: str, preprocessed: dict) -> AgentPlan:
        """
        Call Gemini to produce an AgentPlan.

        The preprocessed entities are injected into the prompt so Gemini
        does not waste tokens re-extracting obvious identifiers like ORD-1002.

        Falls back to a safe INTENT_UNKNOWN plan if the API call fails or
        if the response cannot be parsed as valid JSON.
        """
        user_message = (
            f"Customer question: {question!r}\n\n"
            f"Pre-extracted entities (use these directly):\n"
            f"  order_id   : {preprocessed.get('order_id')!r}\n"
            f"  product_id : {preprocessed.get('product_id')!r}\n"
            f"  cleaned_question: {preprocessed.get('cleaned_question')!r}"
        )

        try:
            response = self._model.generate_content(
                [_SYSTEM_PROMPT, user_message],
                generation_config={"response_mime_type": "application/json"},
            )
            raw = response.text.strip()
            logger.debug(f"[GEMINI] Raw response: {raw}")
            data = json.loads(raw)
            plan = self._parse_plan(data, preprocessed)
            logger.info(
                f"[GEMINI] Plan: intent={plan.intent!r} "
                f"confidence={plan.confidence:.2f} reasoning={plan.reasoning!r}"
            )
            return plan

        except Exception as exc:
            logger.warning(
                f"[GEMINI] Planning failed ({type(exc).__name__}: {exc}). "
                f"Falling back to DeterministicPlanner."
            )
            from agent.planner import DeterministicPlanner
            return DeterministicPlanner().plan(question, preprocessed)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_plan(self, data: dict, preprocessed: dict) -> AgentPlan:
        """Parse and validate the JSON dict returned by Gemini."""
        intent = data.get("intent", INTENT_UNKNOWN)
        if intent not in ALL_INTENTS:
            logger.warning(f"[GEMINI] Unknown intent {intent!r}; falling back to unknown.")
            intent = INTENT_UNKNOWN

        raw_entities = data.get("entities") or {}
        entities = {
            # Always prefer pre-extracted IDs — they are regex-certain
            "order_id":    preprocessed.get("order_id") or raw_entities.get("order_id"),
            "product_id":  preprocessed.get("product_id") or raw_entities.get("product_id"),
            "search_query": raw_entities.get("search_query"),
        }

        # Only keep tool names we actually recognise
        raw_tools = data.get("recommended_tool_sequence") or []
        tools = [t for t in raw_tools if t in ALL_TOOLS]

        confidence = float(data.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        return AgentPlan(
            intent=intent,
            entities=entities,
            reasoning=str(data.get("reasoning", "Classified by Gemini.")),
            confidence=confidence,
            recommended_tool_sequence=tools,
        )

    def _unknown_plan(self, preprocessed: dict) -> AgentPlan:
        """Return a safe fallback plan when Gemini fails."""
        return AgentPlan(
            intent=INTENT_UNKNOWN,
            entities={
                "order_id":    preprocessed.get("order_id"),
                "product_id":  preprocessed.get("product_id"),
                "search_query": None,
            },
            reasoning="Gemini unavailable; returned unknown plan.",
            confidence=0.0,
            recommended_tool_sequence=[],
        )
