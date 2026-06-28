"""
agent/planner.py — Query preprocessing, intent planning, and plan validation.

Three responsibilities live here, in pipeline order:

  Stage 1 — Preprocessor
  ───────────────────────
  preprocess(question) extracts obvious entities (order IDs, product IDs)
  using regex before any planner runs. This removes trivial work from the LLM
  and produces a clean baseline that all planners can rely on.

  Stage 2 — DeterministicPlanner (fallback)
  ──────────────────────────────────────────
  A rule-based planner that uses keyword matching to classify intent and build
  an AgentPlan. It implements the LLMProvider interface so it is completely
  interchangeable with GeminiProvider. It always works — no API key, no
  external dependencies.

  Stage 3 — Plan Validator
  ────────────────────────
  validate_plan() performs lightweight pre-execution checks: does the order
  status plan actually have an order ID? Does the product search plan have a
  search query? Invalid plans are caught before any tool call is made.

  Planner Factory
  ───────────────
  get_planner() selects the appropriate planner based on config/settings.py.
  Selection order:
    1. GeminiProvider  (if LLM_PROVIDER=gemini AND GEMINI_API_KEY is set)
    2. DeterministicPlanner  (always available, zero dependencies)

  The agent NEVER crashes because an LLM is unavailable.
"""

from __future__ import annotations

import re
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
)
from config.settings import LLM_PROVIDER, GEMINI_API_KEY

logger = logging.getLogger("agent.planner")

# ─── Compiled Patterns ────────────────────────────────────────────────────────
#
# Pre-compiled for performance and to guarantee consistent matching across
# the preprocessor and the search-query builder.

ORDER_ID_PATTERN   = re.compile(r"\bORD[-_]?\d{3,6}\b",  re.IGNORECASE)
PRODUCT_ID_PATTERN = re.compile(r"\bPROD[-_]?\d{3,6}\b", re.IGNORECASE)

# ─── Intent Keywords ──────────────────────────────────────────────────────────

# Phrases that ONLY indicate the user wants a cheaper/alternative product
# AND must be combined with an ownership signal (order ID or "my order").
_ALTERNATIVE_PHRASES = [
    "cheaper alternative", "budget alternative", "lower price alternative",
    "less expensive alternative", "cheaper option", "budget option",
    "instead of", "similar to", "something similar", "something cheaper",
    "more affordable", "alternative to",
]

# Keywords that anchor ownership context (required for alternative intent).
_OWNERSHIP_KEYWORDS = [
    "i ordered", "i bought", "what i ordered", "what i bought",
    "in my order", "my order",
]

# Keywords that signal an ORDER STATUS query.
_ORDER_KEYWORDS = [
    "order", "delivery", "shipped", "tracking", "status",
    "where is", "when will", "arrive", "dispatched", "deliver",
]

# Keywords that signal a PRODUCT SEARCH query.
_SEARCH_KEYWORDS = [
    "find", "search", "looking for", "recommend", "suggest",
    "show me", "available", "buy", "purchase",
]

# Filler words stripped from product search queries.
# Sorted longest-first so multi-word phrases match before their components.
_SEARCH_FILLERS: list[str] = sorted(
    [
        "can you", "could you", "please", "find me", "show me",
        "i want", "i need", "looking for", "is there", "i'm", "i am",
        "some", "any", "the", "a",
        # Search adjectives that don't appear in product names/descriptions
        "affordable", "budget", "cheap", "inexpensive", "expensive",
        "best", "good", "great", "latest", "new", "premium",
        "find", "get", "buy", "search", "search for",
        "under budget", "within budget", "under", "help me", "help",
        "hello", "hi", "hey", "thanks", "thank you",
    ],
    key=len,
    reverse=True,
)


# ─── Stage 1: Preprocessor ────────────────────────────────────────────────────

def preprocess(question: str) -> dict:
    """
    Deterministic preprocessing — runs before any planner.

    Extracts structured entities using regex so that the planner (including
    the LLM) does not waste effort identifying obvious identifiers like
    "ORD-1002" or "PROD-201".

    Parameters
    ──────────
    question : str
        Raw customer question.

    Returns
    ───────
    dict
        order_id        : str | None  — normalised to "ORD-XXXX"
        product_id      : str | None  — normalised to "PROD-XXX"
        cleaned_question: str         — question with IDs removed
        raw_question    : str         — original unmodified question
    """
    order_match   = ORDER_ID_PATTERN.search(question)
    product_match = PRODUCT_ID_PATTERN.search(question)

    order_id   = order_match.group(0).upper().replace("_", "-")   if order_match   else None
    product_id = product_match.group(0).upper().replace("_", "-") if product_match else None

    # Produce a cleaned version for LLM context and search queries
    cleaned = re.sub(ORDER_ID_PATTERN,   "", question)
    cleaned = re.sub(PRODUCT_ID_PATTERN, "", cleaned)
    cleaned = " ".join(cleaned.split()).strip()

    logger.debug(
        f"[PREPROCESS] order_id={order_id!r} product_id={product_id!r} "
        f"cleaned={cleaned!r}"
    )
    return {
        "order_id":          order_id,
        "product_id":        product_id,
        "cleaned_question":  cleaned,
        "raw_question":      question,
    }


# ─── Stage 2: Deterministic Planner ──────────────────────────────────────────

class DeterministicPlanner(LLMProvider):
    """
    Rule-based planner — the default and fallback planner.

    Uses regex and keyword matching to classify intent and build an AgentPlan.
    Implements the LLMProvider interface so it is fully interchangeable with
    GeminiProvider; the rest of the codebase never needs to know which one
    it is talking to.

    Confidence values are deterministic fixed scores, not probabilities:
      1.00 — product detail (PROD-XXX present, unambiguous)
      0.95 — order status   (order keyword or ORD-XXX present)
      0.90 — alt search     (alternative phrase + ownership signal)
      0.85 — product search (search keyword detected)
      0.50 — unknown        (nothing matched)
    """

    def plan(self, question: str, preprocessed: dict) -> AgentPlan:
        """
        Classify intent and return a structured AgentPlan.

        Uses the pre-extracted entities from preprocessed to avoid repeating
        the regex work. Builds a search query for product search intents.
        """
        q          = question.lower()
        order_id   = preprocessed.get("order_id")
        product_id = preprocessed.get("product_id")

        # ── Detect signals ─────────────────────────────────────────────────
        has_alternative_phrase = any(phrase in q for phrase in _ALTERNATIVE_PHRASES)
        has_ownership_context  = any(kw in q for kw in _OWNERSHIP_KEYWORDS)
        has_order_keyword      = any(kw in q for kw in _ORDER_KEYWORDS)
        has_search_keyword     = any(kw in q for kw in _SEARCH_KEYWORDS)

        # ── Classify intent (priority order matters) ───────────────────────
        #
        # 1. Cheaper alternative — most specific, checked first
        wants_cheaper_alternative = (
            has_alternative_phrase and (bool(order_id) or has_ownership_context)
        )

        # 2. Direct product-ID lookup — PROD-XXX present, no order context
        is_product_id_lookup = bool(product_id) and not bool(order_id)

        # 3. Order status — order keyword or ORD-XXX present
        wants_order_status = has_order_keyword or bool(order_id)

        # 4. Product search — search keyword or general noun query
        wants_product_search = (
            not wants_cheaper_alternative
            and not wants_order_status
            and (has_search_keyword or is_product_id_lookup or len(q.split()) >= 2)
        )
        # Allow search alongside non-order contexts
        if not wants_order_status and has_search_keyword:
            wants_product_search = True

        # ── Build AgentPlan ────────────────────────────────────────────────
        if wants_cheaper_alternative:
            return AgentPlan(
                intent=INTENT_CHEAPER_ALTERNATIVE,
                entities={
                    "order_id":    order_id,
                    "product_id":  product_id,
                    "search_query": None,
                },
                reasoning=(
                    "Explicit alternative phrase detected with order ownership context."
                ),
                confidence=0.90,
                recommended_tool_sequence=[
                    TOOL_GET_ORDER, TOOL_GET_PRODUCT, TOOL_SEARCH_PRODUCTS
                ],
            )

        if is_product_id_lookup:
            return AgentPlan(
                intent=INTENT_PRODUCT_DETAIL,
                entities={
                    "order_id":    None,
                    "product_id":  product_id,
                    "search_query": None,
                },
                reasoning=f"Direct product-ID lookup for {product_id}.",
                confidence=1.00,
                recommended_tool_sequence=[TOOL_GET_PRODUCT],
            )

        if wants_order_status:
            return AgentPlan(
                intent=INTENT_ORDER_STATUS,
                entities={
                    "order_id":    order_id,
                    "product_id":  product_id,
                    "search_query": None,
                },
                reasoning="Order keyword or order ID detected; fetching order status.",
                confidence=0.95,
                recommended_tool_sequence=[TOOL_GET_ORDER],
            )

        if wants_product_search:
            search_query = self._build_search_query(question)
            if search_query:
                return AgentPlan(
                    intent=INTENT_PRODUCT_SEARCH,
                    entities={
                        "order_id":    None,
                        "product_id":  None,
                        "search_query": search_query,
                    },
                    reasoning="Search keyword detected; performing product catalog search.",
                    confidence=0.85,
                    recommended_tool_sequence=[TOOL_SEARCH_PRODUCTS],
                )

        # ── Fallback ───────────────────────────────────────────────────────
        return AgentPlan(
            intent=INTENT_UNKNOWN,
            entities={
                "order_id":    order_id,
                "product_id":  product_id,
                "search_query": None,
            },
            reasoning="No recognisable intent found; returning fallback guidance.",
            confidence=0.50,
            recommended_tool_sequence=[],
        )

    def _build_search_query(self, question: str) -> Optional[str]:
        """
        Strip order/product IDs and filler words to produce a clean search term.

        For example:
          "Find me affordable running shoes" → "running shoes"
          "Show me wireless headphones"      → "wireless headphones"
          "I'm looking for jeans"            → "jeans"
        """
        clean = re.sub(ORDER_ID_PATTERN,   "", question)
        clean = re.sub(PRODUCT_ID_PATTERN, "", clean)

        for filler in _SEARCH_FILLERS:
            clean = re.sub(
                r"\b" + re.escape(filler) + r"\b", " ", clean, flags=re.IGNORECASE
            )

        query = " ".join(clean.split()).strip(" ?.,!")
        return query if query else None


# ─── Stage 3: Plan Validator ──────────────────────────────────────────────────

def validate_plan(plan: AgentPlan) -> tuple[bool, str]:
    """
    Lightweight pre-execution validation.

    Checks that the plan has the entities it needs to execute successfully.
    For example, an order_status plan without an order_id cannot proceed.
    Catching this here produces a better customer message than a tool error.

    Returns
    ───────
    (True,  "")       — plan is valid, executor should proceed
    (False, "<msg>")  — plan is invalid; msg is the customer-facing response
    """
    e = plan.entities

    if plan.intent == INTENT_ORDER_STATUS and not e.get("order_id"):
        return (
            False,
            "I need your order ID to look that up. "
            "Could you share it? (e.g. ORD-1002)",
        )

    if plan.intent == INTENT_CHEAPER_ALTERNATIVE and not e.get("order_id"):
        return (
            False,
            "To find cheaper alternatives, I need your order ID. "
            "Could you share it? (e.g. ORD-1001)",
        )

    if plan.intent == INTENT_PRODUCT_DETAIL and not e.get("product_id"):
        return (
            False,
            "I need a product ID to fetch those details. (e.g. PROD-201)",
        )

    if plan.intent == INTENT_PRODUCT_SEARCH and not e.get("search_query"):
        return (
            False,
            "I didn't catch what you're looking for. "
            "Could you describe the product?",
        )

    return True, ""


_planner_instance: Optional[LLMProvider] = None

def get_planner() -> LLMProvider:
    """
    Return the appropriate planner based on current configuration.

    Selection order
    ───────────────
    1. GeminiProvider  — when LLM_PROVIDER="gemini" and GEMINI_API_KEY is set
    2. DeterministicPlanner — always available, zero external dependencies

    The agent NEVER crashes because an LLM is unavailable. If GeminiProvider
    raises any exception during initialisation, the deterministic planner is
    returned automatically.
    """
    global _planner_instance
    if _planner_instance is not None:
        return _planner_instance
        
    logger.info("Loaded configuration...")
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        try:
            from llm.gemini_provider import GeminiProvider
            provider = GeminiProvider(GEMINI_API_KEY)
            logger.info("Selected planner: GeminiProvider")
            logger.info("Gemini initialized successfully.")
            _planner_instance = provider
            return provider
        except Exception as exc:
            logger.warning(
                f"Fallback reason: GeminiProvider init failed ({exc}). "
                f"Falling back to DeterministicPlanner."
            )

    logger.info("Selected planner: DeterministicPlanner")
    _planner_instance = DeterministicPlanner()
    return _planner_instance
