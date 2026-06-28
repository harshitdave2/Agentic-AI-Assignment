"""
llm/base.py — LLM Abstraction Layer.

Defines the two shared contracts used throughout the agent:

  AgentPlan    — the structured output every planner must produce.
  LLMProvider  — the abstract interface every planner must implement.

Why this abstraction exists
─────────────────────────────
The deterministic planner and any LLM-backed planner (Gemini, Claude, OpenAI…)
both produce an AgentPlan. The executor consumes only AgentPlan — it never
knows or cares which planner produced it. This means:

  • Switching from the deterministic planner to Gemini requires zero changes
    to executor.py or responder.py.
  • Adding a new LLM provider requires only a new file in this package.
  • Business logic is completely isolated from provider-specific code.

Adding a new LLM provider
──────────────────────────
1. Create  llm/your_provider.py
2. Subclass LLMProvider
3. Implement plan(question, preprocessed) → AgentPlan
4. Register it in agent/planner.py::get_planner()

No other changes needed anywhere in the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# ─── Intent Constants ─────────────────────────────────────────────────────────
#
# Using named constants instead of raw strings eliminates typo bugs and makes
# intent comparisons grep-able across the whole codebase.

INTENT_ORDER_STATUS        = "order_status"
INTENT_CHEAPER_ALTERNATIVE = "cheaper_alternative"
INTENT_PRODUCT_SEARCH      = "product_search"
INTENT_PRODUCT_DETAIL      = "product_detail"
INTENT_UNKNOWN             = "unknown"

ALL_INTENTS: frozenset[str] = frozenset({
    INTENT_ORDER_STATUS,
    INTENT_CHEAPER_ALTERNATIVE,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_DETAIL,
    INTENT_UNKNOWN,
})

# ─── Tool Name Constants ──────────────────────────────────────────────────────

TOOL_GET_ORDER       = "get_order"
TOOL_GET_PRODUCT     = "get_product"
TOOL_SEARCH_PRODUCTS = "search_products"

ALL_TOOLS: frozenset[str] = frozenset({
    TOOL_GET_ORDER,
    TOOL_GET_PRODUCT,
    TOOL_SEARCH_PRODUCTS,
})


# ─── AgentPlan ────────────────────────────────────────────────────────────────

@dataclass
class AgentPlan:
    """
    The structured contract between every planner and the executor.

    Any planner — deterministic or LLM-backed — must return an AgentPlan.
    The executor reads this plan to decide which tools to call and in what
    order. This decoupling is the core of the hybrid architecture.

    Fields
    ──────
    intent : str
        Classified intent. One of the INTENT_* constants defined above.

    entities : dict
        Extracted entities. Recognised keys:
          - "order_id"    : str | None   e.g. "ORD-1002"
          - "product_id"  : str | None   e.g. "PROD-201"
          - "search_query": str | None   e.g. "wireless headphones"

    reasoning : str
        One-sentence explanation of how this plan was derived.
        Useful for logging, debugging, and interview explainability.

    confidence : float
        Planner's confidence in the classification (0.0 – 1.0).
        Deterministic planner uses fixed values (0.85 – 1.0).
        LLM planner uses the model's self-reported confidence.

    recommended_tool_sequence : list[str]
        Ordered list of tool names the planner recommends.
        The executor uses intent-to-pipeline mapping and does NOT
        blindly follow this list — it is informational and for logging.
    """

    intent: str
    entities: dict
    reasoning: str
    confidence: float
    recommended_tool_sequence: list[str] = field(default_factory=list)

    def to_legacy_intents(self) -> dict:
        """
        Convert to the legacy intents dict format for backward compatibility.

        This ensures all 25 existing tests continue to pass without any
        modification. New code should work with AgentPlan directly.
        """
        e = self.entities
        return {
            "order_id":                  e.get("order_id"),
            "product_id":                e.get("product_id"),
            "wants_order_status":        self.intent == INTENT_ORDER_STATUS,
            "wants_cheaper_alternative": self.intent == INTENT_CHEAPER_ALTERNATIVE,
            "wants_product_search":      self.intent == INTENT_PRODUCT_SEARCH,
            "search_query":              e.get("search_query"),
        }


# ─── LLMProvider Interface ────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract base class for all planners — deterministic and LLM-based.

    Both DeterministicPlanner and GeminiProvider implement this interface.
    The agent's orchestrator calls get_planner() which returns one of them,
    and then calls planner.plan() — without knowing which planner it got.

    To add a new LLM provider:
    ──────────────────────────
    1. Create llm/your_provider.py
    2. class YourProvider(LLMProvider):
    3. Implement plan(question, preprocessed) → AgentPlan
    4. Register in agent/planner.py::get_planner()

    That is the complete extension protocol. No changes needed elsewhere.
    """

    @abstractmethod
    def plan(self, question: str, preprocessed: dict) -> AgentPlan:
        """
        Analyse the customer question and return a structured plan.

        Parameters
        ──────────
        question : str
            The raw customer question.
        preprocessed : dict
            Pre-extracted entities from the preprocessor stage.
            Keys: order_id, product_id, cleaned_question, raw_question.
            Use these directly — do NOT re-extract what is already available.

        Returns
        ───────
        AgentPlan
            Structured plan consumed by the executor.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
