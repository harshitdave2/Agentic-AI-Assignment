"""
agent/executor.py — Tool execution and context management.

Two things live here:

  ExecutionContext
  ────────────────
  A dataclass that acts as the shared state for a single agent turn.
  It collects every tool result, error, timing measurement, and metadata
  in one place. The responder reads from this object to build the response.

  execute(plan, preprocessed) → ExecutionContext
  ───────────────────────────────────────────────
  Maps AgentPlan intents to deterministic, predefined execution pipelines.
  Execution order is ALWAYS deterministic — the LLM can suggest tool
  sequences, but the executor maps intents to fixed pipelines. This
  guarantees predictable, auditable behaviour.

  Intent → Pipeline mapping
  ─────────────────────────
    order_status        → get_order
    cheaper_alternative → get_order → get_product → search_products
    product_detail      → get_product
    product_search      → search_products
    unknown             → (no tools called)

  Why no dynamic dispatch?
  ─────────────────────────
  Allowing an LLM to invent arbitrary execution order at runtime would make
  the system unpredictable and hard to test. Predefined pipelines ensure
  each intent always executes the same tools in the same order, making the
  agent's behaviour fully verifiable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from llm.base import (
    AgentPlan,
    INTENT_ORDER_STATUS,
    INTENT_CHEAPER_ALTERNATIVE,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_DETAIL,
)
from tools.store_tools import get_order, get_product, search_products

logger = logging.getLogger("agent.executor")


# ─── Execution Context ────────────────────────────────────────────────────────

@dataclass
class ExecutionContext:
    """
    Shared state for a single agent turn.

    Created by execute() and consumed by build_response().
    Stores all tool outputs, errors, and diagnostic information so that:
      • The responder has everything it needs in one object.
      • Debugging is easy — one object captures the full execution history.
      • Logging is comprehensive — timing and tool call sequence are recorded.

    Fields
    ──────
    question        : original customer question
    plan            : the AgentPlan that drove this execution
    order           : result of get_order(), or None
    ordered_product : result of get_product() on the item in an order
    product_detail  : result of get_product() for a direct product lookup
    search_results  : list of products from search_products()
    errors          : list of human-readable error messages
    tool_calls      : ordered list of tools that were actually called
    execution_time_ms: total wall-clock time for all tool calls (ms)
    metadata        : arbitrary key-value pairs for debugging/extension
    """

    question:           str
    plan:               AgentPlan
    order:              Optional[dict]  = None
    ordered_product:    Optional[dict]  = None
    product_detail:     Optional[dict]  = None
    search_results:     list[dict]      = field(default_factory=list)
    errors:             list[str]       = field(default_factory=list)
    tool_calls:         list[str]       = field(default_factory=list)
    execution_time_ms:  float           = 0.0
    metadata:           dict            = field(default_factory=dict)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def has_errors(self) -> bool:
        """True when at least one error was recorded during execution."""
        return bool(self.errors)

    @property
    def has_results(self) -> bool:
        """True when at least one tool returned usable data."""
        return bool(self.order or self.product_detail or self.search_results or self.ordered_product)

    # ── Recording helpers ──────────────────────────────────────────────────

    def record_tool_call(self, tool_name: str) -> None:
        """Append a tool name to the call history and log it."""
        self.tool_calls.append(tool_name)
        logger.info(f"[EXECUTOR] Calling tool: {tool_name!r}")

    def record_error(self, message: str) -> None:
        """Append a human-readable error and log a warning."""
        self.errors.append(message)
        logger.warning(f"[EXECUTOR] Error recorded: {message!r}")


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def execute(plan: AgentPlan, preprocessed: dict) -> ExecutionContext:
    """
    Execute the plan using predefined, deterministic tool pipelines.

    Parameters
    ──────────
    plan : AgentPlan
        The plan produced by the planner.
    preprocessed : dict
        Pre-extracted entities (order_id, product_id, cleaned_question, raw_question).

    Returns
    ───────
    ExecutionContext
        Fully populated context with all tool results, errors, and timing.
    """
    ctx   = ExecutionContext(question=preprocessed["raw_question"], plan=plan)
    start = time.perf_counter()

    try:
        if plan.intent == INTENT_ORDER_STATUS:
            _pipeline_order_status(plan, ctx)

        elif plan.intent == INTENT_CHEAPER_ALTERNATIVE:
            _pipeline_cheaper_alternative(plan, ctx)

        elif plan.intent == INTENT_PRODUCT_DETAIL:
            _pipeline_product_detail(plan, ctx)

        elif plan.intent == INTENT_PRODUCT_SEARCH:
            _pipeline_product_search(plan, ctx)

        # INTENT_UNKNOWN: no tools called. The responder handles the fallback.

    except Exception as exc:
        logger.exception(f"[EXECUTOR] Unexpected error: {exc}")
        ctx.record_error(f"An unexpected error occurred during tool execution: {exc}")

    ctx.execution_time_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[EXECUTOR] Finished in {ctx.execution_time_ms:.1f}ms | "
        f"tools={ctx.tool_calls} | errors={ctx.errors}"
    )
    return ctx


# ─── Predefined Execution Pipelines ──────────────────────────────────────────

def _pipeline_order_status(plan: AgentPlan, ctx: ExecutionContext) -> None:
    """
    Pipeline: order_status
    ─────────────────────
    get_order(order_id)
    """
    order_id = plan.entities.get("order_id")
    ctx.record_tool_call("get_order")
    result = get_order(order_id)
    if "error" in result:
        ctx.record_error(result["error"])
    else:
        ctx.order = result


def _pipeline_cheaper_alternative(plan: AgentPlan, ctx: ExecutionContext) -> None:
    """
    Pipeline: cheaper_alternative
    ─────────────────────────────
    get_order(order_id)
      → get_product(first_item.product_id)
        → search_products(category)  [filtered: cheaper + in stock]

    Sequential because each step depends on the previous result.
    """
    order_id = plan.entities.get("order_id")

    # Step 1 — Fetch the order
    ctx.record_tool_call("get_order")
    order = get_order(order_id)
    if "error" in order:
        ctx.record_error(order["error"])
        return
    ctx.order = order

    # Step 2 — Fetch the ordered product (first item only)
    items = order.get("items", [])
    if not items:
        ctx.record_error("No items found in this order to find alternatives for.")
        return

    first_item = items[0]
    ctx.record_tool_call("get_product")
    product = get_product(first_item["product_id"])
    if "error" in product:
        ctx.record_error(product["error"])
        return
    ctx.ordered_product = product

    # Step 3 — Search for cheaper in-stock alternatives in the same category
    ctx.record_tool_call("search_products")
    all_in_category = search_products(product["category"])
    cheaper = [
        p for p in all_in_category
        if p["price"]      <  product["price"]
        and p["product_id"] != product["product_id"]
        and p["in_stock"]
    ]
    ctx.search_results = sorted(cheaper, key=lambda p: p["price"])


def _pipeline_product_detail(plan: AgentPlan, ctx: ExecutionContext) -> None:
    """
    Pipeline: product_detail
    ────────────────────────
    get_product(product_id)
    """
    product_id = plan.entities.get("product_id")
    ctx.record_tool_call("get_product")
    result = get_product(product_id)
    if "error" in result:
        ctx.record_error(result["error"])
    else:
        ctx.product_detail = result


def _pipeline_product_search(plan: AgentPlan, ctx: ExecutionContext) -> None:
    """
    Pipeline: product_search
    ────────────────────────
    search_products(search_query)
    """
    query = plan.entities.get("search_query", "")
    ctx.record_tool_call("search_products")
    ctx.search_results = search_products(query)


# ─── Backward-Compatible Legacy Interface ─────────────────────────────────────
#
# These wrappers allow the existing tests in test_agent.py to call
# execute_plan(intents_dict) → dict without any modification.

def execute_legacy(intents: dict) -> dict:
    """
    Backward-compatible wrapper for the old execute_plan(intents: dict) → dict API.

    Converts a legacy intents dict to an AgentPlan, runs execute(), then
    converts the ExecutionContext back to the legacy ctx dict format.

    Used exclusively by agent.py::execute_plan() to preserve existing tests.
    New code should call execute(plan, preprocessed) directly.
    """
    # ── Convert legacy intents dict → AgentPlan ────────────────────────────
    if intents.get("wants_cheaper_alternative") and intents.get("order_id"):
        intent = INTENT_CHEAPER_ALTERNATIVE
    elif intents.get("order_id"):
        intent = INTENT_ORDER_STATUS
    elif intents.get("product_id") and not intents.get("order_id"):
        intent = INTENT_PRODUCT_DETAIL
    elif intents.get("wants_product_search") and intents.get("search_query"):
        intent = INTENT_PRODUCT_SEARCH
    else:
        from llm.base import INTENT_UNKNOWN
        intent = INTENT_UNKNOWN

    plan = AgentPlan(
        intent=intent,
        entities={
            "order_id":    intents.get("order_id"),
            "product_id":  intents.get("product_id"),
            "search_query": intents.get("search_query"),
        },
        reasoning="[Legacy compatibility bridge]",
        confidence=1.0,
        recommended_tool_sequence=[],
    )

    preprocessed = {
        "order_id":         intents.get("order_id"),
        "product_id":       intents.get("product_id"),
        "cleaned_question": "",
        "raw_question":     "",
    }

    ctx = execute(plan, preprocessed)

    # ── Convert ExecutionContext → legacy ctx dict ──────────────────────────
    return {
        "order":           ctx.order,
        "ordered_product": ctx.ordered_product,
        "search_results":  ctx.search_results,
        "product_detail":  ctx.product_detail,
        "errors":          ctx.errors,
    }
