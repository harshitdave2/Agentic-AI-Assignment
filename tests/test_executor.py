"""
tests/test_executor.py — Unit tests for agent/executor.py.

Covers:
  • ExecutionContext dataclass properties and methods
  • execute() for each intent/pipeline
  • Error handling in every pipeline branch
  • execute_legacy() backward-compatibility wrapper

Run:  python -m pytest tests/test_executor.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from llm.base import (
    AgentPlan,
    INTENT_ORDER_STATUS,
    INTENT_CHEAPER_ALTERNATIVE,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_DETAIL,
    INTENT_UNKNOWN,
)
from agent.executor import execute, execute_legacy, ExecutionContext


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_plan(intent: str, entities: dict | None = None) -> AgentPlan:
    return AgentPlan(
        intent=intent,
        entities=entities or {},
        reasoning="test",
        confidence=1.0,
        recommended_tool_sequence=[],
    )


def _preprocessed(
    order_id: str | None = None,
    product_id: str | None = None,
    raw_question: str = "test question",
) -> dict:
    return {
        "order_id":         order_id,
        "product_id":       product_id,
        "cleaned_question": "",
        "raw_question":     raw_question,
    }


# ─── ExecutionContext Tests ───────────────────────────────────────────────────

class TestExecutionContext:
    def _empty_ctx(self) -> ExecutionContext:
        plan = _make_plan(INTENT_UNKNOWN)
        return ExecutionContext(question="test", plan=plan)

    def test_has_errors_is_false_initially(self):
        ctx = self._empty_ctx()
        assert not ctx.has_errors

    def test_has_results_is_false_initially(self):
        ctx = self._empty_ctx()
        assert not ctx.has_results

    def test_record_error_sets_has_errors(self):
        ctx = self._empty_ctx()
        ctx.record_error("Something went wrong")
        assert ctx.has_errors

    def test_record_error_appends_message(self):
        ctx = self._empty_ctx()
        ctx.record_error("Error one")
        ctx.record_error("Error two")
        assert len(ctx.errors) == 2
        assert "Error one" in ctx.errors
        assert "Error two" in ctx.errors

    def test_record_tool_call_appends(self):
        ctx = self._empty_ctx()
        ctx.record_tool_call("get_order")
        ctx.record_tool_call("get_product")
        assert ctx.tool_calls == ["get_order", "get_product"]

    def test_has_results_true_when_order_set(self):
        ctx = self._empty_ctx()
        ctx.order = {"order_id": "ORD-1001"}
        assert ctx.has_results

    def test_has_results_true_when_product_detail_set(self):
        ctx = self._empty_ctx()
        ctx.product_detail = {"product_id": "PROD-201"}
        assert ctx.has_results

    def test_has_results_true_when_search_results_set(self):
        ctx = self._empty_ctx()
        ctx.search_results = [{"name": "Test Product"}]
        assert ctx.has_results

    def test_execution_time_defaults_to_zero(self):
        ctx = self._empty_ctx()
        assert ctx.execution_time_ms == 0.0

    def test_metadata_defaults_to_empty_dict(self):
        ctx = self._empty_ctx()
        assert ctx.metadata == {}

    def test_search_results_defaults_to_empty_list(self):
        ctx = self._empty_ctx()
        assert ctx.search_results == []

    def test_errors_defaults_to_empty_list(self):
        ctx = self._empty_ctx()
        assert ctx.errors == []


# ─── Execute — Order Status Pipeline ─────────────────────────────────────────

class TestExecuteOrderStatus:
    def test_valid_order_populates_ctx_order(self):
        plan = _make_plan(INTENT_ORDER_STATUS, {"order_id": "ORD-1002"})
        ctx  = execute(plan, _preprocessed(order_id="ORD-1002"))
        assert ctx.order is not None
        assert ctx.order["order_id"] == "ORD-1002"

    def test_valid_order_no_errors(self):
        plan = _make_plan(INTENT_ORDER_STATUS, {"order_id": "ORD-1002"})
        ctx  = execute(plan, _preprocessed(order_id="ORD-1002"))
        assert not ctx.has_errors

    def test_valid_order_calls_get_order(self):
        plan = _make_plan(INTENT_ORDER_STATUS, {"order_id": "ORD-1001"})
        ctx  = execute(plan, _preprocessed(order_id="ORD-1001"))
        assert "get_order" in ctx.tool_calls

    def test_invalid_order_id_records_error(self):
        plan = _make_plan(INTENT_ORDER_STATUS, {"order_id": "ORD-9999"})
        ctx  = execute(plan, _preprocessed(order_id="ORD-9999"))
        assert ctx.order is None
        assert ctx.has_errors
        assert len(ctx.errors) == 1

    def test_all_order_statuses_load(self):
        for oid in ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004"]:
            plan = _make_plan(INTENT_ORDER_STATUS, {"order_id": oid})
            ctx  = execute(plan, _preprocessed(order_id=oid))
            assert ctx.order is not None, f"Order {oid} should load"


# ─── Execute — Cheaper Alternative Pipeline ───────────────────────────────────

class TestExecuteCheaperAlternative:
    def _run(self, order_id: str = "ORD-1001") -> ExecutionContext:
        plan = _make_plan(
            INTENT_CHEAPER_ALTERNATIVE,
            {"order_id": order_id, "product_id": None, "search_query": None},
        )
        return execute(plan, _preprocessed(order_id=order_id))

    def test_populates_order(self):
        ctx = self._run()
        assert ctx.order is not None

    def test_populates_ordered_product(self):
        ctx = self._run()
        assert ctx.ordered_product is not None

    def test_calls_all_three_tools(self):
        ctx = self._run()
        assert "get_order"       in ctx.tool_calls
        assert "get_product"     in ctx.tool_calls
        assert "search_products" in ctx.tool_calls

    def test_tool_call_order_is_sequential(self):
        ctx = self._run()
        assert ctx.tool_calls.index("get_order") < ctx.tool_calls.index("get_product")
        assert ctx.tool_calls.index("get_product") < ctx.tool_calls.index("search_products")

    def test_alternatives_are_all_cheaper_than_original(self):
        ctx = self._run()
        if ctx.ordered_product and ctx.search_results:
            orig_price = ctx.ordered_product["price"]
            for alt in ctx.search_results:
                assert alt["price"] < orig_price, (
                    f"{alt['name']} (₹{alt['price']}) should be cheaper than "
                    f"₹{orig_price}"
                )

    def test_alternatives_are_all_in_stock(self):
        ctx = self._run()
        for alt in ctx.search_results:
            assert alt["in_stock"], f"{alt['name']} must be in stock"

    def test_alternatives_exclude_original_product(self):
        ctx = self._run()
        if ctx.ordered_product:
            orig_id = ctx.ordered_product["product_id"]
            alt_ids = [a["product_id"] for a in ctx.search_results]
            assert orig_id not in alt_ids

    def test_invalid_order_stops_pipeline(self):
        ctx = self._run("ORD-9999")
        assert ctx.order is None
        assert ctx.has_errors
        # Pipeline should stop after get_order fails — no get_product call
        assert "get_product" not in ctx.tool_calls

    def test_headphones_alternative(self):
        """ORD-1002 contains Sony WH-1000XM5 (₹24,999) — alternatives should be cheaper."""
        ctx = self._run("ORD-1002")
        if ctx.ordered_product:
            assert ctx.ordered_product["price"] == 24999
        if ctx.search_results:
            for alt in ctx.search_results:
                assert alt["price"] < 24999


# ─── Execute — Product Detail Pipeline ───────────────────────────────────────

class TestExecuteProductDetail:
    def test_valid_product_populates_product_detail(self):
        plan = _make_plan(INTENT_PRODUCT_DETAIL, {"product_id": "PROD-201"})
        ctx  = execute(plan, _preprocessed(product_id="PROD-201"))
        assert ctx.product_detail is not None
        assert ctx.product_detail["product_id"] == "PROD-201"

    def test_valid_product_calls_get_product(self):
        plan = _make_plan(INTENT_PRODUCT_DETAIL, {"product_id": "PROD-201"})
        ctx  = execute(plan, _preprocessed(product_id="PROD-201"))
        assert "get_product" in ctx.tool_calls

    def test_valid_product_does_not_call_search(self):
        plan = _make_plan(INTENT_PRODUCT_DETAIL, {"product_id": "PROD-201"})
        ctx  = execute(plan, _preprocessed(product_id="PROD-201"))
        assert "search_products" not in ctx.tool_calls

    def test_invalid_product_id_records_error(self):
        plan = _make_plan(INTENT_PRODUCT_DETAIL, {"product_id": "PROD-0000"})
        ctx  = execute(plan, _preprocessed(product_id="PROD-0000"))
        assert ctx.product_detail is None
        assert ctx.has_errors


# ─── Execute — Product Search Pipeline ───────────────────────────────────────

class TestExecuteProductSearch:
    def test_search_returns_results(self):
        plan = _make_plan(INTENT_PRODUCT_SEARCH, {"search_query": "headphones"})
        ctx  = execute(plan, _preprocessed())
        assert isinstance(ctx.search_results, list)
        assert len(ctx.search_results) > 0

    def test_search_calls_search_products(self):
        plan = _make_plan(INTENT_PRODUCT_SEARCH, {"search_query": "shoes"})
        ctx  = execute(plan, _preprocessed())
        assert "search_products" in ctx.tool_calls

    def test_empty_search_returns_empty_list_not_error(self):
        plan = _make_plan(INTENT_PRODUCT_SEARCH, {"search_query": "xyznonexistentproduct123"})
        ctx  = execute(plan, _preprocessed())
        assert ctx.search_results == []
        assert not ctx.has_errors   # Empty result is NOT an error

    def test_search_does_not_call_get_order(self):
        plan = _make_plan(INTENT_PRODUCT_SEARCH, {"search_query": "headphones"})
        ctx  = execute(plan, _preprocessed())
        assert "get_order" not in ctx.tool_calls


# ─── Execute — Unknown Intent ─────────────────────────────────────────────────

class TestExecuteUnknown:
    def test_no_tools_called_for_unknown(self):
        plan = _make_plan(INTENT_UNKNOWN, {})
        ctx  = execute(plan, _preprocessed())
        assert ctx.tool_calls == []

    def test_no_errors_for_unknown(self):
        plan = _make_plan(INTENT_UNKNOWN, {})
        ctx  = execute(plan, _preprocessed())
        assert not ctx.has_errors

    def test_no_results_for_unknown(self):
        plan = _make_plan(INTENT_UNKNOWN, {})
        ctx  = execute(plan, _preprocessed())
        assert not ctx.has_results


# ─── Execute — Timing ─────────────────────────────────────────────────────────

class TestExecuteTiming:
    def test_execution_time_is_recorded(self):
        plan = _make_plan(INTENT_PRODUCT_SEARCH, {"search_query": "shoes"})
        ctx  = execute(plan, _preprocessed())
        assert ctx.execution_time_ms >= 0

    def test_execution_time_is_float(self):
        plan = _make_plan(INTENT_ORDER_STATUS, {"order_id": "ORD-1001"})
        ctx  = execute(plan, _preprocessed(order_id="ORD-1001"))
        assert isinstance(ctx.execution_time_ms, float)


# ─── execute_legacy() Backward-Compatibility Tests ───────────────────────────

class TestExecuteLegacy:
    def test_order_status_via_legacy(self):
        intents = {
            "order_id": "ORD-1002", "product_id": None,
            "wants_order_status": True, "wants_cheaper_alternative": False,
            "wants_product_search": False, "search_query": None,
        }
        ctx = execute_legacy(intents)
        assert ctx["order"] is not None
        assert ctx["order"]["order_id"] == "ORD-1002"

    def test_product_search_via_legacy(self):
        intents = {
            "order_id": None, "product_id": None,
            "wants_order_status": False, "wants_cheaper_alternative": False,
            "wants_product_search": True, "search_query": "xyznonexistentproduct123",
        }
        ctx = execute_legacy(intents)
        assert ctx["search_results"] == []
        assert ctx["errors"] == []

    def test_product_detail_via_legacy(self):
        intents = {
            "order_id": None, "product_id": "PROD-201",
            "wants_order_status": False, "wants_cheaper_alternative": False,
            "wants_product_search": False, "search_query": None,
        }
        ctx = execute_legacy(intents)
        assert ctx["product_detail"] is not None
        assert ctx["product_detail"]["product_id"] == "PROD-201"
        assert ctx["search_results"] == []

    def test_cheaper_alternative_via_legacy(self):
        intents = {
            "order_id": "ORD-1001", "product_id": None,
            "wants_order_status": True, "wants_cheaper_alternative": True,
            "wants_product_search": False, "search_query": None,
        }
        ctx = execute_legacy(intents)
        assert ctx["order"] is not None
        assert ctx["ordered_product"] is not None

    def test_legacy_returns_required_keys(self):
        intents = {
            "order_id": None, "product_id": None,
            "wants_order_status": False, "wants_cheaper_alternative": False,
            "wants_product_search": False, "search_query": None,
        }
        ctx = execute_legacy(intents)
        for key in ["order", "ordered_product", "search_results", "product_detail", "errors"]:
            assert key in ctx, f"Legacy ctx must contain key {key!r}"
