"""
tests/test_planner.py — Unit tests for agent/planner.py.

Covers:
  • preprocess()           — entity extraction and cleaned question
  • DeterministicPlanner   — intent classification for all intent types
  • validate_plan()        — plan validation for each intent

Run:  python -m pytest tests/test_planner.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.planner import preprocess, DeterministicPlanner, validate_plan
from llm.base import (
    AgentPlan,
    INTENT_ORDER_STATUS,
    INTENT_CHEAPER_ALTERNATIVE,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_DETAIL,
    INTENT_UNKNOWN,
    TOOL_GET_ORDER,
    TOOL_GET_PRODUCT,
    TOOL_SEARCH_PRODUCTS,
)

# Shared planner instance — DeterministicPlanner is stateless
PLANNER = DeterministicPlanner()


def _plan(question: str) -> AgentPlan:
    """Helper: preprocess + plan in one call."""
    pp = preprocess(question)
    return PLANNER.plan(question, pp)


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestPreprocess:
    def test_extracts_order_id(self):
        r = preprocess("Where is my order ORD-1002?")
        assert r["order_id"] == "ORD-1002"

    def test_extracts_product_id(self):
        r = preprocess("Tell me about PROD-201")
        assert r["product_id"] == "PROD-201"

    def test_normalises_order_id_case(self):
        r = preprocess("status of ord-1001")
        assert r["order_id"] == "ORD-1001"

    def test_normalises_product_id_case(self):
        r = preprocess("show me prod-305")
        assert r["product_id"] == "PROD-305"

    def test_no_ids_when_absent(self):
        r = preprocess("Show me headphones")
        assert r["order_id"]   is None
        assert r["product_id"] is None

    def test_cleaned_question_strips_order_id(self):
        r = preprocess("Where is ORD-1002?")
        assert "ORD-1002" not in r["cleaned_question"]
        assert "ORD" not in r["cleaned_question"]

    def test_cleaned_question_strips_product_id(self):
        r = preprocess("Get details for PROD-201")
        assert "PROD-201" not in r["cleaned_question"]

    def test_raw_question_is_unchanged(self):
        q = "Where is ORD-1002 and PROD-201?"
        r = preprocess(q)
        assert r["raw_question"] == q

    def test_both_ids_extracted(self):
        r = preprocess("Is PROD-201 in ORD-1001?")
        assert r["order_id"]   == "ORD-1001"
        assert r["product_id"] == "PROD-201"


# ─── DeterministicPlanner — Intent Classification ────────────────────────────

class TestDeterministicPlannerIntents:
    # ── Order Status ──────────────────────────────────────────────────────────

    def test_order_status_via_keyword(self):
        plan = _plan("Where is my order ORD-1002?")
        assert plan.intent == INTENT_ORDER_STATUS

    def test_order_status_via_order_id_alone(self):
        plan = _plan("ORD-1003")
        assert plan.intent == INTENT_ORDER_STATUS

    def test_order_status_has_correct_entity(self):
        plan = _plan("Has my order ORD-1003 been shipped?")
        assert plan.entities["order_id"] == "ORD-1003"

    def test_order_status_recommends_get_order(self):
        plan = _plan("Where is ORD-1001?")
        assert TOOL_GET_ORDER in plan.recommended_tool_sequence

    # ── Cheaper Alternative ───────────────────────────────────────────────────

    def test_cheaper_alternative_with_phrase_and_order(self):
        plan = _plan("Is there a cheaper alternative to the shoes in ORD-1001?")
        assert plan.intent == INTENT_CHEAPER_ALTERNATIVE

    def test_cheaper_alternative_with_ownership_context(self):
        plan = _plan("Can you suggest something cheaper than what I ordered in ORD-1001?")
        assert plan.intent == INTENT_CHEAPER_ALTERNATIVE

    def test_cheaper_alternative_recommends_all_three_tools(self):
        plan = _plan("Is there a cheaper alternative to the shoes in ORD-1001?")
        assert TOOL_GET_ORDER       in plan.recommended_tool_sequence
        assert TOOL_GET_PRODUCT     in plan.recommended_tool_sequence
        assert TOOL_SEARCH_PRODUCTS in plan.recommended_tool_sequence

    def test_affordable_alone_is_NOT_alternative(self):
        plan = _plan("Find affordable running shoes")
        assert plan.intent != INTENT_CHEAPER_ALTERNATIVE

    def test_budget_alone_is_NOT_alternative(self):
        plan = _plan("Show me budget headphones")
        assert plan.intent != INTENT_CHEAPER_ALTERNATIVE

    def test_alternative_without_order_is_NOT_alternative(self):
        plan = _plan("Show me an alternative to Nike shoes")
        assert plan.intent != INTENT_CHEAPER_ALTERNATIVE

    # ── Product Detail ────────────────────────────────────────────────────────

    def test_product_detail_via_product_id(self):
        plan = _plan("Get details for PROD-201")
        assert plan.intent == INTENT_PRODUCT_DETAIL

    def test_product_detail_correct_entity(self):
        plan = _plan("Get details for PROD-201")
        assert plan.entities["product_id"] == "PROD-201"

    def test_product_detail_recommends_get_product(self):
        plan = _plan("Get details for PROD-201")
        assert TOOL_GET_PRODUCT in plan.recommended_tool_sequence

    def test_product_detail_has_full_confidence(self):
        plan = _plan("Get details for PROD-201")
        assert plan.confidence == 1.0

    # ── Product Search ────────────────────────────────────────────────────────

    def test_product_search_via_keyword(self):
        plan = _plan("Show me wireless headphones")
        assert plan.intent == INTENT_PRODUCT_SEARCH

    def test_product_search_via_looking_for(self):
        plan = _plan("I'm looking for jeans")
        assert plan.intent == INTENT_PRODUCT_SEARCH

    def test_product_search_has_search_query(self):
        plan = _plan("Show me wireless headphones")
        assert plan.entities["search_query"] is not None
        assert len(plan.entities["search_query"]) > 0

    def test_product_search_recommends_search_products(self):
        plan = _plan("Show me wireless headphones")
        assert TOOL_SEARCH_PRODUCTS in plan.recommended_tool_sequence

    def test_product_search_query_strips_fillers(self):
        plan = _plan("Find me affordable running shoes")
        # "find me" and "affordable" should be stripped; "running shoes" should remain
        q = plan.entities["search_query"]
        assert "running" in q.lower() or "shoes" in q.lower()

    # ── Unknown Intent ────────────────────────────────────────────────────────

    def test_hello_produces_unknown_intent(self):
        plan = _plan("Hello!")
        assert plan.intent == INTENT_UNKNOWN

    def test_unknown_has_no_tools(self):
        plan = _plan("Hello!")
        assert plan.recommended_tool_sequence == []

    # ── Plan metadata ─────────────────────────────────────────────────────────

    def test_confidence_is_float(self):
        plan = _plan("Where is ORD-1002?")
        assert isinstance(plan.confidence, float)

    def test_confidence_in_range(self):
        plan = _plan("Where is ORD-1002?")
        assert 0.0 <= plan.confidence <= 1.0

    def test_reasoning_is_non_empty_string(self):
        plan = _plan("Show me shoes")
        assert isinstance(plan.reasoning, str)
        assert len(plan.reasoning) > 0

    def test_to_legacy_intents_order_status(self):
        plan = _plan("Where is ORD-1002?")
        legacy = plan.to_legacy_intents()
        assert legacy["order_id"] == "ORD-1002"
        assert legacy["wants_order_status"] is True
        assert legacy["wants_cheaper_alternative"] is False

    def test_to_legacy_intents_alternative(self):
        plan = _plan("Is there a cheaper alternative to the shoes in ORD-1001?")
        legacy = plan.to_legacy_intents()
        assert legacy["wants_cheaper_alternative"] is True

    def test_to_legacy_intents_product_search(self):
        plan = _plan("Show me wireless headphones")
        legacy = plan.to_legacy_intents()
        assert legacy["wants_product_search"] is True
        assert legacy["search_query"] is not None


# ─── Plan Validator Tests ─────────────────────────────────────────────────────

class TestPlanValidator:
    def _make(self, intent: str, entities: dict) -> AgentPlan:
        return AgentPlan(
            intent=intent, entities=entities,
            reasoning="test", confidence=1.0,
        )

    # ── Order Status ──────────────────────────────────────────────────────────

    def test_order_status_without_order_id_is_invalid(self):
        plan = self._make(INTENT_ORDER_STATUS, {"order_id": None})
        valid, msg = validate_plan(plan)
        assert not valid
        assert "order ID" in msg

    def test_order_status_with_order_id_is_valid(self):
        plan = self._make(INTENT_ORDER_STATUS, {"order_id": "ORD-1001"})
        valid, msg = validate_plan(plan)
        assert valid
        assert msg == ""

    # ── Cheaper Alternative ───────────────────────────────────────────────────

    def test_cheaper_alternative_without_order_id_is_invalid(self):
        plan = self._make(INTENT_CHEAPER_ALTERNATIVE, {"order_id": None})
        valid, msg = validate_plan(plan)
        assert not valid
        assert len(msg) > 0

    def test_cheaper_alternative_with_order_id_is_valid(self):
        plan = self._make(INTENT_CHEAPER_ALTERNATIVE, {"order_id": "ORD-1001"})
        valid, _ = validate_plan(plan)
        assert valid

    # ── Product Detail ────────────────────────────────────────────────────────

    def test_product_detail_without_product_id_is_invalid(self):
        plan = self._make(INTENT_PRODUCT_DETAIL, {"product_id": None})
        valid, msg = validate_plan(plan)
        assert not valid
        assert len(msg) > 0

    def test_product_detail_with_product_id_is_valid(self):
        plan = self._make(INTENT_PRODUCT_DETAIL, {"product_id": "PROD-201"})
        valid, _ = validate_plan(plan)
        assert valid

    # ── Product Search ────────────────────────────────────────────────────────

    def test_product_search_without_query_is_invalid(self):
        plan = self._make(INTENT_PRODUCT_SEARCH, {"search_query": None})
        valid, msg = validate_plan(plan)
        assert not valid
        assert len(msg) > 0

    def test_product_search_with_query_is_valid(self):
        plan = self._make(INTENT_PRODUCT_SEARCH, {"search_query": "headphones"})
        valid, _ = validate_plan(plan)
        assert valid

    # ── Unknown Intent ────────────────────────────────────────────────────────

    def test_unknown_intent_is_always_valid(self):
        """Unknown intents have no required entities — the responder handles them."""
        plan = self._make(INTENT_UNKNOWN, {})
        valid, _ = validate_plan(plan)
        assert valid

    # ── Validation message quality ────────────────────────────────────────────

    def test_validation_messages_are_customer_friendly(self):
        """Validation messages should not expose internal implementation details."""
        cases = [
            (INTENT_ORDER_STATUS,        {"order_id": None}),
            (INTENT_CHEAPER_ALTERNATIVE, {"order_id": None}),
            (INTENT_PRODUCT_DETAIL,      {"product_id": None}),
            (INTENT_PRODUCT_SEARCH,      {"search_query": None}),
        ]
        for intent, entities in cases:
            plan = self._make(intent, entities)
            _, msg = validate_plan(plan)
            # Must not expose Python internals
            assert "dict" not in msg.lower()
            assert "None" not in msg
            assert "KeyError" not in msg
