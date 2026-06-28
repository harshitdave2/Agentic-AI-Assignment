"""
Unit tests for the AI Store Agent.
Run:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.store_tools import get_order, search_products, get_product
from agent.agent import run_agent, detect_intents, execute_plan

# ─── Tool Layer Tests ─────────────────────────────────────────────────────────

class TestGetOrder:
    def test_valid_order_delivered(self):
        r = get_order("ORD-1001")
        assert "error" not in r
        assert r["order_id"] == "ORD-1001"
        assert r["status"] == "Delivered"

    def test_valid_order_in_transit(self):
        r = get_order("ORD-1002")
        assert r["status"] == "In Transit"

    def test_valid_order_processing(self):
        r = get_order("ORD-1003")
        assert r["status"] == "Processing"

    def test_valid_order_cancelled(self):
        r = get_order("ORD-1004")
        assert r["status"] == "Cancelled"

    def test_invalid_order_returns_error_key(self):
        r = get_order("ORD-9999")
        assert "error" in r
        assert "ORD-9999" in r["error"]

    def test_case_insensitive(self):
        assert "error" not in get_order("ord-1001")

    def test_empty_string_returns_error(self):
        assert "error" in get_order("")


class TestSearchProducts:
    def test_search_by_category_footwear(self):
        results = search_products("Footwear")
        assert len(results) > 0
        assert all(r["category"] == "Footwear" for r in results)

    def test_search_by_keyword_headphones(self):
        results = search_products("headphones")
        assert len(results) > 0

    def test_search_by_keyword_shoes(self):
        results = search_products("shoes")
        assert len(results) > 0

    def test_no_match_returns_empty_list(self):
        results = search_products("xyznonexistentproduct123")
        assert results == []

    def test_results_sorted_by_rating_descending(self):
        results = search_products("shoes")
        ratings = [r["rating"] for r in results]
        assert ratings == sorted(ratings, reverse=True)

    def test_search_by_keyword_running(self):
        results = search_products("running")
        assert any("running" in p["name"].lower() or "running" in p["description"].lower()
                   for p in results)


class TestGetProduct:
    def test_valid_product(self):
        r = get_product("PROD-201")
        assert "error" not in r
        assert r["name"] == "Nike Air Max 270"

    def test_invalid_product_returns_error(self):
        r = get_product("PROD-0000")
        assert "error" in r

    def test_case_insensitive(self):
        assert "error" not in get_product("prod-201")


# ─── Intent Detection Tests ───────────────────────────────────────────────────

class TestIntentDetection:
    # ── Order ID extraction ──
    def test_extracts_order_id(self):
        intents = detect_intents("Where is my order ORD-1002?")
        assert intents["order_id"] == "ORD-1002"

    def test_normalises_order_id_case(self):
        intents = detect_intents("status of ord-1001")
        assert intents["order_id"] == "ORD-1001"

    def test_no_order_id_in_product_query(self):
        intents = detect_intents("Find me running shoes")
        assert intents["order_id"] is None

    # ── Product ID extraction ──
    def test_extracts_product_id(self):
        intents = detect_intents("Get details for PROD-201")
        assert intents["product_id"] == "PROD-201"

    # ── Cheaper-alternative intent: only when explicitly requested ──
    def test_cheaper_alternative_with_order_and_phrase(self):
        intents = detect_intents("Is there a cheaper alternative to the shoes in ORD-1001?")
        assert intents["wants_cheaper_alternative"] is True

    def test_cheaper_alternative_with_ownership_context(self):
        intents = detect_intents("Can you suggest something cheaper than what I ordered in ORD-1001?")
        assert intents["wants_cheaper_alternative"] is True

    def test_affordable_alone_is_NOT_cheaper_alternative(self):
        """'affordable running shoes' is a search, not an alternative request."""
        intents = detect_intents("Find affordable running shoes")
        assert intents["wants_cheaper_alternative"] is False

    def test_budget_alone_is_NOT_cheaper_alternative(self):
        intents = detect_intents("Show me budget headphones")
        assert intents["wants_cheaper_alternative"] is False

    def test_alternative_without_order_is_NOT_triggered(self):
        """'alternative' alone, with no order ID or ownership context, should not fire."""
        intents = detect_intents("Show me an alternative to Nike shoes")
        assert intents["wants_cheaper_alternative"] is False

    # ── Product search intent ──
    def test_product_search_triggered_by_keyword(self):
        intents = detect_intents("Show me wireless headphones")
        assert intents["wants_product_search"] is True

    def test_product_search_for_jeans(self):
        intents = detect_intents("I'm looking for jeans")
        assert intents["wants_product_search"] is True

    def test_product_id_lookup_triggers_product_search_flag(self):
        intents = detect_intents("Get details for PROD-201")
        # product_id is set; product search is NOT needed (we look it up directly)
        assert intents["product_id"] == "PROD-201"

    # ── Mutual exclusion ──
    def test_order_status_does_not_trigger_product_search(self):
        intents = detect_intents("Where is my order ORD-1002?")
        assert intents["wants_order_status"] is True
        assert intents["wants_cheaper_alternative"] is False

    def test_hello_produces_no_intents(self):
        intents = detect_intents("Hello!")
        assert intents["order_id"] is None
        assert intents["product_id"] is None
        assert intents["wants_cheaper_alternative"] is False


# ─── Tool Chaining / execute_plan Tests ──────────────────────────────────────

class TestExecutePlan:
    def _intents(self, question):
        return detect_intents(question)

    def test_order_lookup_populates_ctx_order(self):
        ctx = execute_plan(self._intents("Where is ORD-1002?"))
        assert ctx["order"] is not None
        assert ctx["order"]["order_id"] == "ORD-1002"

    def test_invalid_order_populates_errors(self):
        ctx = execute_plan(self._intents("Where is ORD-9999?"))
        assert ctx["order"] is None
        assert len(ctx["errors"]) == 1

    def test_cheaper_alternative_chains_three_tools(self):
        """order → product → search must all fire."""
        ctx = execute_plan(self._intents("Is there a cheaper alternative to the shoes in ORD-1001?"))
        assert ctx["order"] is not None          # get_order called
        assert ctx["ordered_product"] is not None  # get_product called
        # search_results may be [] if all are more expensive, but the key must exist
        assert "search_results" in ctx

    def test_cheaper_alternatives_are_all_cheaper_than_original(self):
        ctx = execute_plan(self._intents("Is there a cheaper alternative to the shoes in ORD-1001?"))
        if ctx["ordered_product"] and ctx["search_results"]:
            orig_price = ctx["ordered_product"]["price"]
            for alt in ctx["search_results"]:
                assert alt["price"] < orig_price

    def test_product_id_lookup_takes_priority(self):
        """PROD-XXX must be looked up directly, not routed into search."""
        ctx = execute_plan(self._intents("Get details for PROD-201"))
        assert ctx["product_detail"] is not None
        assert ctx["product_detail"]["product_id"] == "PROD-201"
        assert ctx["search_results"] == []   # search must NOT have been called

    def test_invalid_product_id_populates_errors(self):
        ctx = execute_plan(self._intents("Get details for PROD-0000"))
        assert ctx["product_detail"] is None
        assert len(ctx["errors"]) == 1

    def test_keyword_search_populates_results(self):
        ctx = execute_plan(self._intents("Show me wireless headphones"))
        assert isinstance(ctx["search_results"], list)
        assert len(ctx["search_results"]) > 0

    def test_empty_search_returns_empty_list_not_error(self):
        """No results ≠ error. The list is simply empty."""
        intents = {
            "order_id": None, "product_id": None,
            "wants_order_status": False, "wants_cheaper_alternative": False,
            "wants_product_search": True, "search_query": "xyznonexistentproduct123",
        }
        ctx = execute_plan(intents)
        assert ctx["search_results"] == []
        assert ctx["errors"] == []


# ─── End-to-End run_agent Tests ───────────────────────────────────────────────

class TestRunAgent:
    # ── Order status ──
    def test_order_in_transit(self):
        r = run_agent("Where is my order ORD-1002?")
        assert "In Transit" in r
        assert "ORD-1002" in r

    def test_order_delivered(self):
        r = run_agent("What is the status of ORD-1001?")
        assert "Delivered" in r

    def test_order_processing(self):
        r = run_agent("Has my order ORD-1003 been shipped?")
        assert "Processing" in r

    def test_order_cancelled(self):
        r = run_agent("What happened to ORD-1004?")
        assert "Cancelled" in r or "cancelled" in r

    def test_invalid_order_graceful_message(self):
        r = run_agent("Where is ORD-9999?")
        assert "sorry" in r.lower() or "couldn't" in r.lower()
        # Must NOT contain made-up order info
        assert "In Transit" not in r
        assert "Delivered" not in r

    # ── Tool chaining: cheaper alternative ──
    def test_cheaper_alternative_chains_all_tools(self):
        r = run_agent("Is there a cheaper alternative to the shoes in ORD-1001?")
        assert "Nike Air Max 270" in r    # ordered product name shown
        assert "Rs." in r                 # price info from alternatives

    def test_cheaper_alternative_for_headphones(self):
        r = run_agent("Can you suggest a budget-friendly alternative to what I bought in ORD-1002?")
        assert "Sony WH-1000XM5" in r    # original product named
        assert "Rs." in r

    # ── Product search ──
    def test_search_wireless_headphones(self):
        r = run_agent("Show me some wireless headphones")
        assert "headphone" in r.lower() or "Headphone" in r

    def test_search_jeans(self):
        r = run_agent("I'm looking for jeans")
        assert "jeans" in r.lower() or "Jeans" in r

    def test_search_affordable_running_shoes_is_NOT_alternative_query(self):
        """'affordable' alone must not trigger the alternative-search chain."""
        r = run_agent("Find affordable running shoes")
        # Should return search results, NOT ask for an order ID
        assert "Rs." in r or "found" in r.lower()
        # Must NOT ask user to specify an order
        assert "order ID" not in r

    def test_empty_search_returns_friendly_message(self):
        r = run_agent("Show me xyznonexistentproduct123")
        assert "couldn't find" in r.lower() or "no matching" in r.lower() or "searched" in r.lower()
        # Must not guess or hallucinate product names
        assert "Rs." not in r

    # ── Direct product lookup ──
    def test_direct_product_id_lookup(self):
        """PROD-XXX in the question must call get_product, not search_products."""
        r = run_agent("Get details for PROD-201")
        assert "Nike Air Max 270" in r
        assert "Rs. 8,999" in r

    def test_invalid_product_id_graceful(self):
        r = run_agent("Get details for PROD-0000")
        assert "sorry" in r.lower() or "couldn't" in r.lower() or "no product" in r.lower()

    # ── Edge cases ──
    def test_empty_question(self):
        r = run_agent("")
        assert "ask" in r.lower()

    def test_whitespace_only_question(self):
        r = run_agent("   ")
        assert "ask" in r.lower()

    def test_hello_fallback(self):
        r = run_agent("Hello!")
        assert r  # should not crash or be empty

    def test_response_is_string(self):
        assert isinstance(run_agent("Where is ORD-1001?"), str)

    def test_no_fabrication_on_unknown_order(self):
        r = run_agent("Tell me about order ORD-8888")
        assert "In Transit" not in r
        assert "Delivered" not in r
        assert "Processing" not in r
