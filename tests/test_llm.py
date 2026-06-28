"""
tests/test_llm.py — Unit tests for the LLM abstraction layer.

Covers:
  • AgentPlan dataclass creation and to_legacy_intents() conversion
  • LLMProvider abstract interface compliance
  • get_planner() factory logic and fallback behaviour
  • GeminiProvider with mocked responses (no real API calls)

Run:  python -m pytest tests/test_llm.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

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
from agent.planner import DeterministicPlanner, get_planner, preprocess


# ─── AgentPlan Dataclass Tests ────────────────────────────────────────────────

class TestAgentPlan:
    def test_creation_with_all_fields(self):
        plan = AgentPlan(
            intent=INTENT_ORDER_STATUS,
            entities={"order_id": "ORD-1002"},
            reasoning="Test reasoning",
            confidence=0.95,
            recommended_tool_sequence=[TOOL_GET_ORDER],
        )
        assert plan.intent    == INTENT_ORDER_STATUS
        assert plan.confidence == 0.95
        assert TOOL_GET_ORDER in plan.recommended_tool_sequence

    def test_default_tool_sequence_is_empty(self):
        plan = AgentPlan(
            intent=INTENT_UNKNOWN,
            entities={},
            reasoning="test",
            confidence=0.5,
        )
        assert plan.recommended_tool_sequence == []

    def test_all_intent_constants_are_strings(self):
        for intent in ALL_INTENTS:
            assert isinstance(intent, str)
            assert len(intent) > 0

    def test_all_tool_constants_are_strings(self):
        for tool in ALL_TOOLS:
            assert isinstance(tool, str)
            assert len(tool) > 0

    def test_five_distinct_intents(self):
        assert len(ALL_INTENTS) == 5

    def test_three_distinct_tools(self):
        assert len(ALL_TOOLS) == 3

    # ── to_legacy_intents() ───────────────────────────────────────────────────

    def test_legacy_order_status(self):
        plan = AgentPlan(
            intent=INTENT_ORDER_STATUS,
            entities={"order_id": "ORD-1002", "product_id": None, "search_query": None},
            reasoning="test", confidence=1.0,
        )
        legacy = plan.to_legacy_intents()
        assert legacy["order_id"]                == "ORD-1002"
        assert legacy["wants_order_status"]       is True
        assert legacy["wants_cheaper_alternative"] is False
        assert legacy["wants_product_search"]      is False
        assert legacy["search_query"]             is None

    def test_legacy_cheaper_alternative(self):
        plan = AgentPlan(
            intent=INTENT_CHEAPER_ALTERNATIVE,
            entities={"order_id": "ORD-1001", "product_id": None, "search_query": None},
            reasoning="test", confidence=0.9,
        )
        legacy = plan.to_legacy_intents()
        assert legacy["wants_cheaper_alternative"] is True
        assert legacy["wants_order_status"]        is False
        assert legacy["wants_product_search"]      is False
        assert legacy["order_id"]                 == "ORD-1001"

    def test_legacy_product_search(self):
        plan = AgentPlan(
            intent=INTENT_PRODUCT_SEARCH,
            entities={"order_id": None, "product_id": None, "search_query": "headphones"},
            reasoning="test", confidence=0.85,
        )
        legacy = plan.to_legacy_intents()
        assert legacy["wants_product_search"]      is True
        assert legacy["wants_order_status"]        is False
        assert legacy["wants_cheaper_alternative"] is False
        assert legacy["search_query"]              == "headphones"

    def test_legacy_product_detail(self):
        plan = AgentPlan(
            intent=INTENT_PRODUCT_DETAIL,
            entities={"order_id": None, "product_id": "PROD-201", "search_query": None},
            reasoning="test", confidence=1.0,
        )
        legacy = plan.to_legacy_intents()
        assert legacy["product_id"]           == "PROD-201"
        assert legacy["wants_order_status"]    is False
        assert legacy["wants_product_search"]  is False

    def test_legacy_unknown(self):
        plan = AgentPlan(
            intent=INTENT_UNKNOWN,
            entities={"order_id": None, "product_id": None, "search_query": None},
            reasoning="test", confidence=0.5,
        )
        legacy = plan.to_legacy_intents()
        assert legacy["wants_order_status"]        is False
        assert legacy["wants_cheaper_alternative"] is False
        assert legacy["wants_product_search"]      is False

    def test_legacy_has_all_required_keys(self):
        plan = AgentPlan(
            intent=INTENT_UNKNOWN, entities={}, reasoning="test", confidence=0.0,
        )
        legacy = plan.to_legacy_intents()
        required_keys = [
            "order_id", "product_id", "wants_order_status",
            "wants_cheaper_alternative", "wants_product_search", "search_query",
        ]
        for key in required_keys:
            assert key in legacy, f"Legacy intents must contain key {key!r}"


# ─── LLMProvider Interface Tests ──────────────────────────────────────────────

class TestLLMProviderInterface:
    def test_deterministic_planner_is_llm_provider(self):
        """DeterministicPlanner must implement LLMProvider."""
        assert isinstance(DeterministicPlanner(), LLMProvider)

    def test_llm_provider_is_abstract(self):
        """LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore

    def test_deterministic_planner_plan_returns_agent_plan(self):
        planner = DeterministicPlanner()
        pp      = preprocess("Show me shoes")
        plan    = planner.plan("Show me shoes", pp)
        assert isinstance(plan, AgentPlan)

    def test_deterministic_planner_repr(self):
        planner = DeterministicPlanner()
        assert "DeterministicPlanner" in repr(planner)


# ─── get_planner() Factory Tests ──────────────────────────────────────────────

class TestGetPlannerFactory:
    def test_returns_llm_provider_instance(self):
        planner = get_planner()
        assert isinstance(planner, LLMProvider)

    def test_returns_deterministic_when_mode_is_deterministic(self):
        import agent.planner as planner_mod
        planner_mod._planner_instance = None  # reset singleton so patch takes effect
        with patch("agent.planner.LLM_PROVIDER", "deterministic"):
            planner = get_planner()
            assert isinstance(planner, DeterministicPlanner)
        planner_mod._planner_instance = None  # clean up

    def test_returns_deterministic_when_gemini_mode_but_no_key(self):
        """Without an API key, must fall back to deterministic even if mode=gemini."""
        import agent.planner as planner_mod
        planner_mod._planner_instance = None  # reset singleton
        with patch("agent.planner.LLM_PROVIDER", "gemini"), \
             patch("agent.planner.GEMINI_API_KEY", ""):
            planner = get_planner()
            assert isinstance(planner, DeterministicPlanner)
        planner_mod._planner_instance = None  # clean up

    def test_returns_deterministic_when_gemini_provider_init_fails(self):
        """If GeminiProvider.__init__ raises, fall back to DeterministicPlanner."""
        import agent.planner as planner_mod
        planner_mod._planner_instance = None  # reset singleton
        with patch("agent.planner.LLM_PROVIDER", "gemini"), \
             patch("agent.planner.GEMINI_API_KEY", "fake-key-for-test"):
            # Patch inside get_planner's local import
            with patch.dict("sys.modules", {"llm.gemini_provider": MagicMock(
                GeminiProvider=MagicMock(side_effect=Exception("init failed"))
            )}):
                planner = get_planner()
                assert isinstance(planner, DeterministicPlanner)
        planner_mod._planner_instance = None  # clean up

    def test_planner_produces_valid_plan_for_every_sample(self):
        """Smoke test: planner must not crash on any realistic input."""
        planner = get_planner()
        samples = [
            "Where is ORD-1002?",
            "Is there a cheaper alternative to what I ordered in ORD-1001?",
            "Show me wireless headphones",
            "Get details for PROD-201",
            "Hello!",
            "",
        ]
        for question in samples:
            pp   = preprocess(question)
            plan = planner.plan(question, pp)
            assert isinstance(plan, AgentPlan), f"Expected AgentPlan for {question!r}"
            assert plan.intent in ALL_INTENTS,  f"Unexpected intent for {question!r}"


# ─── GeminiProvider Mock Tests ────────────────────────────────────────────────

class TestGeminiProviderMock:
    """
    Tests for GeminiProvider using a mock — no real API calls made.

    These tests verify parsing, validation, and fallback logic without
    requiring a GEMINI_API_KEY or network access.
    """

    def _make_provider(self, mock_model: MagicMock):
        """Create a GeminiProvider with a mocked internal model."""
        from llm.gemini_provider import GeminiProvider
        provider        = GeminiProvider.__new__(GeminiProvider)
        provider._model = mock_model
        return provider

    def _preprocessed(self, order_id=None, product_id=None):
        return {
            "order_id":         order_id,
            "product_id":       product_id,
            "cleaned_question": "",
            "raw_question":     "test",
        }

    def test_parses_valid_order_status_response(self):
        valid_json = """{
            "intent": "order_status",
            "entities": {"order_id": "ORD-1002", "product_id": null, "search_query": null},
            "reasoning": "User asked about order status.",
            "confidence": 0.95,
            "recommended_tool_sequence": ["get_order"]
        }"""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=valid_json)
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed(order_id="ORD-1002")
        plan = provider.plan("Where is ORD-1002?", pp)

        assert plan.intent               == INTENT_ORDER_STATUS
        assert plan.entities["order_id"] == "ORD-1002"
        assert plan.confidence           == 0.95
        assert TOOL_GET_ORDER in plan.recommended_tool_sequence

    def test_parses_valid_product_search_response(self):
        valid_json = """{
            "intent": "product_search",
            "entities": {"order_id": null, "product_id": null, "search_query": "wireless headphones"},
            "reasoning": "User is browsing for headphones.",
            "confidence": 0.88,
            "recommended_tool_sequence": ["search_products"]
        }"""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=valid_json)
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed()
        plan = provider.plan("Show me wireless headphones", pp)

        assert plan.intent                     == INTENT_PRODUCT_SEARCH
        assert plan.entities["search_query"]   == "wireless headphones"
        assert TOOL_SEARCH_PRODUCTS in plan.recommended_tool_sequence

    def test_falls_back_on_malformed_json(self):
        """Bad JSON -> fallback to DeterministicPlanner which classifies the question."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="not valid json {{{")
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed()
        plan = provider.plan("test question", pp)

        # DeterministicPlanner classifies "test question" as product_search
        # (two-word query with no recognised intents => search fallback).
        # Confidence is deterministic, not 0.0.
        from llm.base import ALL_INTENTS
        assert plan.intent in ALL_INTENTS

    def test_falls_back_on_api_exception(self):
        """API exception -> fallback to DeterministicPlanner which classifies the question."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API quota exceeded")
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed()
        plan = provider.plan("test question", pp)

        # DeterministicPlanner classifies "test question" as product_search.
        from llm.base import ALL_INTENTS
        assert plan.intent in ALL_INTENTS

    def test_rejects_invalid_intent_from_model(self):
        """If model returns an unrecognised intent, it should be replaced with unknown."""
        invalid_json = """{
            "intent": "invent_something_new",
            "entities": {"order_id": null, "product_id": null, "search_query": null},
            "reasoning": "test", "confidence": 0.9,
            "recommended_tool_sequence": []
        }"""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=invalid_json)
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed()
        plan = provider.plan("test", pp)

        assert plan.intent == INTENT_UNKNOWN

    def test_rejects_invalid_tool_names_from_model(self):
        """If model suggests non-existent tool names, they should be filtered out."""
        json_with_bad_tools = """{
            "intent": "order_status",
            "entities": {"order_id": "ORD-1001", "product_id": null, "search_query": null},
            "reasoning": "test", "confidence": 0.9,
            "recommended_tool_sequence": ["get_order", "execute_sql_query", "delete_table"]
        }"""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=json_with_bad_tools)
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed(order_id="ORD-1001")
        plan = provider.plan("Where is ORD-1001?", pp)

        # Only valid tool names should survive
        assert "execute_sql_query" not in plan.recommended_tool_sequence
        assert "delete_table"      not in plan.recommended_tool_sequence
        assert TOOL_GET_ORDER in plan.recommended_tool_sequence

    def test_confidence_is_clamped_to_valid_range(self):
        """Confidence from model must be clamped to [0.0, 1.0]."""
        json_with_bad_confidence = """{
            "intent": "product_search",
            "entities": {"order_id": null, "product_id": null, "search_query": "shoes"},
            "reasoning": "test",
            "confidence": 9999.0,
            "recommended_tool_sequence": ["search_products"]
        }"""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=json_with_bad_confidence)
        provider = self._make_provider(mock_model)

        pp   = self._preprocessed()
        plan = provider.plan("Find shoes", pp)

        assert 0.0 <= plan.confidence <= 1.0

    def test_preprocessed_order_id_takes_priority_over_model(self):
        """Pre-extracted order_id should override whatever the model returns."""
        json_with_wrong_id = """{
            "intent": "order_status",
            "entities": {"order_id": "ORD-9999", "product_id": null, "search_query": null},
            "reasoning": "test", "confidence": 0.8,
            "recommended_tool_sequence": ["get_order"]
        }"""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text=json_with_wrong_id)
        provider = self._make_provider(mock_model)

        # Pre-extracted ID is ORD-1002 (from regex) — must not be overridden
        pp   = self._preprocessed(order_id="ORD-1002")
        plan = provider.plan("Where is ORD-1002?", pp)

        assert plan.entities["order_id"] == "ORD-1002"
