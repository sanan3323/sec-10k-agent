"""Tests for the router and shared JSON extraction, with a faked generator."""

from __future__ import annotations

import pytest

from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.router import route
from sec_10k_agent.rag.models import LLMResponse


class _FakeGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_call: tuple[str, str] | None = None

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        self.last_call = (system, user)
        return LLMResponse(text=self._text, model="fake-router")


def test_extract_json_object_plain() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_tolerates_code_fence_and_prose() -> None:
    text = 'Sure, here:\n```json\n{"a": 1, "b": "x"}\n```'
    assert extract_json_object(text) == {"a": 1, "b": "x"}


def test_extract_json_object_raises_without_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("no braces here")


def test_route_single_ticker_single_year() -> None:
    reply = """{
      "needs_decomposition": false,
      "retrieval_modes": ["semantic"],
      "is_temporal": false,
      "candidate_filters": {"tickers": ["AAPL"], "fiscal_years": [2024], "sections": ["Item 1A"]},
      "reasoning": "single company, single year, narrative question"
    }"""
    plan = route("What supply chain risks does Apple disclose?", _FakeGenerator(reply))
    assert plan.needs_decomposition is False
    assert plan.retrieval_modes == ["semantic"]
    assert plan.is_temporal is False
    assert plan.candidate_filters.tickers == ["AAPL"]
    assert plan.candidate_filters.fiscal_years == [2024]
    assert plan.reasoning


def test_route_multi_ticker_needs_decomposition() -> None:
    reply = """{
      "needs_decomposition": true,
      "retrieval_modes": ["semantic"],
      "is_temporal": false,
      "candidate_filters": {"tickers": ["AAPL", "NVDA"], "fiscal_years": [], "sections": ["Item 1A"]},
      "reasoning": "compares two companies"
    }"""
    plan = route("Compare Apple and NVIDIA's supply chain risks", _FakeGenerator(reply))
    assert plan.needs_decomposition is True
    assert set(plan.candidate_filters.tickers) == {"AAPL", "NVDA"}


def test_route_temporal_multi_year() -> None:
    reply = """{
      "needs_decomposition": true,
      "retrieval_modes": ["semantic"],
      "is_temporal": true,
      "candidate_filters": {"tickers": ["AAPL"], "fiscal_years": [2022, 2023, 2024], "sections": []},
      "reasoning": "asks how risk disclosure evolved over years"
    }"""
    plan = route("How has Apple's risk factor language evolved?", _FakeGenerator(reply))
    assert plan.is_temporal is True
    assert plan.needs_decomposition is True
    assert plan.candidate_filters.fiscal_years == [2022, 2023, 2024]


def test_route_structured_xbrl_mode() -> None:
    reply = """{
      "needs_decomposition": false,
      "retrieval_modes": ["structured_xbrl"],
      "is_temporal": false,
      "candidate_filters": {"tickers": ["AAPL"], "fiscal_years": [2024], "sections": []},
      "reasoning": "asks for a specific revenue figure"
    }"""
    plan = route("What was Apple's total revenue in FY2024?", _FakeGenerator(reply))
    assert plan.retrieval_modes == ["structured_xbrl"]


def test_route_defaults_missing_fields() -> None:
    # A minimal reply still produces a valid plan via defaults.
    plan = route("anything", _FakeGenerator('{"needs_decomposition": false}'))
    assert plan.retrieval_modes == ["semantic"]
    assert plan.candidate_filters.tickers == []
    assert plan.reasoning == ""


def test_route_passes_question_to_generator() -> None:
    gen = _FakeGenerator('{"needs_decomposition": false}')
    route("what about JPM's credit risk", gen)
    assert gen.last_call is not None
    system, user = gen.last_call
    assert "AAPL" in system  # corpus scope is in the system prompt
    assert "what about JPM's credit risk" in user
