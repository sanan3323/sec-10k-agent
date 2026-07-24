"""Tests for the decomposer: fan-out-by-mode (no LLM) and LLM-based splitting."""

from __future__ import annotations

import pytest

from sec_10k_agent.agent.decompose import decompose
from sec_10k_agent.agent.state import CandidateFilters, RoutingPlan
from sec_10k_agent.rag.models import LLMResponse


class _FakeGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.called = False

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        self.called = True
        return LLMResponse(text=self._text, model="fake-decomposer")


class _ExplodingGenerator:
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        raise AssertionError("generator should not be called when no decomposition is needed")


def _plan(**over: object) -> RoutingPlan:
    base: dict[str, object] = {
        "needs_decomposition": False,
        "retrieval_modes": ["semantic"],
        "is_temporal": False,
        "candidate_filters": CandidateFilters(
            tickers=["AAPL"], fiscal_years=[2024], sections=["Item 1A"]
        ),
        "reasoning": "r",
    }
    base.update(over)
    return RoutingPlan(**base)  # type: ignore[arg-type]


def test_no_decomposition_single_mode_no_llm_call() -> None:
    gen = _ExplodingGenerator()
    subqueries = decompose("What supply chain risks does Apple disclose?", _plan(), gen)  # type: ignore[arg-type]
    assert len(subqueries) == 1
    sq = subqueries[0]
    assert sq.ticker == "AAPL"
    assert sq.fiscal_year == 2024
    assert sq.section == "Item 1A"
    assert sq.mode == "semantic"


def test_no_decomposition_both_modes_fans_out() -> None:
    plan = _plan(retrieval_modes=["semantic", "structured_xbrl"])
    subqueries = decompose("q", plan, _ExplodingGenerator())  # type: ignore[arg-type]
    assert len(subqueries) == 2
    assert {sq.mode for sq in subqueries} == {"semantic", "structured_xbrl"}
    assert all(sq.ticker == "AAPL" and sq.fiscal_year == 2024 for sq in subqueries)


def test_no_decomposition_no_filters_defaults_to_none() -> None:
    plan = _plan(candidate_filters=CandidateFilters())
    subqueries = decompose("q", plan, _ExplodingGenerator())  # type: ignore[arg-type]
    assert len(subqueries) == 1
    assert subqueries[0].ticker is None
    assert subqueries[0].fiscal_year is None


def test_decomposition_by_ticker_calls_llm() -> None:
    reply = """{"subqueries": [
      {"question": "What supply chain risks does Apple disclose?", "ticker": "AAPL", "fiscal_year": null, "section": "Item 1A", "mode": "semantic"},
      {"question": "What supply chain risks does NVIDIA disclose?", "ticker": "NVDA", "fiscal_year": null, "section": "Item 1A", "mode": "semantic"}
    ]}"""
    plan = _plan(
        needs_decomposition=True,
        candidate_filters=CandidateFilters(tickers=["AAPL", "NVDA"], sections=["Item 1A"]),
    )
    gen = _FakeGenerator(reply)
    subqueries = decompose("Compare Apple and NVIDIA's supply chain risks", plan, gen)  # type: ignore[arg-type]
    assert gen.called
    assert len(subqueries) == 2
    assert {sq.ticker for sq in subqueries} == {"AAPL", "NVDA"}


def test_decomposition_by_fiscal_year() -> None:
    reply = """{"subqueries": [
      {"question": "Apple risk factors FY2022", "ticker": "AAPL", "fiscal_year": 2022, "section": "Item 1A", "mode": "semantic"},
      {"question": "Apple risk factors FY2024", "ticker": "AAPL", "fiscal_year": 2024, "section": "Item 1A", "mode": "semantic"}
    ]}"""
    plan = _plan(
        needs_decomposition=True,
        is_temporal=True,
        candidate_filters=CandidateFilters(tickers=["AAPL"], fiscal_years=[2022, 2024]),
    )
    subqueries = decompose("How did Apple's risk factors evolve?", plan, _FakeGenerator(reply))  # type: ignore[arg-type]
    assert [sq.fiscal_year for sq in subqueries] == [2022, 2024]


def test_decomposition_raises_on_empty_subqueries() -> None:
    plan = _plan(needs_decomposition=True)
    with pytest.raises(ValueError, match="no subqueries"):
        decompose("q", plan, _FakeGenerator('{"subqueries": []}'))  # type: ignore[arg-type]
