"""Tests for the verify_citations node, with a faked generator."""

from __future__ import annotations

from sec_10k_agent.agent.state import Claim, Retrieval, SubQuery
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT
from sec_10k_agent.agent.verify import verify_citations
from sec_10k_agent.rag.models import LLMResponse
from sec_10k_agent.retrieval.models import RetrievedChunk


class _FakeGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self._text, model="fake-verifier")


class _ExplodingGenerator:
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        raise AssertionError("generator should not be called for this claim")


def _retrievals() -> list[Retrieval]:
    chunk = RetrievedChunk(
        chunk_id="c1", ticker="AAPL", section="Item 1A", text="body", distance=0.1
    )
    return [Retrieval(subquery=SubQuery(question="q"), chunks=[chunk])]


def test_abstention_with_no_citations_is_verified_without_llm_call() -> None:
    claim = Claim(text=NO_INFO_TEXT, chunk_ids=[])
    verified = verify_citations([claim], _retrievals(), _ExplodingGenerator())  # type: ignore[arg-type]
    assert verified[0].verified is True


def test_unsupported_assertion_with_no_citations_fails_without_llm_call() -> None:
    claim = Claim(text="Apple's revenue tripled.", chunk_ids=[])
    verified = verify_citations([claim], _retrievals(), _ExplodingGenerator())  # type: ignore[arg-type]
    assert verified[0].verified is False


def test_claim_citing_only_hallucinated_ids_fails_without_llm_call() -> None:
    claim = Claim(text="Some claim.", chunk_ids=["does-not-exist"])
    verified = verify_citations([claim], _retrievals(), _ExplodingGenerator())  # type: ignore[arg-type]
    assert verified[0].verified is False


def test_claim_with_real_citation_calls_generator_and_honors_verdict() -> None:
    claim = Claim(text="Apple discloses supply chain risk.", chunk_ids=["c1"])
    gen = _FakeGenerator('{"supported": true, "reasoning": "matches the source"}')
    verified = verify_citations([claim], _retrievals(), gen)  # type: ignore[arg-type]
    assert verified[0].verified is True
    assert gen.calls == 1


def test_claim_rejected_by_generator() -> None:
    claim = Claim(text="Apple discloses something unrelated.", chunk_ids=["c1"])
    gen = _FakeGenerator('{"supported": false, "reasoning": "not in source"}')
    verified = verify_citations([claim], _retrievals(), gen)  # type: ignore[arg-type]
    assert verified[0].verified is False


def test_verify_citations_preserves_claim_text_and_ids() -> None:
    claim = Claim(text="Text.", chunk_ids=["c1"])
    gen = _FakeGenerator('{"supported": true}')
    verified = verify_citations([claim], _retrievals(), gen)  # type: ignore[arg-type]
    assert verified[0].text == "Text."
    assert verified[0].chunk_ids == ["c1"]
