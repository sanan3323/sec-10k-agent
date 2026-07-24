"""Tests for the synthesize node, with a faked generator."""

from __future__ import annotations

from sec_10k_agent.agent.state import Retrieval, SubQuery, XBRLFact
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT, synthesize
from sec_10k_agent.rag.models import LLMResponse
from sec_10k_agent.retrieval.models import RetrievedChunk


class _FakeGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_prompt: tuple[str, str] | None = None

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        self.last_prompt = (system, user)
        return LLMResponse(text=self._text, model="fake-synth")


def _retrievals() -> list[Retrieval]:
    chunk = RetrievedChunk(
        chunk_id="AAPL__Item 1A__0001", ticker="AAPL", text="Supply chain risk text.", distance=0.1
    )
    fact = XBRLFact(
        ticker="AAPL",
        fiscal_year=2024,
        concept="us-gaap:Revenues",
        value=391035000000.0,
        unit="USD",
    )
    return [Retrieval(subquery=SubQuery(question="q"), chunks=[chunk], facts=[fact])]


def test_synthesize_produces_claims_tagged_with_real_source_ids() -> None:
    reply = """{"claims": [
      {"text": "Apple discloses supply chain concentration risk.", "source_ids": ["AAPL__Item 1A__0001"]},
      {"text": "Apple's FY2024 revenue was $391.035B.", "source_ids": ["fact:AAPL:2024:us-gaap:Revenues"]}
    ]}"""
    claims = synthesize(
        "What are Apple's risks and FY2024 revenue?", _retrievals(), _FakeGenerator(reply)
    )
    assert len(claims) == 2
    assert claims[0].chunk_ids == ["AAPL__Item 1A__0001"]
    assert claims[1].chunk_ids == ["fact:AAPL:2024:us-gaap:Revenues"]


def test_synthesize_drops_hallucinated_source_ids() -> None:
    reply = """{"claims": [
      {"text": "A claim citing a real and a fake source.", "source_ids": ["AAPL__Item 1A__0001", "made-up-id"]}
    ]}"""
    claims = synthesize("q", _retrievals(), _FakeGenerator(reply))
    assert claims[0].chunk_ids == ["AAPL__Item 1A__0001"]


def test_synthesize_empty_claims_falls_back_to_no_info() -> None:
    claims = synthesize("q", _retrievals(), _FakeGenerator('{"claims": []}'))
    assert len(claims) == 1
    assert claims[0].text == NO_INFO_TEXT
    assert claims[0].chunk_ids == []


def test_synthesize_drops_claims_with_empty_text() -> None:
    reply = """{"claims": [
      {"text": "", "source_ids": ["AAPL__Item 1A__0001"]},
      {"text": "A real claim.", "source_ids": ["AAPL__Item 1A__0001"]}
    ]}"""
    claims = synthesize("q", _retrievals(), _FakeGenerator(reply))
    assert len(claims) == 1
    assert claims[0].text == "A real claim."


def test_synthesize_includes_sources_and_question_in_prompt() -> None:
    gen = _FakeGenerator('{"claims": [{"text": "x", "source_ids": []}]}')
    synthesize("What is Apple's revenue?", _retrievals(), gen)
    assert gen.last_prompt is not None
    _, user = gen.last_prompt
    assert "SOURCES:" in user
    assert "QUESTION: What is Apple's revenue?" in user
    assert "AAPL__Item 1A__0001" in user
