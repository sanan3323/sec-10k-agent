"""Unit tests for the single-hop RAG pipeline.

No database, no LLM API: the retriever and generator are faked. Prompt building
and citation parsing are pure and tested directly; the pipeline is tested for
its happy path, its empty-retrieval short-circuit, and citation wiring.
"""

from __future__ import annotations

import pytest

from sec_10k_agent.config import Settings
from sec_10k_agent.rag import (
    Answer,
    RAGPipeline,
    build_generator,
    build_user_prompt,
    format_context,
    parse_citations,
)
from sec_10k_agent.rag.models import LLMResponse
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk(i: int, ticker: str = "AAPL", section: str = "Item 1A") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{ticker}__{section}__{i:04d}",
        ticker=ticker,
        fiscal_year=2024,
        section=section,
        section_title="Risk Factors",
        text=f"Risk factor number {i} about supply chains.",
        distance=0.1 * i,
    )


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_kwargs: dict[str, object] = {}

    def search(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        self.last_kwargs = {"query": query, **kwargs}
        return self._chunks


class _FakeGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_prompt: tuple[str, str] | None = None

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        self.last_prompt = (system, user)
        return LLMResponse(
            text=self._text, model="fake-model", prompt_tokens=10, completion_tokens=5
        )


# --- pure functions ----------------------------------------------------------


def test_format_context_numbers_sources() -> None:
    ctx = format_context([_chunk(1), _chunk(2)])
    assert "[1] AAPL FY2024 Item 1A" in ctx
    assert "[2] AAPL FY2024 Item 1A" in ctx


def test_format_context_empty() -> None:
    assert format_context([]) == "(no sources retrieved)"


def test_build_user_prompt_contains_sources_and_question() -> None:
    prompt = build_user_prompt("What are the supply chain risks?", [_chunk(1)])
    assert "SOURCES:" in prompt
    assert "QUESTION: What are the supply chain risks?" in prompt


def test_parse_citations_dedupes_and_orders() -> None:
    assert parse_citations("First [2], then [1], again [2].", n_sources=3) == [2, 1]


def test_parse_citations_drops_out_of_range() -> None:
    # [5] is hallucinated when only 2 sources exist.
    assert parse_citations("See [1] and [5].", n_sources=2) == [1]


# --- pipeline ----------------------------------------------------------------


def test_answer_happy_path_parses_citations_and_attaches_sources() -> None:
    chunks = [_chunk(1), _chunk(2)]
    pipe = RAGPipeline(
        retriever=_FakeRetriever(chunks),
        generator=_FakeGenerator("Supply is concentrated in Asia [1] and tariffs bite [2]."),
    )
    ans = pipe.answer("supply chain risk?", ticker="AAPL", k=2)
    assert isinstance(ans, Answer)
    assert ans.cited_indices == [1, 2]
    assert ans.sources == chunks
    assert ans.model == "fake-model"
    assert len(ans.cited_sources()) == 2


def test_answer_passes_filters_to_retriever() -> None:
    retriever = _FakeRetriever([_chunk(1)])
    pipe = RAGPipeline(retriever=retriever, generator=_FakeGenerator("Answer [1]."))
    pipe.answer("q", ticker="JPM", fiscal_year=2023, section="Item 7", k=4)
    assert retriever.last_kwargs["ticker"] == "JPM"
    assert retriever.last_kwargs["fiscal_year"] == 2023
    assert retriever.last_kwargs["section"] == "Item 7"
    assert retriever.last_kwargs["k"] == 4


def test_answer_short_circuits_when_no_sources() -> None:
    gen = _FakeGenerator("should not be called")
    pipe = RAGPipeline(retriever=_FakeRetriever([]), generator=gen)
    ans = pipe.answer("something obscure")
    assert ans.sources == []
    assert ans.cited_indices == []
    assert "don't have enough information" in ans.text
    assert gen.last_prompt is None  # generator was never invoked


def test_answer_total_tokens_sums_usage() -> None:
    pipe = RAGPipeline(
        retriever=_FakeRetriever([_chunk(1)]),
        generator=_FakeGenerator("Answer [1]."),
    )
    ans = pipe.answer("q")
    assert ans.prompt_tokens == 10
    assert ans.completion_tokens == 5


# --- generator factory -------------------------------------------------------


def _settings(**over: object) -> Settings:
    base: dict[str, object] = {
        "sec_user_agent": "Test User test@example.com",
        "xai_api_key": None,
        "openrouter_api_key": None,
        "ollama_base_url": None,
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def test_build_generator_prefers_grok_when_key_present() -> None:
    gen = build_generator(_settings(xai_api_key="xai-abc", xai_model="grok-4.3"))
    assert gen._model == "grok-4.3"  # type: ignore[attr-defined]


def test_build_generator_uses_openrouter_when_only_its_key_set() -> None:
    gen = build_generator(
        _settings(openrouter_api_key="or-abc", openrouter_model="x-ai/grok-2-1212")
    )
    assert gen._model == "x-ai/grok-2-1212"  # type: ignore[attr-defined]


def test_build_generator_grok_beats_openrouter() -> None:
    # Both set -> Grok wins by precedence.
    gen = build_generator(
        _settings(xai_api_key="xai-abc", xai_model="grok-4.3", openrouter_api_key="or-abc")
    )
    assert gen._model == "grok-4.3"  # type: ignore[attr-defined]


def test_build_generator_falls_back_to_ollama() -> None:
    gen = build_generator(_settings(ollama_base_url="http://localhost:11434"))
    assert gen._model  # type: ignore[attr-defined]


def test_build_generator_raises_when_nothing_configured() -> None:
    with pytest.raises(RuntimeError, match="No generator configured"):
        build_generator(_settings())
