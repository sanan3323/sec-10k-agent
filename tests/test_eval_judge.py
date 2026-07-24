"""Tests for the LLM-as-judge. No network: the judge model is faked."""

from __future__ import annotations

import pytest

from sec_10k_agent.config import Settings
from sec_10k_agent.eval.judge import Judge, build_judge, parse_score_json
from sec_10k_agent.rag.models import LLMResponse


class _FakeGen:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text=self._text, model="fake-judge")


def test_parse_score_json_plain() -> None:
    raw, reasoning = parse_score_json('{"score": 4, "reasoning": "mostly grounded"}')
    assert raw == 4
    assert reasoning == "mostly grounded"


def test_parse_score_json_tolerates_code_fence_and_prose() -> None:
    text = 'Here is my verdict:\n```json\n{"score": 5, "reasoning": "ok"}\n```'
    raw, reasoning = parse_score_json(text)
    assert raw == 5
    assert reasoning == "ok"


def test_parse_score_json_clamps_out_of_range() -> None:
    raw, _ = parse_score_json('{"score": 9}')
    assert raw == 5
    raw, _ = parse_score_json('{"score": -2}')
    assert raw == 0


def test_parse_score_json_raises_without_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_score_json("the answer is good")


def test_faithfulness_normalizes_score() -> None:
    judge = Judge(_FakeGen('{"score": 5, "reasoning": "all supported"}'))
    result = judge.faithfulness(answer="Apple relies on Asia [1].", context="[1] ...Asia...")
    assert result.metric == "faithfulness"
    assert result.raw_score == 5
    assert result.score == 1.0
    assert result.passed_at(0.6)


def test_correctness_rubric_vs_answer_prompt_differs() -> None:
    gen = _FakeGen('{"score": 3, "reasoning": "partial"}')
    judge = Judge(gen)
    judge.correctness("q?", "ans", "the rubric", is_rubric=True)
    assert "grading rubric" in gen.calls[-1][1]
    judge.correctness("q?", "ans", "the answer", is_rubric=False)
    assert "reference answer" in gen.calls[-1][1]


def test_correctness_score_normalized() -> None:
    judge = Judge(_FakeGen('{"score": 3, "reasoning": "partial"}'))
    result = judge.correctness("q?", "ans", "ref", is_rubric=False)
    assert result.score == 0.6
    assert result.metric == "correctness"


# --- build_judge precedence --------------------------------------------------


def _settings(**over: object) -> Settings:
    base: dict[str, object] = {
        "sec_user_agent": "Test User test@example.com",
        "gemini_api_key": None,
        "openrouter_api_key": None,
        "ollama_base_url": None,
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def test_build_judge_uses_openrouter_gemini_when_available() -> None:
    judge = build_judge(_settings(openrouter_api_key="or-abc"))
    # Judge model should be the Gemini-family default, not the generator model.
    assert judge._gen._model == "google/gemini-2.5-flash"  # type: ignore[attr-defined]


def test_build_judge_prefers_gemini_api() -> None:
    judge = build_judge(_settings(gemini_api_key="g-abc", gemini_judge_model="gemini-2.5-flash"))
    assert judge._gen._model == "gemini-2.5-flash"  # type: ignore[attr-defined]


def test_build_judge_raises_when_nothing_configured() -> None:
    with pytest.raises(RuntimeError, match="No judge configured"):
        build_judge(_settings())
