"""LLM-as-judge for answer quality.

Two metrics, each a single judge call with a versioned prompt:

- faithfulness: is every claim in the answer supported by the retrieved
  context? This catches hallucination and needs no ground truth — only the
  answer and the context it was supposedly drawn from.
- correctness: does the answer address the question and match the reference?
  For single_fact items the reference is the ground-truth `answer`; for
  synthesis/temporal it is the `answer_rubric`.

The judge model must be a different family from the generator (ADR-003) so it
isn't grading its own style. Prompts are versioned so score changes can be
attributed to prompt edits vs system changes. The transport is the same
OpenAI-compatible generator the RAG layer uses — only the model differs.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from sec_10k_agent.config import Settings, get_settings
from sec_10k_agent.rag.llm import Generator, OpenAICompatGenerator

PROMPT_VERSION = "v1"
_MAX_SCORE = 5

FAITHFULNESS_PROMPT = """You are a strict evaluator checking whether an ANSWER is supported by the CONTEXT.

Score how well every factual claim in the ANSWER is grounded in the CONTEXT, on a 0-5 scale:
- 5: every claim is directly supported by the context.
- 3: mostly supported, with minor unsupported detail.
- 0: the answer is largely fabricated or contradicts the context.
An answer that correctly says the information is not present, when the context indeed lacks it, is faithful (score 5).

Respond with ONLY a JSON object: {{"score": <int 0-5>, "reasoning": "<one sentence>"}}

CONTEXT:
{context}

ANSWER:
{answer}"""

CORRECTNESS_PROMPT = """You are a strict evaluator grading an ANSWER to a QUESTION about SEC 10-K filings.

Grade the ANSWER against the REFERENCE ({reference_kind}) on a 0-5 scale:
- 5: fully correct and complete relative to the reference.
- 3: partially correct or missing important points.
- 0: incorrect, irrelevant, or contradicts the reference.
If the reference says the filings contain no such information, an answer that appropriately says it cannot find the information should score 5, and an answer that fabricates details should score 0.

Respond with ONLY a JSON object: {{"score": <int 0-5>, "reasoning": "<one sentence>"}}

QUESTION:
{question}

REFERENCE ({reference_kind}):
{reference}

ANSWER:
{answer}"""


class JudgeScore(BaseModel):
    """One metric's verdict for one answer. `score` is normalized to [0, 1]."""

    metric: str
    score: float
    raw_score: int
    max_score: int = _MAX_SCORE
    reasoning: str = ""
    prompt_version: str = PROMPT_VERSION

    def passed_at(self, threshold: float) -> bool:
        return self.score >= threshold


def parse_score_json(text: str, max_score: int = _MAX_SCORE) -> tuple[int, str]:
    """Extract (raw_score, reasoning) from a judge reply, tolerating code fences
    and surrounding prose. Clamps the score to [0, max_score]."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text!r}")
    obj = json.loads(match.group(0))
    raw = round(float(obj["score"]))
    raw = max(0, min(max_score, raw))
    reasoning = str(obj.get("reasoning", "")).strip()
    return raw, reasoning


class Judge:
    """Scores answers with an LLM. Inject a fake `generator` in tests."""

    def __init__(self, generator: Generator) -> None:
        self._gen = generator

    def _score(self, metric: str, prompt: str) -> JudgeScore:
        resp = self._gen.complete(
            "You are a precise evaluation model. Output only the requested JSON.",
            prompt,
            temperature=0.0,
        )
        raw, reasoning = parse_score_json(resp.text)
        return JudgeScore(
            metric=metric,
            raw_score=raw,
            score=raw / _MAX_SCORE,
            reasoning=reasoning,
        )

    def faithfulness(self, answer: str, context: str) -> JudgeScore:
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        return self._score("faithfulness", prompt)

    def correctness(
        self, question: str, answer: str, reference: str, *, is_rubric: bool
    ) -> JudgeScore:
        prompt = CORRECTNESS_PROMPT.format(
            reference_kind="grading rubric" if is_rubric else "reference answer",
            question=question,
            reference=reference,
            answer=answer,
        )
        return self._score("correctness", prompt)


def build_judge(settings: Settings | None = None) -> Judge:
    """Construct the judge, preferring a Gemini-family model (different family
    from the generator). Precedence: Gemini API -> OpenRouter -> Ollama."""
    settings = settings or get_settings()
    timeout = settings.llm_timeout_seconds
    if settings.gemini_api_key:
        gen: Generator = OpenAICompatGenerator(
            model=settings.gemini_judge_model,
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout,
        )
    elif settings.openrouter_api_key:
        gen = OpenAICompatGenerator(
            model=settings.openrouter_judge_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=timeout,
        )
    elif settings.ollama_base_url:
        base = settings.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        gen = OpenAICompatGenerator(model=settings.ollama_model, api_key="ollama", base_url=base)
    else:
        raise RuntimeError(
            "No judge configured. Set GEMINI_API_KEY, OPENROUTER_API_KEY, or OLLAMA_BASE_URL."
        )
    return Judge(gen)
