"""Tests for the eval runner and report, with a faked pipeline and judge."""

from __future__ import annotations

from sec_10k_agent.eval.dataset import CiteSpec, GoldenItem
from sec_10k_agent.eval.judge import JudgeScore
from sec_10k_agent.eval.report import format_markdown
from sec_10k_agent.eval.runner import run_eval
from sec_10k_agent.rag.models import Answer
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk(ticker: str, section: str, fy: int = 2024) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{ticker}__{section}__0001",
        ticker=ticker,
        fiscal_year=fy,
        section=section,
        text="body",
        distance=0.1,
    )


class _FakePipeline:
    """Returns a canned answer with a fixed AAPL Item 1A source, ignoring inputs."""

    def __init__(self, cited: bool = True) -> None:
        self._cited = cited

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        k: int = 5,
    ) -> Answer:
        sources = [_chunk("AAPL", "Item 1A")]
        return Answer(
            question=question,
            text="Grounded answer [1].",
            sources=sources,
            cited_indices=[1] if self._cited else [],
            model="fake-gen",
            prompt_tokens=100,
            completion_tokens=20,
        )


class _RaisingPipeline:
    def answer(self, question: str, **kwargs: object) -> Answer:
        raise RuntimeError("db down")


class _FakeJudge:
    def faithfulness(self, answer: str, context: str) -> JudgeScore:
        return JudgeScore(metric="faithfulness", raw_score=5, score=1.0)

    def correctness(
        self, question: str, answer: str, reference: str, *, is_rubric: bool
    ) -> JudgeScore:
        return JudgeScore(metric="correctness", raw_score=4, score=0.8)


def _item(id_: str, ticker: str, section: str, bucket: str = "single_fact") -> GoldenItem:
    kwargs: dict[str, object] = (
        {"answer": "ref"} if bucket == "single_fact" else {"answer_rubric": "rubric"}
    )
    return GoldenItem(
        id=id_,
        bucket=bucket,  # type: ignore[arg-type]
        question="q?",
        must_cite=[CiteSpec(ticker=ticker, section=section)],
        **kwargs,
    )


def test_run_eval_scores_and_aggregates() -> None:
    items = [
        _item("a", "AAPL", "Item 1A", "single_fact"),
        _item("b", "AAPL", "Item 1A", "synthesis"),
    ]
    report = run_eval(items, _FakePipeline(), _FakeJudge(), k=5)  # type: ignore[arg-type]

    assert len(report.results) == 2
    # Retrieval hit both times (fake sources match must_cite).
    assert report.overall.context_recall_mean == 1.0
    assert report.overall.hit_at_k_rate == 1.0
    assert report.overall.mrr == 1.0
    # Judge means.
    assert report.overall.faithfulness_mean == 1.0
    assert report.overall.correctness_mean == 0.8
    # Per-bucket split present.
    assert set(report.by_bucket) == {"single_fact", "synthesis"}
    # Token totals summed.
    assert report.total_prompt_tokens == 200
    assert report.total_completion_tokens == 40


def test_run_eval_retrieval_only_when_no_judge() -> None:
    report = run_eval([_item("a", "AAPL", "Item 1A")], _FakePipeline(), judge=None)  # type: ignore[arg-type]
    r = report.results[0]
    assert r.context_recall == 1.0
    assert r.faithfulness is None
    assert r.correctness is None
    assert report.overall.faithfulness_mean is None


def test_run_eval_records_per_item_errors_and_continues() -> None:
    items = [_item("a", "AAPL", "Item 1A"), _item("b", "NVDA", "Item 1A")]
    report = run_eval(items, _RaisingPipeline(), _FakeJudge())  # type: ignore[arg-type]
    assert all(r.error == "db down" for r in report.results)
    # Aggregates over empty metric lists are None, not a crash.
    assert report.overall.context_recall_mean is None


def test_run_eval_miss_lowers_recall() -> None:
    # must_cite asks for JPM but the fake pipeline returns AAPL -> miss.
    item = _item("a", "JPM", "Item 1A")
    report = run_eval([item], _FakePipeline(), _FakeJudge())  # type: ignore[arg-type]
    assert report.results[0].context_recall == 0.0
    assert report.results[0].hit_at_k is False


def test_format_markdown_renders_table_and_handles_none() -> None:
    report = run_eval([_item("a", "AAPL", "Item 1A")], _FakePipeline(), judge=None)  # type: ignore[arg-type]
    md = format_markdown(report)
    assert "# Eval report" in md
    assert "| Bucket | n |" in md
    assert "overall" in md
    # Judge columns are em-dashes when not judged.
    assert "—" in md
