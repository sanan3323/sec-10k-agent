"""Eval runner: execute the pipeline over the golden set and score it.

For each item: run the RAG pipeline, compute the deterministic retrieval metrics
against `must_cite`, then (optionally) judge faithfulness and correctness. Results
aggregate per bucket and overall into an EvalReport.

Baseline scope: items are run WITH their `filters` applied to retrieval, since
there is no router yet (Phase 6). This measures retrieval-ranking + generation
quality given correct routing. Re-run with use_filters=False once the router
lands to fold routing error into the numbers.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from sec_10k_agent.eval.dataset import GoldenItem
from sec_10k_agent.eval.judge import Judge
from sec_10k_agent.eval.retrieval_metrics import context_recall, hit_at_k, reciprocal_rank
from sec_10k_agent.rag.pipeline import RAGPipeline
from sec_10k_agent.rag.prompts import format_context


class ItemResult(BaseModel):
    """Scored outcome for one golden item."""

    id: str
    bucket: str
    tags: list[str] = Field(default_factory=list)
    answer: str = ""
    n_sources: int = 0
    n_cited: int = 0
    # Retrieval metrics (None when the item has no must_cite, e.g. negatives).
    context_recall: float | None = None
    reciprocal_rank: float | None = None
    hit_at_k: bool | None = None
    # Judge metrics (None when judging is disabled or not applicable).
    faithfulness: float | None = None
    correctness: float | None = None
    faithfulness_reasoning: str = ""
    correctness_reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class BucketAggregate(BaseModel):
    """Mean metrics over a set of items."""

    n: int
    faithfulness_mean: float | None = None
    correctness_mean: float | None = None
    context_recall_mean: float | None = None
    hit_at_k_rate: float | None = None
    mrr: float | None = None


class EvalReport(BaseModel):
    """Full eval run: per-item results plus aggregates and run metadata."""

    created_at: str
    config: dict[str, object] = Field(default_factory=dict)
    results: list[ItemResult] = Field(default_factory=list)
    overall: BucketAggregate
    by_bucket: dict[str, BucketAggregate] = Field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _aggregate(results: list[ItemResult]) -> BucketAggregate:
    faith = [r.faithfulness for r in results if r.faithfulness is not None]
    corr = [r.correctness for r in results if r.correctness is not None]
    recall = [r.context_recall for r in results if r.context_recall is not None]
    hits = [r.hit_at_k for r in results if r.hit_at_k is not None]
    rr = [r.reciprocal_rank for r in results if r.reciprocal_rank is not None]
    return BucketAggregate(
        n=len(results),
        faithfulness_mean=_mean(faith),
        correctness_mean=_mean(corr),
        context_recall_mean=_mean(recall),
        hit_at_k_rate=(sum(hits) / len(hits)) if hits else None,
        mrr=_mean(rr),
    )


def _score_item(
    item: GoldenItem,
    pipeline: RAGPipeline,
    judge: Judge | None,
    k: int,
    use_filters: bool,
) -> ItemResult:
    f = item.filters
    answer = pipeline.answer(
        item.question,
        ticker=f.ticker if (use_filters and f) else None,
        fiscal_year=f.fiscal_year if (use_filters and f) else None,
        section=f.section if (use_filters and f) else None,
        k=k,
    )
    result = ItemResult(
        id=item.id,
        bucket=item.bucket,
        tags=item.tags,
        answer=answer.text,
        n_sources=len(answer.sources),
        n_cited=len(answer.cited_indices),
        context_recall=context_recall(item, answer.sources),
        reciprocal_rank=reciprocal_rank(item, answer.sources),
        hit_at_k=hit_at_k(item, answer.sources, k=k),
        prompt_tokens=answer.prompt_tokens or 0,
        completion_tokens=answer.completion_tokens or 0,
    )
    if judge is not None:
        # Faithfulness only makes sense when there is retrieved context.
        if answer.sources:
            fs = judge.faithfulness(answer.text, format_context(answer.sources))
            result.faithfulness = fs.score
            result.faithfulness_reasoning = fs.reasoning
        cs = judge.correctness(
            item.question,
            answer.text,
            item.grading_reference(),
            is_rubric=item.bucket in ("synthesis", "temporal"),
        )
        result.correctness = cs.score
        result.correctness_reasoning = cs.reasoning
    return result


def run_eval(
    items: list[GoldenItem],
    pipeline: RAGPipeline,
    judge: Judge | None = None,
    *,
    k: int = 5,
    use_filters: bool = True,
    on_progress: Callable[[int, int, ItemResult], None] | None = None,
) -> EvalReport:
    """Run every item, score it, and aggregate. Per-item errors are captured so
    one failure doesn't abort the run."""
    results: list[ItemResult] = []
    for i, item in enumerate(items, start=1):
        try:
            result = _score_item(item, pipeline, judge, k, use_filters)
        except Exception as exc:  # keep going; record the failure
            result = ItemResult(id=item.id, bucket=item.bucket, tags=item.tags, error=str(exc))
        results.append(result)
        if on_progress is not None:
            on_progress(i, len(items), result)

    buckets = sorted({r.bucket for r in results})
    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "k": k,
            "use_filters": use_filters,
            "judged": judge is not None,
            "n_items": len(items),
        },
        results=results,
        overall=_aggregate(results),
        by_bucket={b: _aggregate([r for r in results if r.bucket == b]) for b in buckets},
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
    )
