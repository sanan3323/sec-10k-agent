"""Render an EvalReport as Markdown (for humans / CI logs) and JSON (for diffing
baselines over time)."""

from __future__ import annotations

from sec_10k_agent.eval.runner import BucketAggregate, EvalReport


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.0f}%" if pct else f"{value:.2f}"


def _row(name: str, agg: BucketAggregate) -> str:
    return (
        f"| {name} | {agg.n} | {_fmt(agg.faithfulness_mean)} | {_fmt(agg.correctness_mean)} "
        f"| {_fmt(agg.context_recall_mean, pct=True)} | {_fmt(agg.hit_at_k_rate, pct=True)} "
        f"| {_fmt(agg.mrr)} |"
    )


def format_markdown(report: EvalReport) -> str:
    """Human-readable summary: config, aggregate table, and any errors."""
    cfg = report.config
    lines = [
        "# Eval report",
        "",
        f"- Run: `{report.created_at}`",
        f"- Items: {cfg.get('n_items')}  ·  k={cfg.get('k')}  ·  "
        f"filters={cfg.get('use_filters')}  ·  judged={cfg.get('judged')}",
        f"- Tokens: {report.total_prompt_tokens} prompt + "
        f"{report.total_completion_tokens} completion",
        "",
        "| Bucket | n | Faithful | Correct | Ctx recall | Hit@k | MRR |",
        "|---|---|---|---|---|---|---|",
        _row("**overall**", report.overall),
    ]
    for bucket, agg in report.by_bucket.items():
        lines.append(_row(bucket, agg))

    errors = [r for r in report.results if r.error]
    if errors:
        lines += ["", f"## Errors ({len(errors)})", ""]
        lines += [f"- `{r.id}`: {r.error}" for r in errors]

    lines += ["", "_Faithful/Correct are judge scores in [0,1]; Ctx recall / Hit@k are retrieval._"]
    return "\n".join(lines) + "\n"


def to_json(report: EvalReport) -> str:
    """Full report as indented JSON (per-item detail included) for baselines."""
    return report.model_dump_json(indent=2)
