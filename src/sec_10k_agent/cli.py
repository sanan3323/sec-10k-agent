"""Command-line interface.

Subcommands:
    version       Print the installed package version.
    download      Download 10-K filings to data/raw/.
    parse         Parse cached HTML into ParsedFiling intermediates.
    chunk         Chunk parsed filings into data/processed/chunks.parquet.
    xbrl          Extract structured XBRL facts to data/processed/xbrl.parquet.
    ask           Ask a question over the indexed corpus (single-hop RAG).
"""

from __future__ import annotations

import logging

import typer

from sec_10k_agent import __version__
from sec_10k_agent.config import get_settings
from sec_10k_agent.ingestion import EdgarClient, FilingNotFound
from sec_10k_agent.scope import FISCAL_YEARS, TICKERS

app = typer.Typer(
    name="sec10k",
    help="SEC 10-K Q&A agent CLI.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def download(
    tickers: str = typer.Option(
        "",
        "--tickers",
        "-t",
        help="Comma-separated tickers, e.g. 'AAPL,NVDA'. Defaults to all in scope.",
    ),
    years: str = typer.Option(
        "",
        "--years",
        "-y",
        help="Comma-separated fiscal years, e.g. '2024,2025'. Defaults to all in scope.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download 10-K filings to data/raw/.

    With no flags, downloads every (ticker, fiscal_year) in the project
    scope. Already-cached filings are skipped without hitting EDGAR.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    target_tickers = _split_csv(tickers, default=list(TICKERS))
    target_years = [int(y) for y in _split_csv(years, default=[str(y) for y in FISCAL_YEARS])]

    settings = get_settings()
    client = EdgarClient(
        user_agent=settings.sec_user_agent,
        rate_limit_per_sec=settings.sec_rate_limit_per_sec,
        cache_dir=settings.raw_dir,
    )

    ok = 0
    skipped = 0
    failed: list[tuple[str, int, str]] = []
    for ticker in target_tickers:
        for fy in target_years:
            try:
                filing = client.get_10k(ticker, fy)
                ok += 1
                typer.echo(
                    f"  OK    {ticker} FY{fy}  "
                    f"acc={filing.accession_number}  "
                    f"period={filing.period_of_report}"
                )
            except FilingNotFound as e:
                skipped += 1
                typer.echo(f"  SKIP  {ticker} FY{fy}  {e}", err=True)
            except Exception as e:
                failed.append((ticker, fy, str(e)))
                typer.echo(f"  FAIL  {ticker} FY{fy}  {e}", err=True)

    typer.echo(f"\nDone. ok={ok}  skipped={skipped}  failed={len(failed)}")
    if failed:
        raise typer.Exit(code=1)


def _split_csv(value: str, default: list[str]) -> list[str]:
    if not value.strip():
        return default
    return [item.strip().upper() for item in value.split(",") if item.strip()]


@app.command()
def parse(
    force: bool = typer.Option(
        False, "--force", help="Re-parse even if a cached intermediate exists."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parse every cached 10-K HTML into a ParsedFiling intermediate.

    Reads from data/raw/, writes to data/processed/parsed/. Skips filings
    whose cached intermediate is up-to-date with the current parser version.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from sec_10k_agent.ingestion.parser import parse_all_cached

    settings = get_settings()
    parsed, skipped, failures = parse_all_cached(
        raw_dir=settings.raw_dir,
        processed_dir=settings.processed_dir,
        force=force,
    )
    typer.echo(f"Parsed {parsed}, skipped {skipped}, failed {len(failures)}")
    for accession, msg in failures:
        typer.echo(f"  FAIL  {accession}: {msg}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def chunk(
    use_word_count: bool = typer.Option(
        False,
        "--word-count",
        help="Use word-count token counter (fast, no model download). Default: BGE tokenizer.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Chunk every cached ParsedFiling into data/processed/chunks.parquet.

    By default uses the BGE tokenizer to size chunks correctly for the
    embedding model. Pass `--word-count` for a faster but approximate run.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import pandas as pd

    from sec_10k_agent.ingestion import (
        BgeTokenCounter,
        Chunker,
        ParsedFiling,
        WordCountTokenCounter,
    )

    settings = get_settings()
    parsed_dir = settings.processed_dir / "parsed"
    if not parsed_dir.exists():
        typer.echo(f"No parsed filings at {parsed_dir}. Run `sec10k parse` first.", err=True)
        raise typer.Exit(code=1)

    counter = WordCountTokenCounter() if use_word_count else BgeTokenCounter()
    chunker = Chunker(token_counter=counter)

    all_rows: list[dict] = []
    for path in sorted(parsed_dir.glob("*.json")):
        parsed = ParsedFiling.model_validate_json(path.read_text(encoding="utf-8"))
        chunks = chunker.chunk(parsed)
        all_rows.extend(c.model_dump() for c in chunks)
        typer.echo(
            f"  OK   {parsed.filing.ticker} FY{parsed.filing.fiscal_year}  chunks={len(chunks)}"
        )

    if not all_rows:
        typer.echo("No parsed filings found.", err=True)
        raise typer.Exit(code=1)

    out_path = settings.processed_dir / "chunks.parquet"
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    typer.echo(f"\nWrote {len(all_rows)} chunks to {out_path}")


@app.command()
def xbrl(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Extract structured XBRL facts to data/processed/xbrl.parquet.

    For each downloaded filing, fetches the structured XBRL data from EDGAR
    (cached on first run), then converts to validated XBRLFact rows. Output
    captures dimensional axes (geographic, segment, product) — the
    architecture doc's acceptance test depends on this.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from sec_10k_agent.ingestion import EdgarClient, extract_all_cached

    settings = get_settings()
    client = EdgarClient(
        user_agent=settings.sec_user_agent,
        rate_limit_per_sec=settings.sec_rate_limit_per_sec,
        cache_dir=settings.raw_dir,
    )

    n_filings, n_facts, failures = extract_all_cached(
        edgar_client=client,
        raw_dir=settings.raw_dir,
        processed_dir=settings.processed_dir,
    )
    typer.echo(f"\nDone. filings={n_filings}  facts={n_facts}  failed={len(failures)}")
    for accession, msg in failures:
        typer.echo(f"  FAIL  {accession}: {msg}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer over the 10-K corpus."),
    ticker: str = typer.Option("", "--ticker", "-t", help="Restrict to one ticker, e.g. AAPL."),
    year: int = typer.Option(0, "--year", "-y", help="Restrict to one fiscal year, e.g. 2024."),
    section: str = typer.Option(
        "", "--section", "-s", help="Restrict to one 10-K item, e.g. 'Item 1A'."
    ),
    k: int = typer.Option(5, "--k", "-k", help="Number of chunks to retrieve."),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Print the retrieved sources under the answer."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Answer a question over the indexed corpus with cited sources.

    Requires the pgvector corpus (Phase 2) and a generator: set XAI_API_KEY for
    Grok or OLLAMA_BASE_URL for a local model. Filters map straight onto the
    retriever's pre-filter.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from sec_10k_agent.rag import RAGPipeline

    pipeline = RAGPipeline()
    answer = pipeline.answer(
        question,
        ticker=ticker or None,
        fiscal_year=year or None,
        section=section or None,
        k=k,
    )

    typer.echo("\n" + answer.text.strip() + "\n")
    if show_sources and answer.sources:
        cited = set(answer.cited_indices)
        typer.echo("Sources:")
        for i, src in enumerate(answer.sources, start=1):
            mark = "*" if i in cited else " "
            typer.echo(f"  [{i}]{mark} {src.citation()}  (score={src.score:.3f})")
    total_tokens = (answer.prompt_tokens or 0) + (answer.completion_tokens or 0)
    if total_tokens:
        typer.echo(f"\n({answer.model}, {total_tokens} tokens)")


if __name__ == "__main__":
    app()
