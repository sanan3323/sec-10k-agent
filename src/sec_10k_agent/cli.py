"""Command-line interface.

Subcommands:
    version       Print the installed package version.
    download      Download 10-K filings to data/raw/.
    parse         Parse cached HTML into ParsedFiling intermediates.
    chunk         Chunk parsed filings into data/processed/chunks.parquet.

Phase 1 will add `xbrl`. Phase 2 adds `index`.
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


if __name__ == "__main__":
    app()
