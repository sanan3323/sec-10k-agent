"""Embed chunks and load them into pgvector.

The canonical, reproducible path from `data/processed/chunks.parquet` to a
queryable `text_chunks` table: embed the `text` column with fastembed
(BAAI/bge-large-en-v1.5, ONNX — no torch) and bulk-load with the vector column.
This replaces the earlier Colab/sentence-transformers embedding step so the
corpus can be rebuilt anywhere the core deps are installed (including CI).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from sec_10k_agent.config import get_settings

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024

# Columns loaded into text_chunks, matching scripts/postgres-init.sql + Chunk.
CHUNK_COLUMNS = [
    "chunk_id",
    "cik",
    "ticker",
    "fiscal_year",
    "accession_number",
    "section",
    "section_title",
    "text",
    "token_count",
    "prev_chunk_id",
    "next_chunk_id",
]


def select_chunk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the known chunk columns present in `df`, preserving order.

    Raises if the required `chunk_id`/`text` columns are missing.
    """
    for required in ("chunk_id", "text"):
        if required not in df.columns:
            raise ValueError(f"chunks frame missing required column {required!r}")
    keep = [c for c in CHUNK_COLUMNS if c in df.columns]
    return df[keep].copy()


def embed_texts(
    texts: list[str], model_name: str = EMBED_MODEL, batch_size: int = 256
) -> list[list[float]]:
    """Embed texts into BGE vectors as plain float lists (pgvector-ready)."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name)
    return [[float(x) for x in vec] for vec in model.embed(texts, batch_size=batch_size)]


def index_chunks(
    chunks_path: Path | None = None,
    dsn: str | None = None,
    model_name: str = EMBED_MODEL,
    truncate: bool = True,
) -> int:
    """Embed `chunks.parquet` and load into text_chunks. Returns the row count.

    With `truncate=True` (default) the table is cleared first, so the load is
    idempotent — re-running produces the same corpus rather than duplicating it.
    """
    settings = get_settings()
    chunks_path = chunks_path or (settings.processed_dir / "chunks.parquet")
    dsn = dsn or settings.postgres_dsn
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. Run `sec10k chunk` first."
        )

    from pgvector.sqlalchemy import Vector

    df = select_chunk_columns(pd.read_parquet(chunks_path))
    df["embedding"] = embed_texts(df["text"].tolist(), model_name)

    engine = create_engine(dsn)
    if truncate:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE text_chunks"))
    df.to_sql(
        "text_chunks",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=100,
        dtype={"embedding": Vector(EMBED_DIM)},
    )
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT count(*) FROM text_chunks")).scalar_one())
