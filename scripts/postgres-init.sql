-- Postgres init script, runs once on first container start.
-- Mounted at /docker-entrypoint-initdb.d/01-init.sql by docker-compose.
--
-- This schema is the source of truth for a fresh DB and matches the tables the
-- Phase 1/2 loaders (ingestion/load_vectors.py, ingestion/db_loader.py) write
-- to. Keep it in sync with src/sec_10k_agent/ingestion/models.py (Chunk,
-- XBRLFact) and the live database. See ADR-001.

-- Langfuse needs its own database. The application DB (sec10k) is created by
-- POSTGRES_DB; we just create the observability DB here.
CREATE DATABASE langfuse;

CREATE EXTENSION IF NOT EXISTS vector;

-- Text chunks: one retrievable unit of 10-K prose per row. Mirrors the Chunk
-- model. `text` holds the chunk body; prev/next_chunk_id form the reading-order
-- chain used for context expansion during retrieval.
CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id         TEXT PRIMARY KEY,
    cik              TEXT,
    ticker           TEXT,
    fiscal_year      INTEGER,
    accession_number TEXT,
    section          TEXT,
    section_title    TEXT,
    text             TEXT NOT NULL,
    token_count      INTEGER,
    prev_chunk_id    TEXT,
    next_chunk_id    TEXT,
    embedding        VECTOR(1024)  -- BGE-large-en-v1.5
);

-- HNSW over cosine distance for dense retrieval.
CREATE INDEX IF NOT EXISTS text_chunks_embedding_hnsw
    ON text_chunks USING hnsw (embedding vector_cosine_ops);

-- Composite btree for the metadata pre-filter (ticker + fiscal_year + section).
CREATE INDEX IF NOT EXISTS text_chunks_filter
    ON text_chunks (ticker, fiscal_year, section);

-- Structured XBRL facts. Mirrors the XBRLFact model. `dimensions` is the
-- critical field for segment/geographic facts; the UNIQUE constraint makes the
-- loader idempotent across re-runs.
CREATE TABLE IF NOT EXISTS financial_facts (
    id               SERIAL PRIMARY KEY,
    cik              TEXT,
    ticker           TEXT NOT NULL,
    fiscal_year      INTEGER NOT NULL,
    concept          TEXT NOT NULL,
    value            NUMERIC,
    unit             TEXT,
    period_start     DATE,
    period_end       DATE,
    dimensions       JSONB DEFAULT '{}'::jsonb,
    accession_number TEXT,
    UNIQUE (accession_number, concept, dimensions, period_end)
);

-- GIN index for filtering on segments like {'srt:StatementGeographicalAxis': 'country:CN'}.
CREATE INDEX IF NOT EXISTS idx_facts_dimensions
    ON financial_facts USING GIN (dimensions);
