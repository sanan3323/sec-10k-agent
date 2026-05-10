-- Postgres init script, runs once on first container start.
-- Mounted at /docker-entrypoint-initdb.d/01-init.sql by docker-compose.

-- Langfuse needs its own database. The application DB (sec10k) is created by
-- POSTGRES_DB, so we just create langfuse here.
-- 1. Create the Langfuse database
-- Since the container starts connected to 'sec10k' (from your ENV), 
-- this creates a second, empty database for the observability tool.
CREATE DATABASE langfuse;


CREATE EXTENSION IF NOT EXISTS vector;

-- 3. Financial Facts Table 
CREATE TABLE IF NOT EXISTS financial_facts (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    concept TEXT NOT NULL,
    value NUMERIC,
    unit VARCHAR(10),
    period_start DATE,
    period_end DATE,
    dimensions JSONB DEFAULT '{}'::jsonb,
    accession_number VARCHAR(25),
    -- Idempotency: prevents duplicate facts if you run the loader twice
    UNIQUE(accession_number, concept, dimensions, period_end)
);

-- 4. Text Chunks Table 
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    accession_number VARCHAR(25),
    section VARCHAR(50),
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE, -- Prevents duplicate chunks
    token_count INTEGER,
    embedding VECTOR(1024), -- 1024 dims for BGE-large-en-v1.5
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 5. Performance Indexes
-- GIN index for lightning-fast filtering on Apple segments like 'country:CN'
CREATE INDEX IF NOT EXISTS idx_facts_dimensions ON financial_facts USING GIN (dimensions);
-- HNSW index for high-speed vector similarity search (Phase 3)
CREATE INDEX IF NOT EXISTS idx_chunks_vector ON text_chunks USING hnsw (embedding vector_cosine_ops);
