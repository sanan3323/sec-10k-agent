-- Postgres init script, runs once on first container start.
-- Mounted at /docker-entrypoint-initdb.d/01-init.sql by docker-compose.

-- Langfuse needs its own database. The application DB (sec10k) is created by
-- POSTGRES_DB, so we just create langfuse here.
CREATE DATABASE langfuse;

-- Enable pgvector on the application DB.
\c sec10k
CREATE EXTENSION IF NOT EXISTS vector;
