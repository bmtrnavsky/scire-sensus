-- schema.sql -- Scire Sensus core schema (Postgres + pgvector).
--
-- Run once on a fresh database (after CREATE EXTENSION vector):
--   psql -d ragdb -f schema.sql
-- Then optionally: research_schema.sql, migrations/002_native_fts.sql
-- (002 adds the stored-tsvector column and the hybrid_search() function).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid(), if needed

CREATE TABLE IF NOT EXISTS source_files (
    id           BIGSERIAL PRIMARY KEY,
    file_path    TEXT NOT NULL UNIQUE,
    file_hash    TEXT,
    doc_type     TEXT,             -- 'markdown' | 'pdf'
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    title        TEXT,
    domain       TEXT,             -- frontmatter domain, or 'timebound_context'
    note_type    TEXT,
    note_status  TEXT,
    tags         TEXT,
    content_date DATE,             -- set for timebound_context files; else NULL
    last_ingested TIMESTAMPTZ DEFAULT now(),
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    source_id    BIGINT REFERENCES source_files(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    chunk_text   TEXT NOT NULL,
    token_count  INTEGER,
    heading_path TEXT,
    page_number  INTEGER,          -- PDFs; NULL for Markdown
    domain       TEXT,
    embedding    halfvec(4000),   -- Qwen3-Embedding-8B, Matryoshka-truncated
    metadata     JSONB,
    created_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = '32', ef_construction = '128');

CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks (domain);

-- Cache of one short LLM summary per PDF (contextual-retrieval prefix).
CREATE TABLE IF NOT EXISTS pdf_doc_summaries (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT,
    summary   TEXT,
    model     TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
