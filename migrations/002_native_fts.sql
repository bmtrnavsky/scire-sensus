-- 002_native_fts.sql -- native FTS: stored tsvector + GIN, OR matching, RRF hybrid in SQL.
--
-- Why: the old FTS leg used plainto_tsquery (AND of every term) over an
-- expression GIN index. On the 30-query golden set 13-18/30 queries matched
-- zero FTS rows, so "hybrid" silently ran as vector-only, and OR-matching
-- over the expression index took 1.2-2.5s (seq recompute) vs 0.02-0.18s.
-- A stored GENERATED column makes the GIN index directly usable and OR
-- queries fast. Measured on eval/queries.jsonl: OR + ts_rank(...,1) matches
-- AND recall (hit@8 0.967) with MRR +0.010 and zero-FTS queries 13 -> 0.
--
-- Safe to re-run (IF NOT EXISTS / OR REPLACE / IF EXISTS).

-- 1. Stored tsvector, kept in sync by Postgres on every INSERT/UPDATE.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

-- 2. GIN index on the stored column (replaces the expression index below).
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

-- 3. Drop the now-redundant expression index (same posting data, maintained twice).
DROP INDEX IF EXISTS idx_chunks_fts;

ANALYZE chunks;

-- 4. Hybrid search: vector top (k*3) + FTS top (k*3), RRF k=60, all in SQL.
-- Signature and RETURNS are unchanged, so rag_flask_mcp.py needs no edit.
-- FTS leg: OR of the plainto stemmed lexemes (stopwords already dropped),
-- ranked with ts_rank normalization 1 (length-norm; the eval-backed choice).
-- NULLIF guards stopword-only queries (tq IS NULL -> c.tsv @@ NULL is NULL
-- -> no FTS rows, vector leg still returns).
CREATE OR REPLACE FUNCTION public.hybrid_search(query_text text, query_vec halfvec, top_k integer DEFAULT 8, filter_domain text DEFAULT NULL::text, exclude_domain text DEFAULT NULL::text)
 RETURNS TABLE(chunk_id bigint, source_path text, chunk_text text, heading_path text, page_number integer, domain text, rrf_score double precision, vec_score double precision, fts_score double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    WITH q AS (
        SELECT NULLIF(replace(plainto_tsquery('english', query_text)::text, ' & ', ' | '), '')::tsquery AS tq
    ),
    vec AS (
        SELECT c.id,
               (1 - (c.embedding <=> query_vec))::FLOAT AS score,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_vec)::FLOAT AS rank
        FROM chunks c
        WHERE (filter_domain IS NULL OR c.domain = filter_domain)
          AND (exclude_domain IS NULL OR c.domain IS DISTINCT FROM exclude_domain)
        ORDER BY c.embedding <=> query_vec
        LIMIT top_k * 3
    ),
    fts AS (
        SELECT c.id,
               ts_rank(c.tsv, q.tq, 1)::FLOAT AS score,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(c.tsv, q.tq, 1) DESC
               )::FLOAT AS rank
        FROM chunks c, q
        WHERE c.tsv @@ q.tq
          AND (filter_domain IS NULL OR c.domain = filter_domain)
          AND (exclude_domain IS NULL OR c.domain IS DISTINCT FROM exclude_domain)
        ORDER BY ts_rank(c.tsv, q.tq, 1) DESC
        LIMIT top_k * 3
    ),
    rrf AS (
        SELECT COALESCE(vec.id, fts.id) AS id,
               (COALESCE(1.0/(60+vec.rank), 0) + COALESCE(1.0/(60+fts.rank), 0))::FLOAT AS rrf_score,
               COALESCE(vec.score, 0)::FLOAT AS vec_score,
               COALESCE(fts.score, 0)::FLOAT AS fts_score
        FROM vec FULL OUTER JOIN fts ON vec.id = fts.id
    )
    SELECT c.id, sf.file_path, c.chunk_text, c.heading_path, c.page_number, c.domain,
           r.rrf_score, r.vec_score, r.fts_score
    FROM rrf r
    JOIN chunks c        ON c.id = r.id
    JOIN source_files sf ON sf.id = c.source_id
    ORDER BY r.rrf_score DESC
    LIMIT top_k;
END;
$function$;
