-- research_rag_chunks isolated schema — mirrors main chunks/source_files
-- pattern exactly, adds research-pipeline-specific fields. Isolated
-- table on the SAME Postgres instance/pgvector extension as main RAG,
-- never mixed into the main chunks table.

CREATE TABLE IF NOT EXISTS research_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    review_type TEXT NOT NULL,  -- systematic|scoping|critical|integrative|conceptual|case_study
    lens TEXT,                  -- academic|current_reality|balanced|combined
    protocol JSONB,             -- Woodward's full protocol (research question, inclusion/exclusion, search strings)
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research_source_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES research_projects(id),
    file_path TEXT,
    file_hash TEXT,
    doi TEXT,
    title TEXT,
    year INTEGER,
    source_type TEXT,           -- peer_reviewed|grey_lit
    screening_status TEXT DEFAULT 'candidate',  -- candidate|included|excluded|paywalled
    exclusion_reason TEXT,      -- ERR_NO_ARTIFACT etc, from Woodward's protocol
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB,
    last_ingested TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research_rag_chunks (
    id BIGSERIAL PRIMARY KEY,
    source_id UUID REFERENCES research_source_files(id),
    project_id UUID REFERENCES research_projects(id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER,
    heading_path TEXT,
    section_name TEXT,          -- e.g. Introduction/Methodology/Discussion
    page_number INTEGER,
    doi TEXT,
    year INTEGER,
    source_type TEXT,           -- peer_reviewed|grey_lit, mirrors source_files
    embedding halfvec(4000),    -- same Qwen3-embedding-8b dimension as main RAG
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding
    ON research_rag_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m='32', ef_construction='128');

CREATE INDEX IF NOT EXISTS idx_research_chunks_fts
    ON research_rag_chunks USING gin (to_tsvector('english', chunk_text));

CREATE INDEX IF NOT EXISTS idx_research_chunks_project
    ON research_rag_chunks (project_id);

CREATE INDEX IF NOT EXISTS idx_research_chunks_doi
    ON research_rag_chunks (doi);
