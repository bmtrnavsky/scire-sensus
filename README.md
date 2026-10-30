# Scire Sensus

Personal knowledge RAG: ingest a Markdown vault + PDF library into Postgres/pgvector, then serve hybrid (vector + full-text, RRF fusion) search over HTTP for agent tools.

*Scire sensus* (SKEE-reh SEN-soos): Latin for "to know through the senses / perception". A retrieval layer that lets an agent query what you have read and written.

## What makes it different

Most personal RAG setups are tuned by feel. This one is tuned by measurement:

1. **Eval-measured, not vibe-tuned.** A 30-query golden set with answer-phrase judgments scores every change on hit@k and MRR before it ships. Fixes that help (query instruction format, document prefix removal) shipped; fixes that looked plausible but measured flat or negative (OR full-text queries alone, title prepending, larger PDF chunks, layout-preserving extraction) did not. The full audit trail is in `eval/AUDIT.md`.
2. **Hybrid retrieval with a rerank stage.** Dense vectors (Qwen3-Embedding-8B via OpenRouter, Matryoshka-truncated to 4000 dims for pgvector HNSW) plus native Postgres full-text search, fused with Reciprocal Rank Fusion (k=60), then reranked with Cohere Rerank 4 Pro over a 45-candidate pool with a garbage-trim cutoff. Either leg can be queried alone (`rag_semantic`) or fused (`rag_search`).
3. **Provenance on every hit.** Each result returns the source file path, heading path (Markdown) or page number (PDF), document type, and both the fusion and component scores, so the calling agent can cite where an answer came from and judge how much to trust it.
4. **Ingest is careful with your data.** Per-file transactions, content-hash skip (unchanged files are not re-embedded), purge-safety guards (a missing mount or a subset run can never trigger mass deletion), and PDF handling that degrades gracefully: pdftotext first, PyMuPDF second, Tesseract OCR last, with loud failures instead of silent gaps.
5. **Contextual retrieval for PDFs.** One short LLM summary per PDF (Mistral Small via OpenRouter, cached in `pdf_doc_summaries`) is prepended to every chunk of that document before embedding, so short chunks still carry document-level context. Markdown notes are self-titled and skip this step.

## How it works

```
Markdown vault --\
                +--> ingest_rag_pgvector_v3.py --> Postgres + pgvector --> rag_flask_mcp.py (HTTP :5556) --> agent tools
PDF library ----/                                                              ^
                                                     Qwen3-Embedding-8B (OpenRouter) | Cohere rerank
```

1. **Ingest** (`ingest_rag_pgvector_v3.py`) - heading-based Markdown chunking (400 words, 50 overlap), PDF chunking (500 words, 75 overlap) with per-document summaries, PDF extraction (pdftotext, PyMuPDF, Tesseract OCR fallback), chunk embeddings via `qwen/qwen3-embedding-8b` on OpenRouter. Per-file transactions, content-hash skip, purge-safety guards.
2. **Serve** (`rag_flask_mcp.py`) - Flask API on `127.0.0.1:5556` with five tools: `rag_search` (hybrid RRF + Cohere rerank), `rag_semantic` (pure vector), `rag_search_time_bound` (daily-notes/archives slice, semantic or chronological), `research_rag_search` / `research_rag_semantic` (isolated per-project research corpus).
3. **Evaluate** (`eval/`) - golden-query harness (`run_eval.py`, hit@k / MRR), ablation variant builder (`build_variant.py`), rerank eval, and the audit report (`AUDIT.md`) documenting measured retrieval fixes.

## Methodology

The eval harness (`eval/run_eval.py`) scores retrieval against a golden set of 30 queries (22 vault notes, 8 PDFs; 9 keyword, 21 paraphrase). Each query names its gold files plus answer phrases. A hit means a returned chunk comes from a gold file AND contains an answer phrase (case and whitespace normalised), so judgments hold across re-chunking.

Metrics: hit@1, hit@8, hit@20, and MRR@8, reported for the hybrid path, the pure-vector path, and the keyword slice separately, plus a count of queries with zero full-text matches (a diagnostic for FTS health).

Ablations run in shadow indexes (`rag_eval.chunks_<tag>`) rebuilt from current ingest code by `eval/build_variant.py`, with candidate fixes prototyped as `--patch` flags. Embeddings are cached (`rag_eval.embed_cache`) so variants are cheap. The live `public.chunks` table is never written by eval code. Exact vector search is used as a control wherever HNSW recall itself is under suspicion.

Decision rule: a change ships only if it moves a metric clearly above noise on the 30-query set (one query is worth 0.033, so only deltas of several points count). Everything else is recorded in `eval/AUDIT.md` with its measured effect and left out.

## Results

Live index, 30-query golden set, hybrid path unless noted. Full per-variant numbers in `eval/AUDIT.md`.

| Stage | hit@8 | MRR@8 | hit@1 | Notes |
|---|---|---|---|---|
| Baseline (deployed settings at audit start) | 0.900 | 0.643 | 0.467 | 14/30 queries match zero FTS rows |
| + Fix 1: correct Qwen query instruction | 0.900 | 0.753 | 0.633 | Query-side only, safe on existing index |
| + Fix 2: drop stale `search_document:` chunk prefix (full re-embed) | 0.967 | 0.718 | 0.567 | Vector hit@8 0.867 to 0.967 |
| + PDF 500-word rechunk + per-doc summaries | 0.967 | 0.769 | 0.667 | Keyword hit@8 1.000, k20 hit@20 1.000 |
| + Native FTS (stored tsvector, OR query, length-norm rank) | 0.967 | 0.779 | 0.667 | Zero-FTS queries 18 to 0 |

Not shipped (measured, rejected): OR-only FTS without re-embed (no net gain once Fix 2 landed, and slower on expression index), title prepending (MRR up but recall@8 down), 2000-word PDF chunks (mixed), `-layout` extraction (no recall gain), code-fence heading fix (no change on this set).

Caveat: n=30, so treat small deltas as noise. Only the Fix 1, Fix 2, and rechunk/summary deltas are clearly above it.

## Provenance

Every search hit returns:

- `file_path`: full source path of the document
- `heading_path`: Markdown section trail (e.g. `Methods > Sampling`), or
- `page_number`: PDF page the chunk came from
- `doc_type`: markdown or pdf (plus domain tags; daily-notes/archives carry `timebound_context` and are excluded from default search)
- `rrf_score`: fused rank score, plus the underlying vector and FTS component scores

The Flask formatter renders these as a header line per hit (`[i] filename | doc_type | section or page | score=...`) followed by the full path and chunk text, so agents get citable context in plain text with no extra lookup.

## Quickstart

```bash
# 1. Postgres with pgvector, e.g.
createdb ragdb
psql -d ragdb -c "CREATE EXTENSION vector;"

# 2. Schema
psql -d ragdb -f schema.sql
psql -d ragdb -f research_schema.sql   # optional: isolated research corpus
psql -d ragdb -f migrations/002_native_fts.sql

# 3. Configure
cp .env.example .env        # fill in DB_DSN + OPENROUTER_API_KEY
# Point at your sources: VAULT_ROOT=~/my-notes PDF_ROOT_DIR=~/my-pdfs

# 4. Install + ingest + serve
pip install -r requirements.txt
python ingest_rag_pgvector_v3.py
python rag_flask_mcp.py     # listens on 127.0.0.1:5556
```

System dependencies for PDF ingest: `pdftotext` (poppler) and optionally `tesseract` for scanned PDFs.

## Configuration

All tuning lives in `config.py`: embedding model/dims, chunk sizes, rerank model + pool + cutoff, exclude markers, and the `timebound_context` markers that route daily-notes/archives into `rag_search_time_bound` instead of default search. Source folders come from `VAULT_ROOT` / `PDF_ROOT_DIR` env vars (see `.env.example`).

## Tests

```bash
python -m unittest test_ingest_purge_guard -v   # no DB or network needed (13 tests: purge guards + end-to-end ingest with mocked DB)
python -m unittest test_ocr_and_format -v       # needs Tesseract + poppler (10 tests: OCR fallback, result formatting, tesseract resolution)
python test_rag.py                              # smoke test against a live index
```

Unit suite status (2026-10-30, Windows bare metal): `test_ingest_purge_guard` 13/13 pass, `test_ocr_and_format` 10/10 pass with dependencies installed.

## Eval

`eval/sample_queries.jsonl` is a 5-query sample in the golden-set format (`id, type, query, gold_files, answer`). Build your own full set from your corpus, then:

```bash
cd eval && python run_eval.py --help
```

See `eval/AUDIT.md` for the method and an example findings report.

## License

MIT - see [LICENSE](LICENSE).
