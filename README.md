# Scire Sensus

Personal knowledge RAG: ingest a Markdown vault + PDF library into Postgres/pgvector, then serve hybrid (vector + full-text, RRF fusion) search over HTTP for agent tools.

*Scire sensus* — Latin for "to know through the senses / perception". A retrieval layer that lets an agent query what you have read and written.

## How it works

```
Markdown vault ──┐
                 ├─► ingest_rag_pgvector_v3.py ──► Postgres + pgvector ──► rag_flask_mcp.py (HTTP :5556) ──► agent tools
PDF library ─────┘                                                            ▲
                                                     Qwen3-Embedding-8B (OpenRouter) │ Cohere rerank
```

1. **Ingest** (`ingest_rag_pgvector_v3.py`) — heading-based Markdown chunking, PDF extraction (pdftotext → PyMuPDF → Tesseract OCR fallback), one short LLM summary per PDF prepended before embedding (contextual retrieval, PDF-only), chunk embeddings via `qwen/qwen3-embedding-8b` on OpenRouter (Matryoshka-truncated to 4000 dims to fit pgvector HNSW). Per-file transactions, content-hash skip, purge-safety guards.
2. **Serve** (`rag_flask_mcp.py`) — Flask API on `127.0.0.1:5556` with five tools: `rag_search` (hybrid RRF + Cohere rerank), `rag_semantic` (pure vector), `rag_search_time_bound` (daily-notes/archives slice, semantic or chronological), `research_rag_search` / `research_rag_semantic` (isolated per-project research corpus).
3. **Evaluate** (`eval/`) — golden-query harness (`run_eval.py`, hit@k / MRR), ablation variant builder (`build_variant.py`), rerank eval, and the audit report (`AUDIT.md`) documenting measured retrieval fixes.

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
python -m unittest test_ingest_purge_guard -v   # no DB or network needed
python -m unittest test_ocr_and_format -v       # needs Tesseract + poppler
python test_rag.py                              # smoke test against a live index
```

## Eval

`eval/sample_queries.jsonl` is a 5-query sample in the golden-set format (`id, type, query, gold_files, answer`). Build your own full set from your corpus, then:

```bash
cd eval && python run_eval.py --help
```

See `eval/AUDIT.md` for the method and an example findings report.

## License

MIT — see [LICENSE](LICENSE).
