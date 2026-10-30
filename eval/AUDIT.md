# RAG retrieval audit — 2026-10-29

Golden set: `queries.jsonl`, 30 queries (22 vault notes, 8 PDFs; 9 keyword, 21 paraphrase).
A hit means a chunk from a gold file that contains the answer phrase, so the judgments hold
across re-chunking. Exact vector search is used everywhere. The deployed `hybrid_search()`
(HNSW, ef_search=40) scored the same hit@8 as exact search (0.900), so HNSW recall isn't a defect.

Shadow indexes (`rag_eval.chunks_<tag>`) are rebuilt from the current ingest code by
`build_variant.py`, with candidate fixes prototyped as `--patch`. Embeddings are cached in
`rag_eval.embed_cache`. `public.chunks` was never written.

## Baseline (live index, deployed settings)
hybrid hit@8 0.900 · MRR@8 0.643 · hit@1 0.467 · 14/30 queries match zero FTS rows (AND tsquery).

## Defects ranked by measured impact

| # | Defect | Where | Measured effect (hybrid unless noted) | Decision |
|---|---|---|---|---|
| 1 | nomic `search_query:` prefix on queries | rag_flask_mcp.py | MRR@8 0.643→0.753, hit@1 0.467→0.633, k20 hit@20 0.900→0.967, hit@8 flat | **Shipped** (commit "Fix 1") |
| 2 | nomic `search_document:` prefix on chunks | ingest | hit@8 0.900→0.967, k20 hit@20 0.967→1.000, vec MRR 0.736→0.746; hybrid MRR 0.736→0.718 | **Shipped** (commit "Fix 2"). Needs full re-embed |
| 3 | FTS is AND-only `plainto_tsquery` + un-normalised `ts_rank` | hybrid_search SQL | Before Fix 2: hit@8 +0.033, keyword hit@8 0.889→1.000. After Fix 2: hit@8 flat, MRR +0.010, hit@20 −0.033, with 15/30 queries changing rank in both directions | **Not shipped.** No net gain once Fix 2 is in. It would also need a stored tsvector column, because the OR query on the expression index takes 1.2–2.5 s vs 0.02–0.18 s |
| 4 | No title in chunk text (MD + PDF) | ingest | On top of Fix 2: MRR@8 0.718→0.762, hit@1 +0.067, but hit@8 0.967→0.933 (q20 drops to rank >8, still in top 20) | **Not shipped.** Recall@8 goes down. Revisit with a bigger set; cost is the same re-embed as Fix 2 |
| 5 | PDF chunks 2000 words | config | hybrid hit@8 +0.033, vector hit@8 −0.033, zero-FTS 13→19 | Not shipped (mixed result) |
| 6 | pdftotext `-layout` column noise | ingest | hit@8 flat; vector MRR +0.034 | Not shipped (no recall gain) |
| 7 | `#` lines inside code fences parsed as headings | ingest | no change on any metric | Not shipped |

Skipped by design (out of scope or exotic per the quality bar): Rerank 5, contextual
retrieval, dim reduction, weighted fusion, pg_search/BM25 extensions.

## Final (Fix 1 + Fix 2) vs baseline
Live baseline → rebuilt corpus with both fixes:
hybrid hit@8 0.900 → 0.967 · MRR@8 0.643 → 0.718 · hit@1 0.467 → 0.567 ·
vector hit@8 0.867 → 0.967 · k20 hit@20 0.900 → 1.000.
(The rebuilt corpus has 19,681 chunks vs 19,833 live, because 15 tracked files no longer exist on disk.
Rebuilt baseline with Fix 1 only: hit@8 0.900, MRR 0.736.)

## Deploy (not done; live index and server untouched)
1. Restart the RAG Flask service. Fix 1 takes effect right away and is safe on the current index.
2. Re-embed everything (Fix 2): `python ingest_rag_pgvector_v3.py --clear-folder <vault-folder>`,
   then `--clear-folder <pdf-folder>`, then a normal run. Until this runs, any
   incremental ingest writes un-prefixed vectors into a prefixed index.
3. Re-run `python eval/run_eval.py --qprefix qwen --name live_after` and check hit@8 ≥ 0.967.

Caveat: n=30, so one query is worth 0.033. Only the Fix 1 and Fix 2 deltas are clearly above noise.

## Native FTS (stored tsvector + GIN) + OR hybrid in SQL — deployed 2026-10-30

- `migrations/002_native_fts.sql`: `chunks.tsv` GENERATED ALWAYS AS
  (to_tsvector('english', chunk_text)) STORED + `idx_chunks_tsv` GIN;
  dropped the redundant expression index `idx_chunks_fts`.
  `hybrid_search()` rewritten on the stored column: OR of the plainto
  lexemes + `ts_rank(...,1)` length-norm, RRF k=60 unchanged, same
  signature (no Flask change needed, takes effect immediately).
- `eval/run_eval.py`: live-table paths now read the stored `c.tsv`;
  `live_fn` (deployed function) scored for every `--fts` mode.
- Live `hybrid_search()` on `eval/results/live_fts_native.json` (qwen prefix):
  hit@1 0.667 (flat) · hit@8 0.967 (flat) · MRR@8 0.769 → **0.779** ·
  zero-FTS queries 18 → **0**. Only miss is q09 (vec rank 17, hard for both).
  Matches the pre-registered Fix-3 prediction (+0.010 MRR, recall flat).
- FTS leg latency, same 3 probes: expr-AND 0.04/0.00/0.01s with n=0/0/4 rows
  → stored-OR 0.42/0.02/0.15s with n=24/24/24 rows. Slower worst case
  (bigger OR posting lists) but still far under the embedding call, and
  every query now gets a real FTS vote instead of 60% getting none.

## Combined pass (Fix 2 + PDF 500w rechunk + Mistral-Small PDF summaries) — deployed 2026-10-29

- `config.py`: `PDF_CHUNK_MAX` 2000→500, overlap 150→75. New `chunk_pdf()` wraps
  `chunk_pdf_pages` and prepends `[stem: 1-2 sentence summary]` (Mistral
  `mistralai/mistral-small-3.2-24b-instruct` via OpenRouter, cached in
  `pdf_doc_summaries`, fail-open) to every PDF chunk. MD path untouched.
- Shadow `rag_eval.chunks_combo` (24,153 chunks) then full live re-embed
  (24,218 chunks, 0 failures, 15 stale rows purged): 1984 MD / 12,362 chunks,
  65 PDF / 11,916 chunks.
- Live `hybrid hit@8 0.967 · MRR@8 0.769 · hit@1 0.667 · k20 hit@20 1.000 ·
  keyword hit@8 1.000` (`live_after_combo.json`), matching the shadow run.
  Beats Fix-1+Fix-2 alone (MRR 0.718, hit@1 0.567) on every hybrid metric.
