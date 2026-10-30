#!/usr/bin/env python3
"""
chroma_api_pgvector.py -- Flask MCP server for pgvector RAG
Replaces ChromaDB Flask API (same tool interface, port changed to 5556).
Binds 127.0.0.1:5556 (native Windows Service via NSSM: RagFlaskAPI, since Sept 2026 migration off WSL2).
Created: 2026-05-22
"""

import json
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests
from flask import Flask, request, jsonify

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DB_DSN, OPENROUTER_API_KEY, OPENROUTER_URL, EMBED_MODEL,
    FLASK_PORT, DEFAULT_TOP_K, MAX_TOP_K, QUERY_INSTRUCT,
    RERANK_MODEL, RERANK_POOL, RERANK_MIN_SCORE, RERANK_MIN_KEEP,
    RERANK_FALLBACK_LOG,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_db_conn():
    conn = psycopg2.connect(DB_DSN)
    psycopg2.extras.register_uuid()
    return conn


def get_query_embedding(query: str) -> list:
    """Embed query via OpenRouter Qwen3-Embedding-8B with the Qwen3 instruction prefix.

    Pinned to DeepInfra as of 2026-07-30 (re-confirmed). Original pin was
    Nebius (2026-07-10, ~1-2s vs DeepInfra's ~23s then). RAG search
    regressed to ~30s+ per call by 2026-07-30; re-testing found Nebius
    unreliable and DeepInfra fast in isolated single calls. A rate-limit
    hit during same-day testing was traced to the testing methodology
    itself (6+ back-to-back calls in under 4 minutes with no delay --
    a burst pattern real usage doesn't produce), not a genuine DeepInfra
    problem. Re-pinned to DeepInfra after confirming this. If search
    feels slow again, re-run a SINGLE clean provider.only timing test
    (not a rapid loop) before concluding a provider has degraded --
    back-to-back test bursts can trip rate limits that normal spaced-out
    usage never would.
    """
    # Qwen3-Embedding is instruction-aware on the query side only. The old
    # "search_query: " prefix was a nomic-embed leftover; measured on
    # eval/queries.jsonl it cost 0.11 MRR@8 (0.643 -> 0.753 with this).
    prefixed = QUERY_INSTRUCT + query
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json={
            "model": EMBED_MODEL,
            "input": [prefixed],
            "dimensions": 4000,
            "provider": {"only": ["DeepInfra"]},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]



def expand_query(query: str) -> list:
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        prompt = ("Generate 2 alternative search queries for the following query. Each should rephrase the same intent using different vocabulary or framing. Return ONLY a JSON array of 2 strings, no explanation, no markdown.\n\nQuery: " + query)
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={"model": "deepseek/deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0.7}, timeout=15)
        resp.raise_for_status()
        import json as _j
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"): raw = raw.split("\n",1)[1].rsplit("```",1)[0].strip()
        e = _j.loads(raw)
        if isinstance(e, list): return ([query]+[q for q in e if q and q!=query])[:3]
    except: pass
    return [query]


def rerank_results(query, rows, top_n=7):
    """Rerank using Cohere Rerank via OpenRouter.

    Model: RERANK_MODEL (Rerank 4 Pro -- Rerank 5 was NOT on OpenRouter as of
    2026-10-29; flip RERANK_MODEL to "cohere/rerank-5-pro" when it lands, no
    other change needed). Scores below RERANK_MIN_SCORE are dropped, except
    the top RERANK_MIN_KEEP rows are always kept so a cutoff can never empty
    the result set.
    Falls back to RRF order if the API call fails -- LOUDLY (stderr + log
    file), so a dead reranker can't silently degrade quality again.
    """
    import time as _t
    import traceback as _tb
    try:
        docs = [r.get("chunk_text", r.get("content", "")) for r in rows]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/rerank",
            headers=headers,
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": docs,
                "top_n": top_n,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        reranked = []
        for r in results:
            row = dict(rows[r["index"]])
            row["rerank_score"] = float(r["relevance_score"])
            reranked.append(row)
        if RERANK_MIN_SCORE > 0:
            kept = [r for r in reranked if r["rerank_score"] >= RERANK_MIN_SCORE]
            if len(kept) < RERANK_MIN_KEEP:
                kept = reranked[:RERANK_MIN_KEEP]
            dropped = len(reranked) - len(kept)
            if dropped:
                print(f"[rerank] cutoff {RERANK_MIN_SCORE}: dropped {dropped}/{len(reranked)} "
                      f"below-threshold docs for query {query[:60]!r}", flush=True)
            reranked = kept
        return reranked
    except Exception as e:
        # LOUD fallback to RRF order -- pipeline never breaks, but a dead
        # reranker must be visible, not silent (pre-2026-10-29 bug).
        msg = (f"[rerank] FALLBACK model={RERANK_MODEL} pool={len(rows)} "
               f"query={query[:80]!r} err={e}")
        print(msg, flush=True)
        print(_tb.format_exc(limit=3), flush=True)
        try:
            with open(RERANK_FALLBACK_LOG, "a", encoding="utf-8") as f:
                f.write(f"{_t.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
        except OSError:
            pass
        return rows[:top_n]

def format_results(rows: list) -> str:
    if not rows:
        return "No results found."
    parts = []
    for i, row in enumerate(rows, 1):
        chunk_text   = row["chunk_text"]
        # hybrid_search() (rag_search) returns source_path and no doc_type;
        # the direct SELECTs return file_path + doc_type. Accept either, then
        # fall back to the per-chunk metadata written at ingest.
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {}
        file_path    = (row.get("file_path") or row.get("source_path")
                        or meta.get("full_path") or "")
        doc_type     = (row.get("doc_type") or meta.get("file_type")
                        or Path(file_path).suffix.lstrip(".").lower())
        if doc_type == "md":
            doc_type = "markdown"
        rrf_score    = row.get("rrf_score", 0.0)
        heading_path = row.get("heading_path") or ""
        page_number  = row.get("page_number")
        domain       = row.get("domain") or ""
        content_date = row.get("content_date")
        filename     = Path(file_path).name if file_path else "unknown"

        if heading_path:
            header = f"[{i}] {filename} | {doc_type} | § {heading_path} | score={rrf_score:.4f}"
        elif page_number:
            header = f"[{i}] {filename} | {doc_type} | p.{page_number} | score={rrf_score:.4f}"
        else:
            header = f"[{i}] {filename} | {doc_type} | score={rrf_score:.4f}"
        if domain:
            header += f" | domain={domain}"
        if content_date:
            header += f" | date={content_date}"

        parts.append(f"{header}\nPath: {file_path}\n\n{chunk_text}\n\n---")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCP tool handler
# ---------------------------------------------------------------------------

def handle_tool_call(tool_name: str, tool_input: dict) -> dict:
    """
    Dispatch tool calls. Returns MCP tool_result content block.
    """
    query = tool_input.get("query", "").strip()
    top_k = min(int(tool_input.get("top_k", DEFAULT_TOP_K)), MAX_TOP_K)

    if not query:
        return {"type": "text", "text": "Error: query parameter is required."}

    try:
        query_vec = get_query_embedding(query)
    except Exception as e:
        return {"type": "text", "text": f"Embedding error: {e}"}

    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            if tool_name == "rag_search":
                queries = expand_query(query)
                seen_ids = {}
                rrf_k = 60
                for q_text in queries:
                    try:
                        q_vec = get_query_embedding(q_text) if q_text != query else query_vec
                    except: q_vec = query_vec
                    cur.execute(
                        "SELECT * FROM hybrid_search(%s, %s::halfvec, %s, NULL, %s)",
                        (q_text, json.dumps(q_vec), top_k, "timebound_context")
                    )
                    for pos, row in enumerate(cur.fetchall()):
                        row = dict(row); cid = row["chunk_id"]; contrib = 1.0/(rrf_k+pos+1)
                        if cid in seen_ids: seen_ids[cid]["rrf_score"] += contrib
                        else: row["rrf_score"] = contrib; seen_ids[cid] = row
                rows = sorted(seen_ids.values(), key=lambda x: x["rrf_score"], reverse=True)[:RERANK_POOL]
                rows = rerank_results(query, list(rows), top_n=top_k)

            elif tool_name == "rag_semantic":
                cur.execute(
                    """SELECT
                           c.id AS chunk_id,
                           c.chunk_text,
                           c.heading_path,
                           c.page_number,
                           c.domain,
                           c.metadata,
                           1 - (c.embedding <=> %s::halfvec) AS rrf_score,
                           sf.file_path,
                           sf.doc_type
                       FROM chunks c
                       JOIN source_files sf ON sf.id = c.source_id
                       WHERE c.domain IS DISTINCT FROM 'timebound_context'
                       ORDER BY c.embedding <=> %s::halfvec
                       LIMIT %s""",
                    (json.dumps(query_vec), json.dumps(query_vec), top_k)
                )
                rows = cur.fetchall()

            elif tool_name == "rag_search_time_bound":
                sort_mode = (tool_input.get("sort") or "chronological").strip().lower()
                if sort_mode not in ("chronological", "relevance"):
                    sort_mode = "chronological"

                if sort_mode == "relevance":
                    cur.execute(
                        """SELECT
                               c.id AS chunk_id,
                               c.chunk_text,
                               c.heading_path,
                               c.page_number,
                               c.domain,
                               c.metadata,
                               1 - (c.embedding <=> %s::halfvec) AS rrf_score,
                               sf.file_path,
                               sf.doc_type,
                               sf.content_date
                           FROM chunks c
                           JOIN source_files sf ON sf.id = c.source_id
                           WHERE c.domain = 'timebound_context'
                           ORDER BY c.embedding <=> %s::halfvec
                           LIMIT %s""",
                        (json.dumps(query_vec), json.dumps(query_vec), top_k)
                    )
                else:
                    # Chronological: still restrict to semantically relevant
                    # chunks (top_k * 3 candidate pool by cosine distance),
                    # then order that pool by content_date newest-first --
                    # otherwise "chronological" would just return the most
                    # recent content in the vault regardless of relevance
                    # to the query, which isn't what a search tool should do.
                    cur.execute(
                        """SELECT * FROM (
                               SELECT
                                   c.id AS chunk_id,
                                   c.chunk_text,
                                   c.heading_path,
                                   c.page_number,
                                   c.domain,
                                   c.metadata,
                                   1 - (c.embedding <=> %s::halfvec) AS rrf_score,
                                   sf.file_path,
                                   sf.doc_type,
                                   sf.content_date
                               FROM chunks c
                               JOIN source_files sf ON sf.id = c.source_id
                               WHERE c.domain = 'timebound_context'
                               ORDER BY c.embedding <=> %s::halfvec
                               LIMIT %s
                           ) candidates
                           ORDER BY content_date DESC NULLS LAST""",
                        (json.dumps(query_vec), json.dumps(query_vec), top_k * 3)
                    )
                rows = cur.fetchall()[:top_k]

            elif tool_name == "research_rag_semantic":
                project_id = tool_input.get("project_id")
                if project_id:
                    cur.execute(
                        """SELECT
                               c.id AS chunk_id, c.chunk_text, c.heading_path,
                               c.section_name, c.page_number, c.doi, c.year,
                               c.source_type, c.metadata,
                               1 - (c.embedding <=> %s::halfvec) AS rrf_score,
                               sf.file_path, sf.title AS doc_type
                           FROM research_rag_chunks c
                           JOIN research_source_files sf ON sf.id = c.source_id
                           WHERE c.project_id = %s
                           ORDER BY c.embedding <=> %s::halfvec
                           LIMIT %s""",
                        (json.dumps(query_vec), project_id, json.dumps(query_vec), top_k)
                    )
                else:
                    cur.execute(
                        """SELECT
                               c.id AS chunk_id, c.chunk_text, c.heading_path,
                               c.section_name, c.page_number, c.doi, c.year,
                               c.source_type, c.metadata,
                               1 - (c.embedding <=> %s::halfvec) AS rrf_score,
                               sf.file_path, sf.title AS doc_type
                           FROM research_rag_chunks c
                           JOIN research_source_files sf ON sf.id = c.source_id
                           ORDER BY c.embedding <=> %s::halfvec
                           LIMIT %s""",
                        (json.dumps(query_vec), json.dumps(query_vec), top_k)
                    )
                rows = cur.fetchall()

            elif tool_name == "research_rag_search":
                project_id = tool_input.get("project_id")
                proj_clause = "AND c.project_id = %s" if project_id else ""
                base_params = [project_id] if project_id else []
                queries = expand_query(query)
                seen_ids = {}
                rrf_k = 60
                for q_text in queries:
                    try:
                        q_vec = get_query_embedding(q_text) if q_text != query else query_vec
                    except: q_vec = query_vec
                    cur.execute(
                        f"""SELECT
                               c.id AS chunk_id, c.chunk_text, c.heading_path,
                               c.section_name, c.page_number, c.doi, c.year,
                               c.source_type, c.metadata,
                               sf.file_path, sf.title AS doc_type,
                               1 - (c.embedding <=> %s::halfvec) AS vec_score
                           FROM research_rag_chunks c
                           JOIN research_source_files sf ON sf.id = c.source_id
                           WHERE 1=1 {proj_clause}
                           ORDER BY vec_score DESC
                           LIMIT %s""",
                        [json.dumps(q_vec)] + base_params + [top_k * 2]
                    )
                    for pos, row in enumerate(cur.fetchall()):
                        row = dict(row); cid = row["chunk_id"]; contrib = 1.0/(rrf_k+pos+1)
                        if cid in seen_ids: seen_ids[cid]["rrf_score"] += contrib
                        else: row["rrf_score"] = contrib; seen_ids[cid] = row
                rows = sorted(seen_ids.values(), key=lambda x: x["rrf_score"], reverse=True)[:RERANK_POOL]
                rows = rerank_results(query, list(rows), top_n=top_k)

            else:
                return {"type": "text", "text": f"Unknown tool: {tool_name}"}

        result_text = format_results([dict(r) for r in rows])
        return {"type": "text", "text": result_text}

    except Exception as e:
        return {"type": "text", "text": f"Database error: {e}"}
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# MCP HTTP endpoints
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "rag_search",
        "description": (
            "Hybrid RAG search (vector + full-text + RRF fusion). "
            "Use for factual queries, specific terms, named concepts, acronyms, "
            "or when exact wording matters. Returns ranked chunks from vault notes and PDFs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results to return (default {DEFAULT_TOP_K}, max {MAX_TOP_K})",
                    "default": DEFAULT_TOP_K,
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "rag_semantic",
        "description": (
            "Semantic vector-only RAG search. "
            "Use for conceptual, associative, or exploratory queries where meaning "
            "matters more than exact keywords."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The semantic search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results to return (default {DEFAULT_TOP_K}, max {MAX_TOP_K})",
                    "default": DEFAULT_TOP_K,
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "rag_search_time_bound",
        "description": (
            "Search Daily Notes and Archives -- the timebound_context domain, "
            "excluded from rag_search/rag_semantic by default. Use for 'what "
            "happened with X', 'tell me about my prayer life', or any query "
            "about a specific past period rather than current state. Defaults "
            "to chronological order (newest first) among semantically relevant "
            "results, since a timeline usually tells the story better than a "
            "pure relevance ranking -- pass sort='relevance' to rank by best "
            "match instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results to return (default {DEFAULT_TOP_K}, max {MAX_TOP_K})",
                    "default": DEFAULT_TOP_K,
                },
                "sort": {
                    "type": "string",
                    "enum": ["chronological", "relevance"],
                    "description": "chronological (default, newest first) or relevance (best match first)",
                    "default": "chronological",
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "research_rag_search",
        "description": (
            "Hybrid search on the ISOLATED research pipeline staging table "
            "(research_rag_chunks) -- staged and screened academic and "
            "grey-lit papers, NOT the main vault/PDF corpus. Use this for any "
            "query within the research pipeline. Optional project_id scopes to one research thread."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results (default {DEFAULT_TOP_K}, max {MAX_TOP_K})",
                    "default": DEFAULT_TOP_K,
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional research_projects UUID to scope results to one thread"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "research_rag_semantic",
        "description": (
            "Semantic vector-only search on the ISOLATED research pipeline "
            "staging table (research_rag_chunks). Same scope note as "
            "research_rag_search -- these are staged papers, not the "
            "main vault corpus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The semantic search query"},
                "top_k": {
                    "type": "integer",
                    "description": f"Number of results (default {DEFAULT_TOP_K}, max {MAX_TOP_K})",
                    "default": DEFAULT_TOP_K,
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional research_projects UUID to scope results to one thread"
                }
            },
            "required": ["query"]
        }
    }
]


@app.route("/mcp", methods=["GET", "POST"])
def mcp_endpoint():
    """Main MCP endpoint -- handles initialize, tools/list, tools/call."""
    if request.method == "GET":
        # Health check
        return jsonify({"status": "ok", "service": "rag-pgvector-mcp"})

    data = request.get_json(force=True, silent=True) or {}
    method = data.get("method", "")
    req_id = data.get("id")

    def ok(result):
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": result})

    def err(code, message):
        return jsonify({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message}
        })

    if method == "initialize":
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "rag-pgvector", "version": "1.0.0"}
        })

    elif method == "tools/list":
        return ok({"tools": MCP_TOOLS})

    elif method == "tools/call":
        params = data.get("params", {})
        tool_name  = params.get("name", "")
        tool_input = params.get("arguments", {})
        content    = handle_tool_call(tool_name, tool_input)
        return ok({"content": [content]})

    elif method == "notifications/initialized":
        return jsonify({}), 200

    else:
        return err(-32601, f"Method not found: {method}")


@app.route("/health", methods=["GET"])
def health():
    """Detailed health check including DB connectivity."""
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM source_files")
            file_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cur.fetchone()[0]
        conn.close()
        db_status = "ok"
    except Exception as e:
        file_count = chunk_count = 0
        db_status = str(e)

    return jsonify({
        "status": "ok",
        "db": db_status,
        "source_files": file_count,
        "chunks": chunk_count,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting RAG pgvector MCP server on 127.0.0.1:{FLASK_PORT}")
    print(f"Health check: http://localhost:{FLASK_PORT}/health")
    print(f"MCP endpoint: http://localhost:{FLASK_PORT}/mcp")
    app.run(host="127.0.0.1", port=FLASK_PORT, debug=False, threaded=True)

