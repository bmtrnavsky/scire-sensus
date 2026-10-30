#!/usr/bin/env python3
"""
test_rag.py -- sanity check queries against pgvector RAG
Run after ingest to validate hybrid search returns sensible results.
"""
import json
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_DSN, OLLAMA_URL, EMBED_MODEL

def get_embedding(query: str) -> list:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": f"search_query: {query}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def hybrid_search(query: str, top_k: int = 5):
    vec = get_embedding(query)
    conn = psycopg2.connect(DB_DSN)
    psycopg2.extras.register_uuid()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM hybrid_search(%s, %s::vector, %s, 60)",
                    (query, json.dumps(vec), top_k))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def semantic_search(query: str, top_k: int = 5):
    vec = get_embedding(query)
    conn = psycopg2.connect(DB_DSN)
    psycopg2.extras.register_uuid()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT c.id AS chunk_id, c.source_id, c.chunk_text, c.chunk_index,
                   c.metadata, 1 - (c.embedding <=> %s::vector) AS rrf_score,
                   sf.file_path, sf.doc_type
            FROM chunks c
            JOIN source_files sf ON sf.id = c.source_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (json.dumps(vec), json.dumps(vec), top_k))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def print_results(rows, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not rows:
        print("  NO RESULTS")
        return
    for i, r in enumerate(rows, 1):
        fname = Path(r["file_path"]).name
        score = r.get("rrf_score", 0)
        snippet = r["chunk_text"][:120].replace("\n", " ")
        print(f"  [{i}] {fname} | score={score:.4f}")
        print(f"      {snippet}...")

TESTS = [
    ("Hermes agent routing chain configuration", "hybrid"),
    ("pgvector schema hybrid search RRF", "hybrid"),
    ("DMAIC humanizer voice tuning project", "hybrid"),
    ("nomic-embed-text chunking strategy", "semantic"),
    ("INF630 systems analysis discussion post", "semantic"),
]

for query, mode in TESTS:
    if mode == "hybrid":
        rows = hybrid_search(query)
        print_results(rows, f"HYBRID: {query}")
    else:
        rows = semantic_search(query)
        print_results(rows, f"SEMANTIC: {query}")

print("\nAll tests complete.")
