"""Shared helpers for the RAG retrieval eval harness.

Everything lives in the `rag_eval` schema so the live `public.chunks`
index is never touched:
  rag_eval.embed_cache   (key -> halfvec) so unchanged chunks are never
                         re-embedded between variants
  rag_eval.pdf_cache     (file_path, file_hash -> extracted pages JSON) so
                         PDF extraction/OCR runs once
  rag_eval.chunks_<tag>  one shadow index per ingest variant
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
from config import DB_DSN, OPENROUTER_API_KEY, OPENROUTER_URL, EMBED_MODEL, QUERY_INSTRUCT  # noqa: E402

DIMS = 4000
EVAL_DIR = Path(__file__).resolve().parent

# Query-side prefixes. "nomic" is what rag_flask_mcp.py sent before the fix;
# "qwen" is the Qwen3-Embedding instruction format (config.QUERY_INSTRUCT).
QWEN_INSTRUCT = QUERY_INSTRUCT
QUERY_PREFIXES = {"nomic": "search_query: ", "qwen": QWEN_INSTRUCT, "none": ""}


def conn():
    c = psycopg2.connect(DB_DSN)
    c.autocommit = True
    return c


def init_schema():
    with conn() as c, c.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS rag_eval")
        cur.execute("""CREATE TABLE IF NOT EXISTS rag_eval.embed_cache (
                         key text PRIMARY KEY, embedding halfvec(4000) NOT NULL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS rag_eval.pdf_cache (
                         file_path text, file_hash text, pages jsonb,
                         PRIMARY KEY (file_path, file_hash))""")


def cache_key(text: str) -> str:
    """Key on the exact string sent to the API (prefix included)."""
    return hashlib.sha256("{}|{}|{}".format(EMBED_MODEL, DIMS, text).encode("utf-8")).hexdigest()


def _embed_api(texts, provider="Nebius", tries=6):
    for attempt in range(tries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": "Bearer " + OPENROUTER_API_KEY,
                         "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": texts, "dimensions": DIMS,
                      "provider": {"only": [provider]}},
                timeout=180)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda x: x["index"])
            if len(data) != len(texts):
                raise RuntimeError("got {} embeddings for {} inputs".format(len(data), len(texts)))
            return [d["embedding"] for d in data]
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print("  embed retry {} ({})".format(attempt + 1, str(e)[:120]), flush=True)
            time.sleep(2 * (attempt + 1))


def embed_cached(texts, batch=32, workers=6, provider="Nebius"):
    """Returns {text: cache_key}. Embeds only texts missing from the cache."""
    keys = {t: cache_key(t) for t in texts}
    uniq = list(dict.fromkeys(keys.values()))
    have = set()
    with conn() as c, c.cursor() as cur:
        for i in range(0, len(uniq), 5000):
            cur.execute("SELECT key FROM rag_eval.embed_cache WHERE key = ANY(%s)", (uniq[i:i + 5000],))
            have.update(r[0] for r in cur.fetchall())
    todo = [t for t in dict.fromkeys(texts) if keys[t] not in have]
    print("  embed: {} unique, {} cached, {} to embed".format(len(uniq), len(have), len(todo)), flush=True)
    if not todo:
        return keys
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    done = [0]

    def work(b):
        vecs = _embed_api(b, provider=provider)
        with conn() as c, c.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO rag_eval.embed_cache (key, embedding) VALUES %s ON CONFLICT DO NOTHING",
                [(keys[t], json.dumps(v)) for t, v in zip(b, vecs)],
                template="(%s, %s::halfvec)")
        done[0] += len(b)
        if done[0] % (batch * 20) < batch:
            print("    embedded {}/{}".format(done[0], len(todo)), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, batches))
    return keys


def embed_query(text: str, provider="Nebius"):
    k = cache_key(text)
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT embedding::text FROM rag_eval.embed_cache WHERE key=%s", (k,))
        row = cur.fetchone()
        if row:
            return row[0]
    embed_cached([text], provider=provider)
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT embedding::text FROM rag_eval.embed_cache WHERE key=%s", (k,))
        return cur.fetchone()[0]


def norm_path(p: str) -> str:
    return p.replace("\\", "/").lower()
