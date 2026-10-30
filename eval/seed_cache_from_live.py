"""Seed rag_eval.embed_cache from the live index so rebuilding the current
ingest output costs no API calls: key = ("search_document: " + chunk_text)."""
from common import *

init_schema()
with conn() as c, c.cursor() as cur:
    cur.execute("SELECT id, chunk_text FROM chunks")
    rows = cur.fetchall()
    recs = [(cache_key("search_document: " + t), i) for i, t in rows]
    cur.execute("CREATE TEMP TABLE seed (key text, id bigint)")
    psycopg2.extras.execute_values(cur, "INSERT INTO seed VALUES %s", recs, page_size=5000)
    cur.execute("""INSERT INTO rag_eval.embed_cache (key, embedding)
                   SELECT DISTINCT ON (s.key) s.key, c.embedding FROM seed s JOIN chunks c ON c.id = s.id
                   ON CONFLICT DO NOTHING""")
    print("seeded", cur.rowcount)
