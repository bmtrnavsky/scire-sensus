"""Time the FTS leg on the live table: expression to_tsvector vs a stored tsvector (shadow table)."""
import time
from common import *

Q = ["Why do so many small business owners end up just doing the work instead of building the business?",
     "how long was the siege of Vicksburg", "add a border to a chart in Power BI Desktop"]
OR = "NULLIF(replace(plainto_tsquery('english', %s)::text, ' & ', ' | '), '')::tsquery"
c = conn(); cur = c.cursor()
for q in Q:
    for label, sql in [
        ("live expr AND", "SELECT c.id FROM chunks c WHERE to_tsvector('english', c.chunk_text) @@ plainto_tsquery('english', %s) "
                          "ORDER BY ts_rank(to_tsvector('english', c.chunk_text), plainto_tsquery('english', %s), 1) DESC LIMIT 24"),
        ("live expr OR ", "SELECT c.id FROM chunks c WHERE to_tsvector('english', c.chunk_text) @@ " + OR +
                          " ORDER BY ts_rank(to_tsvector('english', c.chunk_text), " + OR + ", 1) DESC LIMIT 24"),
        ("stored OR    ", "SELECT t.id FROM rag_eval.chunks_base t WHERE t.tsv @@ " + OR +
                          " ORDER BY ts_rank(t.tsv, " + OR + ", 1) DESC LIMIT 24"),
    ]:
        t0 = time.time(); cur.execute(sql, (q, q)); n = len(cur.fetchall())
        print("{} {:6.2f}s n={}  {}".format(label, time.time() - t0, n, q[:40]))
