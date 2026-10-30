"""Run the 30-query golden set against an index variant.

  python run_eval.py --table public.chunks --qprefix nomic --fts plain --name baseline

A query counts as a HIT at rank r when the chunk at rank r comes from one of
the query's gold files AND contains one of its answer phrases (case and
whitespace normalised). Answer phrases survive re-chunking, so the same
judgments work across every chunking variant.

Pipelines scored (timebound_context excluded, as in rag_search):
  vec     exact cosine top-50
  hybrid  replica of public.hybrid_search: vec top (k*3) + FTS top (k*3),
          RRF k=60, top k. Scored at k=8 (what rag_search fetches per query)
          and k=20 (a rerank-sized pool).
  live_fn the deployed public.hybrid_search function itself (only for
          --table public.chunks), HNSW + whatever ef_search the session has.
"""
import argparse
import re
import statistics

from common import *
from validate_queries import load_queries, norm

FTS_QUERY = {
    # deployed: AND of every term
    "plain": "plainto_tsquery('english', %(q)s)",
    # OR of the same stemmed lexemes (stopwords already dropped by plainto)
    "or": "NULLIF(replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | '), '')::tsquery",
}


def view_for(table):
    """Normalise a variant table to (id, file_path, chunk_text, domain, embedding, tsv)."""
    if table == "public.chunks":
        # stored GENERATED column (migrations/002_native_fts.sql), same data
        # the deployed hybrid_search() reads -- no per-row recompute
        return ("(SELECT c.id, sf.file_path, c.chunk_text, c.domain, c.embedding, "
                "c.tsv FROM chunks c "
                "JOIN source_files sf ON sf.id = c.source_id)")
    return table


def vec_sql(src):
    return ("SELECT id, file_path, chunk_text FROM {src} t "
            "WHERE t.domain IS DISTINCT FROM 'timebound_context' "
            "ORDER BY t.embedding <=> %(v)s::halfvec LIMIT %(n)s").format(src=src)


def hybrid_sql(src, fts, ranker):
    tsq = FTS_QUERY[fts]
    rank = {"ts_rank": "ts_rank(t.tsv, q.tq)", "ts_rank_cd": "ts_rank_cd(t.tsv, q.tq, 1|32)",
            "ts_rank_n1": "ts_rank(t.tsv, q.tq, 1)"}[ranker]
    if src == "public.chunks":
        # stored GENERATED column + GIN (migrations/002_native_fts.sql);
        # t. here is the view_for subquery which already exposes tsv
        fts_cte = """SELECT t.id, ROW_NUMBER() OVER (ORDER BY {rank} DESC) AS rnk
                     FROM {src} t, q WHERE t.tsv @@ q.tq AND t.domain IS DISTINCT FROM 'timebound_context'
                     ORDER BY {rank} DESC LIMIT %(pool)s""".format(rank=rank, src=src)
    else:
        fts_cte = """SELECT t.id, ROW_NUMBER() OVER (ORDER BY {rank} DESC) AS rnk
                     FROM {src} t, q WHERE t.tsv @@ q.tq AND t.domain IS DISTINCT FROM 'timebound_context'
                     ORDER BY {rank} DESC LIMIT %(pool)s""".format(rank=rank, src=src)
    return """
    WITH q AS (SELECT {tsq} AS tq),
    vec AS (SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.embedding <=> %(v)s::halfvec) AS rnk
            FROM {src} t WHERE t.domain IS DISTINCT FROM 'timebound_context'
            ORDER BY t.embedding <=> %(v)s::halfvec LIMIT %(pool)s),
    fts AS ({fts_cte}),
    rrf AS (SELECT COALESCE(vec.id, fts.id) AS id,
                   COALESCE(1.0/(60+vec.rnk),0) + COALESCE(1.0/(60+fts.rnk),0) AS s
            FROM vec FULL OUTER JOIN fts ON vec.id = fts.id)
    SELECT t.id, t.file_path, t.chunk_text, (SELECT count(*) FROM fts) AS nfts
    FROM rrf JOIN {src} t ON t.id = rrf.id ORDER BY rrf.s DESC, t.id LIMIT %(k)s
    """.format(tsq=tsq, src=src, fts_cte=fts_cte)


def first_hit(rows, q):
    golds = {g.lower() for g in q["gold_files"]}
    ans = [norm(a) for a in q["answer"]]
    hit = fhit = None
    for i, r in enumerate(rows, 1):
        if Path(r[1]).name.lower() in golds:
            fhit = fhit or i
            if hit is None and any(a in norm(r[2]) for a in ans):
                hit = i
    return hit, fhit


def summarise(ranks, ks):
    n = len(ranks)
    out = {"hit@{}".format(k): round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in ks}
    mk = max(ks)
    out["mrr@{}".format(mk)] = round(sum(1.0 / r for r in ranks if r and r <= mk) / n, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="public.chunks")
    ap.add_argument("--qprefix", default="nomic", choices=list(QUERY_PREFIXES))
    ap.add_argument("--fts", default="plain", choices=list(FTS_QUERY))
    ap.add_argument("--ranker", default="ts_rank", choices=["ts_rank", "ts_rank_cd", "ts_rank_n1"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    qs = load_queries()
    prefix = QUERY_PREFIXES[a.qprefix]
    embed_cached([prefix + q["query"] for q in qs])
    src = view_for(a.table)
    c = conn(); cur = c.cursor()
    cur.execute("SET enable_indexscan = off")   # exact vector search everywhere (no HNSW recall loss)
    cur.execute("SET jit = off")
    per_q, res = [], {"vec": [], "hyb8": [], "hyb20": [], "vec_file": [], "hyb8_file": [], "live_fn": []}
    zero_fts = 0
    for q in qs:
        v = embed_query(prefix + q["query"])
        cur.execute(vec_sql(src), {"v": v, "n": 50})
        vh, vf = first_hit(cur.fetchall(), q)
        cur.execute(hybrid_sql(src, a.fts, a.ranker), {"v": v, "q": q["query"], "pool": 24, "k": 8})
        rows8 = cur.fetchall()
        h8, f8 = first_hit(rows8, q)
        nfts = rows8[0][3] if rows8 else 0
        zero_fts += 1 if nfts == 0 else 0
        cur.execute(hybrid_sql(src, a.fts, a.ranker), {"v": v, "q": q["query"], "pool": 60, "k": 20})
        h20, _ = first_hit(cur.fetchall(), q)
        lf = None
        if a.table == "public.chunks":
            # deployed public.hybrid_search itself (HNSW + session ef_search),
            # independent of the --fts replica flag above
            cur.execute("RESET enable_indexscan")
            cur.execute("SELECT chunk_id, source_path, chunk_text FROM hybrid_search(%s, %s::halfvec, 8, NULL, 'timebound_context')",
                        (q["query"], v))
            lf, _ = first_hit(cur.fetchall(), q)
            cur.execute("SET enable_indexscan = off")
        res["vec"].append(vh); res["vec_file"].append(vf); res["hyb8"].append(h8)
        res["hyb8_file"].append(f8); res["hyb20"].append(h20); res["live_fn"].append(lf)
        per_q.append({"id": q["id"], "type": q["type"], "vec": vh, "hyb8": h8, "hyb20": h20, "nfts": nfts, "live_fn": lf})
        if a.verbose:
            print("  {} vec={} hyb8={} hyb20={} nfts={}".format(q["id"], vh, h8, h20, nfts))

    summary = {
        "name": a.name, "table": a.table, "qprefix": a.qprefix, "fts": a.fts, "ranker": a.ranker,
        "n_queries": len(qs),
        "vec": summarise(res["vec"], [1, 8, 20, 50]),
        "vec_file_hit@8": round(sum(1 for r in res["vec_file"] if r and r <= 8) / len(qs), 3),
        "hybrid_k8": summarise(res["hyb8"], [1, 8]),
        "hybrid_k8_file_hit@8": round(sum(1 for r in res["hyb8_file"] if r and r <= 8) / len(qs), 3),
        "hybrid_k20": summarise(res["hyb20"], [8, 20]),
        "queries_with_zero_fts_matches": zero_fts,
    }
    if any(x is not None for x in res["live_fn"]) and a.table == "public.chunks":
        summary["live_fn_hit@8"] = round(sum(1 for r in res["live_fn"] if r) / len(qs), 3)
    for t in ("keyword", "paraphrase"):
        idx = [i for i, q in enumerate(qs) if q["type"] == t]
        summary["hybrid_k8_hit@8_" + t] = round(sum(1 for i in idx if res["hyb8"][i]) / len(idx), 3)
    out = EVAL_DIR / "results"
    out.mkdir(exist_ok=True)
    (out / (a.name + ".json")).write_text(json.dumps({"summary": summary, "per_query": per_q}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
