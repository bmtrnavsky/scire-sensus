"""Before/after eval for the reranker upgrade (pool 20 -> 45, cutoff, model const).

Pool construction is identical for both arms (single-query hybrid, RRF k=60,
pool 60 -> top 45). The arms differ only in what the reranker sees:
  old: first 20 of the pool, no score cutoff, top_n=8
  new: all 45 of the pool, RERANK_MIN_SCORE cutoff + MIN_KEEP, top_n=8
Both arms call the same RERANK_MODEL, so the delta measures the pool+cutoff
change (the Rerank 4 -> 5 model swap lands separately, when OpenRouter has it).

  python rerank_eval.py --name rerank_pool45

Writes eval/results/<name>.json in the same shape as run_eval.py summaries.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common import EVAL_DIR, conn, embed_cached, embed_query
from run_eval import FTS_QUERY, first_hit, hybrid_sql, summarise, view_for
from validate_queries import load_queries, norm

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
from config import OPENROUTER_API_KEY, QUERY_INSTRUCT, RERANK_MODEL  # noqa: E402

RERANK_URL = "https://openrouter.ai/api/v1/rerank"
OLD_POOL, NEW_POOL, TOP_N = 20, 45, 8
MIN_SCORE, MIN_KEEP = 0.05, 3


def rerank(query, docs, top_n):
    r = requests.post(
        RERANK_URL,
        headers={"Authorization": "Bearer " + OPENROUTER_API_KEY,
                 "Content-Type": "application/json"},
        json={"model": RERANK_MODEL, "query": query,
              "documents": docs, "top_n": top_n},
        timeout=60,
    )
    r.raise_for_status()
    return [(h["index"], float(h["relevance_score"])) for h in r.json()["results"]]


def score_arm(args):
    qi, q, pool_rows = args
    docs = [r[2] for r in pool_rows]
    try:
        res = rerank(q["query"], docs, TOP_N)
        idx = [i for i, _ in res]
    except Exception as e:  # loud fallback, same as production
        print("  FALLBACK {}: {}".format(q["id"], str(e)[:120]), flush=True)
        idx = list(range(TOP_N))
        res = [(i, 1.0) for i in idx]
    old_rows = [pool_rows[i] for i in idx if i < OLD_POOL][:TOP_N]
    new_idx = [i for i, s in res if s >= MIN_SCORE][:TOP_N]
    if len(new_idx) < MIN_KEEP:
        new_idx = [i for i, _ in res][:MIN_KEEP]
    new_rows = [pool_rows[i] for i in new_idx]
    return qi, q, old_rows, new_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fts", default="or", choices=list(FTS_QUERY))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    qs = load_queries()
    embed_cached([QUERY_INSTRUCT + q["query"] for q in qs])
    src = view_for("public.chunks")
    c = conn()
    cur = c.cursor()
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET jit = off")
    pools = []
    for q in qs:
        v = embed_query(QUERY_INSTRUCT + q["query"])
        cur.execute(hybrid_sql(src, a.fts, "ts_rank_cd"),
                    {"v": v, "q": q["query"], "pool": 60, "k": NEW_POOL})
        pools.append(cur.fetchall())
    c.close()

    old_r, new_r, per_q = [], [], []
    with ThreadPoolExecutor(a.workers) as ex:
        for qi, q, orows, nrows in ex.map(score_arm, [(i, q, p) for i, (q, p) in enumerate(zip(qs, pools))]):
            oh, _ = first_hit([(None, r[1], r[2]) for r in orows], q)
            nh, _ = first_hit([(None, r[1], r[2]) for r in nrows], q)
            old_r.append(oh)
            new_r.append(nh)
            per_q.append({"id": q["id"], "old_rank": oh, "new_rank": nh,
                          "new_n": len(nrows)})
            print("  {} old={} new={} n_new={}".format(q["id"], oh, nh, len(nrows)), flush=True)
            time.sleep(0.25)

    summary = {
        "name": a.name, "model": RERANK_MODEL, "fts": a.fts,
        "old_pool": OLD_POOL, "new_pool": NEW_POOL, "top_n": TOP_N,
        "cutoff": MIN_SCORE, "min_keep": MIN_KEEP, "n_queries": len(qs),
        "old_pool20": summarise(old_r, [1, 8]),
        "new_pool45_cutoff": summarise(new_r, [1, 8]),
        "new_mean_returned": round(sum(p["new_n"] for p in per_q) / len(per_q), 2),
    }
    out = EVAL_DIR / "results" / (a.name + ".json")
    out.write_text(__import__("json").dumps({"summary": summary, "per_query": per_q}, indent=1))
    print(__import__("json").dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
