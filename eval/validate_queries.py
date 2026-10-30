"""Validate queries.jsonl against the live index: every answer phrase must
appear (whitespace/case-normalised) in at least one chunk of a gold file."""
import re
from common import *


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def load_queries():
    return [json.loads(l) for l in open(EVAL_DIR / "queries.jsonl", encoding="utf-8") if l.strip()]


if __name__ == "__main__":
    qs = load_queries()
    c = conn(); cur = c.cursor()
    bad = 0
    for q in qs:
        golds = [g.lower() for g in q["gold_files"]]
        cur.execute("""select sf.file_path, c.chunk_text from chunks c join source_files sf on sf.id=c.source_id
                       where """ + " or ".join(["c.chunk_text ilike %s"] * len(q["answer"])),
                    ["%" + a + "%" for a in q["answer"]])
        rows = cur.fetchall()
        in_gold = [r for r in rows if Path(r[0]).name.lower() in golds]
        other = sorted({Path(r[0]).name for r in rows if Path(r[0]).name.lower() not in golds})
        # also check whitespace-normalised match inside gold file chunks
        cur.execute("""select c.chunk_text from chunks c join source_files sf on sf.id=c.source_id
                       where lower(sf.file_path) like any(%s)""", (["%\\" + g for g in golds],))
        gold_chunks = [r[0] for r in cur.fetchall()]
        nhit = sum(1 for t in gold_chunks if any(norm(a) in norm(t) for a in q["answer"]))
        flag = "OK " if nhit else "BAD"
        bad += 0 if nhit else 1
        print(flag, q["id"], "gold_chunks={} answer_chunks={} other_files={}".format(len(gold_chunks), nhit, other[:3]))
    print("bad:", bad)
