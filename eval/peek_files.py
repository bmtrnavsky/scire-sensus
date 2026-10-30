"""Print a few random chunks from named source files (used once to author queries.jsonl)."""
import sys
from common import *

names = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
c = conn(); cur = c.cursor()
cur.execute("select setseed(0.7)")
for nm in names:
    cur.execute("""select c.chunk_index, c.chunk_text, sf.title from chunks c join source_files sf on sf.id=c.source_id
                   where sf.file_path ilike %s and c.token_count > 60 order by random() limit %s""", ("%\\" + nm, n))
    print("=" * 10, nm)
    for idx, t, title in cur.fetchall():
        print("  [{}] (title={}) {}".format(idx, title, " ".join(t.split())[:450]))
