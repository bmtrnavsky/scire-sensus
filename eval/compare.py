"""Print a comparison table of result files."""
import json, sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
names = sys.argv[1:] or sorted(p.stem for p in R.glob("*.json"))
print("{:24} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>6} {:>6} | {:>4}".format(
    "run", "h@1", "h@8", "mrr8", "vh@1", "vh@8", "vmrr", "h20@20", "kw@8", "0fts"))
for n in names:
    s = json.loads((R / (n + ".json")).read_text())["summary"]
    h, v, h20 = s["hybrid_k8"], s["vec"], s["hybrid_k20"]
    print("{:24} {:7.3f} {:7.3f} {:7.3f} | {:7.3f} {:7.3f} {:7.3f} | {:6.3f} {:6.3f} | {:4}".format(
        n, h["hit@1"], h["hit@8"], h["mrr@8"], v["hit@1"], v["hit@8"], v["mrr@50"],
        h20["hit@20"], s["hybrid_k8_hit@8_keyword"], s["queries_with_zero_fts_matches"]))
