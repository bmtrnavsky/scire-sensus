"""Build a shadow index (rag_eval.chunks_<tag>) by running the CURRENT
ingest_rag_pgvector_v3.py chunkers over every file the live index tracks.

  python build_variant.py --tag base
  python build_variant.py --tag p_noprefix --patch noprefix   # audit prototype

--patch applies an audit prototype of a candidate fix on top of the ingest
module (monkeypatch). Once a fix is committed to the ingest script the
prototype is no longer needed; rebuilding without it must reproduce the
same chunks (embeddings come from rag_eval.embed_cache, so it's cheap).
"""
import argparse
import importlib
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from common import *

ing = importlib.import_module("ingest_rag_pgvector_v3")
ing.logger.setLevel("WARNING")


# ---------------------------------------------------------------- prototypes
def _p_noprefix():
    ing.embed_input = lambda text, **kw: text


def _p_pdfsize():
    ing.PDF_CHUNK_MAX, ing.PDF_CHUNK_OVERLAP = 450, 60


def _p_nolayout():
    def extract_pdf_pdftotext(path):
        r = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True,
                           timeout=180, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not r.stdout.strip():
            return None
        pages = [{"text": t.strip(), "page_number": i + 1} for i, t in enumerate(r.stdout.split("\f")) if t.strip()]
        return pages or None
    ing.extract_pdf_pdftotext = extract_pdf_pdftotext
    ing.PDF_EXTRACT_MODE = "raw"


def _p_title():
    orig_md = ing.chunk_markdown

    def chunk_markdown(path):
        res = orig_md(path)
        if not res:
            return res
        raw = path.read_text(encoding="utf-8", errors="ignore")
        _, fm = ing.parse_frontmatter(raw)
        title = (fm.get("title") or "").strip() or path.stem
        return [("[{}]\n{}".format(title, t), h, p) for t, h, p in res]
    ing.chunk_markdown = chunk_markdown
    ing._PROTO_PDF_TITLE = True


def _p_codefence():
    # don't treat '#' lines inside ``` fences as headings
    orig_split = ing._HEADING_SPLIT

    class FenceAware:
        def split(self, body):
            fence = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
            masked = fence.sub(lambda m: m.group(0).replace("#", "\uE000"), body)
            return [p.replace("\uE000", "#") if isinstance(p, str) else p for p in orig_split.split(masked)]
    ing._HEADING_SPLIT = FenceAware()


PATCHES = {"noprefix": _p_noprefix, "pdfsize": _p_pdfsize, "nolayout": _p_nolayout,
           "title": _p_title, "codefence": _p_codefence}


# ---------------------------------------------------------------- pipeline
def embed_input_for(text):
    fn = getattr(ing, "embed_input", None)
    return fn(text) if fn else "search_document: " + text


def pdf_pages(path, fhash):
    mode = getattr(ing, "PDF_EXTRACT_MODE", "layout")
    key = "{}|{}".format(fhash, mode)
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT pages FROM rag_eval.pdf_cache WHERE file_path=%s AND file_hash=%s", (str(path), key))
        row = cur.fetchone()
        if row:
            return row[0]
    pages = None
    if mode != "layout" and ing.is_image_based(path):
        # OCR output doesn't depend on the pdftotext mode; reuse it
        with conn() as c, c.cursor() as cur:
            cur.execute("SELECT pages FROM rag_eval.pdf_cache WHERE file_path=%s AND file_hash=%s",
                        (str(path), "{}|layout".format(fhash)))
            row = cur.fetchone()
            pages = row[0] if row else None
    if pages is None:
        pages = ing.extract_pdf(path)
    with conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO rag_eval.pdf_cache VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (str(path), key, json.dumps(pages)))
    return pages


def chunks_for(row):
    fp, doc_type, domain, fhash = row
    path = Path(fp)
    if doc_type == "markdown":
        res = ing.chunk_markdown(path) or []
    else:
        pages = pdf_pages(path, fhash)
        if not pages:
            return fp, domain, []
        if hasattr(ing, "chunk_pdf"):
            res = ing.chunk_pdf(path, pages)
        else:
            res = ing.chunk_pdf_pages(pages, ing.PDF_CHUNK_MAX, ing.PDF_CHUNK_OVERLAP)
            if getattr(ing, "_PROTO_PDF_TITLE", False):
                res = [("[{}]\n{}".format(path.stem, t), h, p) for t, h, p in res]
    return fp, domain, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--patch", action="append", default=[])
    a = ap.parse_args()
    for p in a.patch:
        PATCHES[p]()
    init_schema()
    with conn() as c, c.cursor() as cur:
        cur.execute("""SELECT file_path, doc_type, domain, file_hash FROM source_files
                       WHERE is_excluded = false AND excluded_at IS NULL ORDER BY file_path""")
        rows = [r for r in cur.fetchall() if os.path.exists(r[0])]
    print("files:", len(rows), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        built = list(ex.map(chunks_for, rows))
    print("chunked in {:.0f}s".format(time.time() - t0), flush=True)
    recs = []
    for fp, domain, res in built:
        for t, h, p in res:
            recs.append((fp, t, h, domain, embed_input_for(t)))
    print("chunks:", len(recs), flush=True)
    keys = embed_cached([r[4] for r in recs])
    tbl = "rag_eval.chunks_" + a.tag
    with conn() as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS " + tbl)
        cur.execute("""CREATE TABLE {} (id serial PRIMARY KEY, file_path text, chunk_text text,
                       heading_path text, domain text, ekey text, embedding halfvec(4000), tsv tsvector)""".format(tbl))
        psycopg2.extras.execute_values(
            cur, "INSERT INTO {} (file_path, chunk_text, heading_path, domain, ekey) VALUES %s".format(tbl),
            [(fp, t, h, d, keys[e]) for fp, t, h, d, e in recs], page_size=2000)
        cur.execute("UPDATE {} t SET embedding = e.embedding FROM rag_eval.embed_cache e WHERE e.key = t.ekey".format(tbl))
        cur.execute("UPDATE {} SET tsv = to_tsvector('english', chunk_text)".format(tbl))
        cur.execute("CREATE INDEX ON {} USING gin (tsv)".format(tbl))
        cur.execute("SELECT count(*), count(embedding) FROM " + tbl)
        print("table {} rows={} embedded={}".format(tbl, *cur.fetchone()))
        cur.execute("ANALYZE " + tbl)


if __name__ == "__main__":
    main()
