#!/usr/bin/env python3
"""
ingest_rag_pgvector_v3.py
=========================
Upgrade from v2.0:
  - Embedding: nomic-embed-text (Ollama 768d) -> qwen3-embedding-8b (OpenRouter 4096d)
  - Markdown: heading-based chunking (H1/H2/H3), YAML stripped, breadcrumb prefix
  - PDF: heading-pattern detection + 2000-word fallback, no YAML ingested
  - New source_files columns: title, domain, note_type, note_status, tags
  - New chunks columns: heading_path, page_number, domain
  - Batch size 5 (OpenRouter rate-limit safe)

Preserved from v2.0:
  - pdftotext primary, pymupdf fallback, Tesseract OCR for image PDFs
  - Per-file transaction commit
  - Self-healing chunk_count=0 repair
  - --remove-file / --clear-folder CLI
  - SKIP_FILES blocklist
  - Auto-retry on failure

Run schema migration before first use:
  psql -U postgres -p 5433 -d ragdb -f schema_v3.sql
"""

import hashlib, json, logging, re, subprocess, sys, yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2, psycopg2.extras, requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DB_DSN, SOURCE_ROOTS, EXCLUDE_MARKERS, EXCLUDE_PDF_MARKERS,
    TIMEBOUND_MARKERS,
    OPENROUTER_API_KEY, OPENROUTER_URL, EMBED_MODEL, EMBED_BATCH_SIZE,
    MD_CHUNK_MAX, MD_CHUNK_OVERLAP,
    PDF_CHUNK_MAX, PDF_CHUNK_OVERLAP,
    SUMMARY_MODEL, SUMMARY_MAX_CHARS, SUMMARY_MAX_TOKENS,
)

IMAGE_THRESHOLD = 100
SKIP_FILES: set = {
    "default-misspellings.md", "ai-vocabulary.md",
    "wiki-batch-run-log.md", "AGENTS_backup_20260531.md",
}
RETRY_SCHEDULE = [(500, 75), (100, 25)]

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#  UTILITIES
# ------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def is_excluded(file_path: str, doc_type: str) -> bool:
    """Match exclude markers against the path with forward slashes, since
    EXCLUDE_MARKERS use Unix-style '/folder/' separators but Windows paths
    use backslashes. A plain substring check silently never matched on
    Windows -- confirmed 2026-09-05 (Archives, 99 System, and Daily Notes
    were all leaking into RAG despite being listed as excluded)."""
    normalized = file_path.replace("\\", "/")
    markers = EXCLUDE_MARKERS if doc_type == "markdown" else EXCLUDE_PDF_MARKERS
    return any(m in normalized for m in markers)

def is_timebound(file_path: str) -> bool:
    """Daily Notes + Archives: ingested normally but tagged
    domain='timebound_context' instead of excluded. See TIMEBOUND_MARKERS
    in config.py."""
    normalized = file_path.replace("\\", "/")
    return any(m in normalized for m in TIMEBOUND_MARKERS)

# ------------------------------------------------------------------
#  CONTENT DATE RESOLUTION (for timebound_context sort order)
#  3-tier fallback: frontmatter date_created -> filename-embedded date
#  -> filesystem creation time. Confirmed 2026-09-05 there is no
#  separate Obsidian-native timestamp to rely on -- filesystem metadata
#  is the true last resort.
# ------------------------------------------------------------------
_FILENAME_DATE_PATTERNS = [
    # 2026-04-17, 2026-04-14 to 2026-04-19 (takes the first date found)
    re.compile(r'(\d{4})-(\d{2})-(\d{2})'),
    # April 17 2026, April 19 2026
    re.compile(
        r'(January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
        re.IGNORECASE
    ),
]
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december"]

def _parse_filename_date(name: str) -> Optional[str]:
    """Returns ISO date string (YYYY-MM-DD) if a date is embedded in the
    filename, else None. Tries YYYY-MM-DD first, then 'Month DD YYYY'."""
    m = _FILENAME_DATE_PATTERNS[0].search(name)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            datetime(y, mo, d)  # validate
            return "{:04d}-{:02d}-{:02d}".format(y, mo, d)
        except ValueError:
            pass
    m = _FILENAME_DATE_PATTERNS[1].search(name)
    if m:
        try:
            month = _MONTH_NAMES.index(m.group(1).lower()) + 1
            d, y = int(m.group(2)), int(m.group(3))
            datetime(y, month, d)  # validate
            return "{:04d}-{:02d}-{:02d}".format(y, month, d)
        except ValueError:
            pass
    return None

def resolve_content_date(path: Path, fm: dict) -> Optional[str]:
    """Returns ISO date string for 'when this content is about', using the
    3-tier fallback. Used only for timebound_context files (Daily Notes /
    Archives) -- not needed for normal domain content."""
    dc = (fm or {}).get("date_created", "")
    if dc:
        try:
            # frontmatter date_created is usually 'YYYY-MM-DD' already
            datetime.strptime(dc[:10], "%Y-%m-%d")
            return dc[:10]
        except ValueError:
            pass

    from_name = _parse_filename_date(path.name)
    if from_name:
        return from_name

    try:
        # Windows: st_ctime is creation time. On Linux this would be
        # metadata-change time, but this script runs on Windows bare metal.
        ts = path.stat().st_ctime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return None

def get_db_conn():
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    return conn


# ------------------------------------------------------------------
#  FRONTMATTER PARSER
#  Extracts structured fields WITHOUT including YAML in chunk text
# ------------------------------------------------------------------
_YAML_BLOCK = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def parse_frontmatter(text: str) -> tuple:
    """
    Returns (body_text, fm_dict).
    body_text has YAML stripped and H1 title removed.
    fm_dict contains title, domain, note_type, note_status, tags.
    """
    fm = {}
    m = _YAML_BLOCK.match(text)
    if m:
        try:
            raw = yaml.safe_load(m.group(1)) or {}
            fm["title"]        = str(raw.get("title", ""))
            fm["domain"]       = str(raw.get("domain", ""))
            fm["note_type"]    = str(raw.get("type", ""))
            fm["note_status"]  = str(raw.get("status", ""))
            fm["date_created"] = str(raw.get("date_created", ""))
            tags = raw.get("tags", [])
            if isinstance(tags, list):
                fm["tags"] = [str(t) for t in tags]
            elif isinstance(tags, str):
                fm["tags"] = [tags]
            else:
                fm["tags"] = []
        except Exception:
            pass
        body = text[m.end():]
    else:
        body = text
    # Strip H1 title line (note name -- not content)
    body = re.sub(r'^#\s+.+\n', '', body.lstrip(), count=1)
    return body.strip(), fm

# ------------------------------------------------------------------
#  MARKDOWN CHUNKER  (heading-based, H1/H2/H3)
# ------------------------------------------------------------------
_HEADING_SPLIT = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)

def _word_chunk(text: str, max_w: int, overlap: int) -> list:
    words = text.split()
    if not words:
        return []
    chunks, step, i = [], max(1, max_w - overlap), 0
    while i < len(words):
        c = ' '.join(words[i:i + max_w])
        if c.strip():
            chunks.append(c)
        i += step
    return chunks

def chunk_markdown(path: Path) -> Optional[list]:
    """
    Returns list of (chunk_text, heading_path, None) tuples.
    YAML frontmatter is stripped and never embedded.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error("  Cannot read {}: {}".format(path.name, e))
        return None

    body, _ = parse_frontmatter(raw)
    if not body.strip():
        return None

    parts = _HEADING_SPLIT.split(body)
    results = []
    heading_stack = []  # (level, text)

    # Preamble before first heading
    if parts[0].strip():
        results.append((parts[0].strip(), "", None))

    i = 1
    while i < len(parts) - 1:
        hashes  = parts[i]
        h_text  = parts[i+1].strip()
        section = parts[i+2].strip() if i+2 < len(parts) else ""
        level   = len(hashes)
        i += 3

        # Maintain heading stack
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, h_text))

        crumb = " > ".join(h for _, h in heading_stack)
        prefix = "{} {}\n\n".format(hashes, crumb)

        if not section:
            continue

        words = section.split()
        if len(words) <= MD_CHUNK_MAX:
            results.append(((prefix + section).strip(), crumb, None))
        else:
            for idx, sub in enumerate(_word_chunk(section, MD_CHUNK_MAX, MD_CHUNK_OVERLAP)):
                cont = "" if idx == 0 else "[continued]\n\n"
                results.append(((prefix + cont + sub).strip(), crumb, None))

    return results if results else None


# ------------------------------------------------------------------
#  PDF EXTRACTION  (preserved from v2.0)
# ------------------------------------------------------------------
def is_image_based(path: Path) -> bool:
    try:
        import fitz
        doc = fitz.open(str(path))
        pages_to_check = min(5, len(doc))
        if pages_to_check == 0:
            return True
        total_chars = sum(len(doc[i].get_text().strip()) for i in range(pages_to_check))
        doc.close()
        return (total_chars / pages_to_check) < IMAGE_THRESHOLD
    except Exception:
        return True

def extract_pdf_pdftotext(path: Path) -> Optional[list]:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace"
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        pages = []
        for i, page_text in enumerate(result.stdout.split("\f")):
            text = page_text.strip()
            if text:
                pages.append({"text": text, "page_number": i + 1})
        return pages if pages else None
    except Exception:
        return None

def extract_pdf_pymupdf(path: Path) -> Optional[list]:
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "page_number": i + 1})
        doc.close()
        return pages if pages else None
    except Exception as e:
        logger.error("  pymupdf failed: {}".format(e))
        return None

class OCRError(RuntimeError):
    """Raised when OCR cannot run or fails. process_file() reports it as a
    failed file instead of quietly ingesting a near-empty pymupdf result."""

_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
]

def resolve_tesseract_cmd() -> str:
    """Find tesseract.exe. Order: TESSERACT_CMD env var, PATH, then the
    standard Windows install locations. Scheduled tasks don't get the
    installer's PATH entry, which is why pytesseract's default ("tesseract")
    used to fail on every page."""
    import os, shutil
    env = os.environ.get("TESSERACT_CMD")
    if env:
        if Path(env).is_file():
            return env
        raise OCRError("TESSERACT_CMD={} does not exist".format(env))
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for cand in _TESSERACT_CANDIDATES:
        if Path(cand).is_file():
            return cand
    raise OCRError("tesseract.exe not found (checked TESSERACT_CMD, PATH, {})".format(
        ", ".join(_TESSERACT_CANDIDATES)))

def extract_pdf_ocr(path: Path) -> Optional[list]:
    """OCR every page. Raises OCRError if Tesseract is missing or any page
    fails, so the file shows up as failed in the ingest log."""
    import io
    try:
        import fitz, pytesseract
        from PIL import Image
    except ImportError as e:
        raise OCRError("OCR dependency missing: {}".format(e)) from e

    pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd()

    doc = fitz.open(str(path))
    pages, page_errors = [], []
    mat = fitz.Matrix(2.0, 2.0)
    try:
        for i, page in enumerate(doc):
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang="eng").strip()
            except Exception as e:
                logger.error("  OCR failed on page {}: {}".format(i + 1, e))
                page_errors.append((i + 1, e))
                continue
            if text:
                pages.append({"text": text, "page_number": i + 1})
    finally:
        doc.close()

    if page_errors:
        first_page, first_err = page_errors[0]
        raise OCRError("OCR failed on {} page(s); first: p.{}: {}".format(
            len(page_errors), first_page, first_err))
    if not pages:
        # OCR ran fine but found no text (blank/decorative pages).
        logger.warning("  OCR found no text -- falling back to pymupdf")
        return extract_pdf_pymupdf(path)
    return pages

def extract_pdf(path: Path) -> Optional[list]:
    if is_image_based(path):
        logger.info("  Image-based -- OCR")
        return extract_pdf_ocr(path)
    logger.info("  Native text -- pdftotext")
    result = extract_pdf_pdftotext(path)
    if result:
        return result
    logger.info("  Fallback pymupdf")
    return extract_pdf_pymupdf(path)

# ------------------------------------------------------------------
#  PDF CHUNKER  (heading-aware, word-based fallback)
# ------------------------------------------------------------------
# Patterns that signal a heading line in pdftotext output:
#   "Chapter 3: Title"  |  "CHAPTER 3 TITLE"  |  "Section Name:"
#   Short title-case lines standing alone (< 80 chars, no sentence-ending punct)
_PDF_HEADING = re.compile(
    r'^(?:'
    r'(?:Chapter|CHAPTER|Section|SECTION|Part|PART)\s+[\dIVXivx]+[:\s].+|'  # Chapter N: Title
    r'[A-Z][A-Za-z\s\-]{2,60}:|'                                             # Title Case Line:
    r'[A-Z][A-Z\s]{4,59}$'                                                   # ALL CAPS LINE
    r')',
    re.MULTILINE
)

def _is_heading_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    if line.endswith(('.', ',', ';', '?', '!')):
        return False
    return bool(_PDF_HEADING.match(line))

def chunk_pdf_pages(pages: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Returns list of (chunk_text, heading_path, page_number) tuples.
    Splits on detected heading lines first; falls back to word-based chunking
    within sections that exceed chunk_size.
    """
    # Combine all pages into (text, page_num) segments
    full_text = ""
    page_map = []  # (char_start, char_end, page_num)
    for p in pages:
        start = len(full_text)
        full_text += p["text"] + "\n\n"
        page_map.append((start, len(full_text), p["page_number"]))

    def char_to_page(pos: int) -> int:
        for s, e, pn in page_map:
            if s <= pos < e:
                return pn
        return page_map[-1][2] if page_map else 1

    # Split on heading lines
    lines = full_text.split("\n")
    sections = []   # (heading, [lines])
    current_heading = ""
    current_lines = []

    for line in lines:
        if _is_heading_line(line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    results = []
    char_pos = 0

    for heading, body in sections:
        if not body.strip():
            char_pos += len(heading) + len(body) + 2
            continue
        pg = char_to_page(char_pos)
        words = body.split()
        prefix = "{}\n\n".format(heading) if heading else ""

        if len(words) <= chunk_size:
            text = (prefix + body).strip()
            if text:
                results.append((text, heading, pg))
        else:
            for idx, sub in enumerate(_word_chunk(body, chunk_size, chunk_overlap)):
                cont = "" if idx == 0 else "[continued] "
                text = (prefix + cont + sub).strip()
                if text:
                    results.append((text, heading, pg))
        char_pos += len(heading) + len(body) + 2

    return results if results else []


# ------------------------------------------------------------------
#  PDF DOC SUMMARIES  (contextual retrieval, PDF-only)
# ------------------------------------------------------------------
def get_pdf_summary(doc_path: Path, pages: list, file_hash: str) -> str:
    """1-2 sentence doc summary, cached in pdf_doc_summaries by file hash.

    Prepended to every chunk of the PDF so small (~500w) chunks still carry
    doc-level context at embed and BM25 time. Fail-open: any error returns
    "" and the chunks are stored unprefixed, exactly as before.
    """
    try:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS pdf_doc_summaries ("
                        "file_path text PRIMARY KEY, file_hash text, summary text, "
                        "model text, created_at timestamptz DEFAULT now())")
            cur.execute("SELECT summary FROM pdf_doc_summaries "
                        "WHERE file_path = %s AND file_hash = %s",
                        (str(doc_path), file_hash))
            row = cur.fetchone()
            if row and (row[0] or "").strip():
                conn.close()
                return row[0].strip()
        head = "\n\n".join(p["text"] for p in pages[:8])[:SUMMARY_MAX_CHARS]
        if not head.strip():
            conn.close()
            return ""
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer {}".format(OPENROUTER_API_KEY),
                     "Content-Type": "application/json"},
            json={"model": SUMMARY_MODEL,
                  "messages": [{"role": "user",
                                "content": ("You are describing a document for a retrieval index. "
                                            "In 1-2 sentences (max 40 words), state what this "
                                            "document is about. Reply with only the summary, no preamble.\n\n"
                                            "Document (head):\n" + head)}],
                  "max_tokens": SUMMARY_MAX_TOKENS, "temperature": 0.2},
            timeout=120,
        )
        r.raise_for_status()
        summary = r.json()["choices"][0]["message"]["content"].strip()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO pdf_doc_summaries (file_path, file_hash, summary, model) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (file_path) DO UPDATE SET "
                        "file_hash = EXCLUDED.file_hash, summary = EXCLUDED.summary, "
                        "model = EXCLUDED.model, created_at = now()",
                        (str(doc_path), file_hash, summary, SUMMARY_MODEL))
        conn.close()
        return summary
    except Exception as e:
        logger.warning("  summary failed for {}: {}".format(doc_path.name, str(e)[:120]))
        return ""


def chunk_pdf(path: Path, pages: list, chunk_size: int = PDF_CHUNK_MAX,
              chunk_overlap: int = PDF_CHUNK_OVERLAP) -> list:
    """PDF chunker used by process_file() and eval/build_variant.py.

    Heading-aware ~500w chunks, each prefixed with a Mistral-Small doc
    summary for doc-level context (PDFs have no titles/frontmatter).
    Signature chunk_pdf(path, pages) matches what build_variant prefers.
    """
    res = chunk_pdf_pages(pages, chunk_size, chunk_overlap)
    if not res:
        return res
    try:
        file_hash = sha256_file(path)
    except Exception:
        file_hash = ""
    summary = get_pdf_summary(path, pages, file_hash)
    if not summary:
        return res
    prefix = "[{}: {}]\n\n".format(path.stem, summary)
    return [(prefix + t, h, p) for t, h, p in res]


# ------------------------------------------------------------------
#  EMBED  (OpenRouter Qwen3-Embedding-8B)
# ------------------------------------------------------------------
def embed_input(text: str) -> str:
    """Exact string sent to the embedding API for a document chunk.

    Qwen3-Embedding takes an instruction on the query side only; documents
    are embedded raw. The old "search_document: " prefix was a nomic-embed
    leftover. Changing this changes every stored vector: after editing it,
    re-embed everything (--clear-folder <vault-folder>, then
    --clear-folder pgvector-rag-sources, then a normal run) so the index
    is never a mix of both styles.
    """
    return text

def get_embeddings_batch(texts: list) -> list:
    """Embed list of texts via OpenRouter. Returns list of vectors.

    Pinned to Nebius provider: OpenRouter's default 'Balanced' routing
    intermittently lands on DeepInfra for this model (~23s/call vs Nebius's
    ~1s), multiplying badly across ingest batches. Confirmed 2026-07-10.
    """
    all_embeddings = []
    headers = {
        "Authorization": "Bearer {}".format(OPENROUTER_API_KEY),
        "Content-Type": "application/json",
    }
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        prefixed = [embed_input(t) for t in batch]
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json={
                "model": EMBED_MODEL,
                "input": prefixed,
                "dimensions": 4000,
                "provider": {"only": ["Nebius"]},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenRouter returns {"data": [{"embedding": [...], "index": N}, ...]}
        batch_embeddings = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(e["embedding"] for e in batch_embeddings)
    return all_embeddings

# ------------------------------------------------------------------
#  DATABASE OPS
# ------------------------------------------------------------------
def get_all_db_files(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_path, id, file_hash, is_excluded, chunk_count "
            "FROM source_files"
        )
        return {
            row[0]: {"id": row[1], "hash": row[2], "excluded": row[3], "chunk_count": row[4]}
            for row in cur.fetchall()
        }

def upsert_source_record(conn, file_path: str, file_hash: str, doc_type: str,
                         fm: dict, content_date: Optional[str] = None) -> int:
    """Insert/update source_files row including frontmatter fields. Returns id.

    content_date is only set for timebound_context files (Daily Notes /
    Archives) -- see resolve_content_date() and is_timebound(). For all
    other files it stays NULL, which is fine since only
    rag_search_time_bound reads this column.

    NOTE: domain here is intentionally the raw frontmatter domain for
    non-timebound files, but process_file() overrides it to
    'timebound_context' before calling this for Daily Notes/Archives --
    that override must happen in fm's caller context, not here, since fm
    itself isn't mutated. See process_file()."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO source_files
               (file_path, file_hash, doc_type, chunk_count,
                title, domain, note_type, note_status, tags, content_date)
               VALUES (%s, %s, %s, 0, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (file_path) DO UPDATE SET
                   file_hash    = EXCLUDED.file_hash,
                   last_ingested = now(),
                   chunk_count  = 0,
                   title        = EXCLUDED.title,
                   domain       = EXCLUDED.domain,
                   note_type    = EXCLUDED.note_type,
                   note_status  = EXCLUDED.note_status,
                   tags         = EXCLUDED.tags,
                   content_date = EXCLUDED.content_date
               RETURNING id""",
            (
                file_path, file_hash, doc_type,
                fm.get("title") or None,
                # domain column: force timebound_context for timebound
                # paths (regardless of whether content_date resolved),
                # else fall back to frontmatter domain like before
                "timebound_context" if is_timebound(file_path)
                    else (fm.get("domain") or None),
                fm.get("note_type") or None,
                fm.get("note_status") or None,
                fm.get("tags") or None,
                content_date,
            )
        )
        return cur.fetchone()[0]

def write_chunks(conn, source_id: int, chunk_tuples: list,
                 doc_path: Path, domain: str):
    """
    chunk_tuples: list of (chunk_text, heading_path, page_number)
    Deletes old chunks, embeds, inserts new ones.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))

    texts = [t for t, _, _ in chunk_tuples]
    embeddings = get_embeddings_batch(texts)

    with conn.cursor() as cur:
        for idx, ((chunk, heading, page_num), embedding) in enumerate(
            zip(chunk_tuples, embeddings)
        ):
            meta = {
                "source":      doc_path.name,
                "full_path":   str(doc_path),
                "chunk_index": idx,
                "file_type":   doc_path.suffix.lower().lstrip("."),
                "ingested_at": datetime.now().isoformat(),
            }
            cur.execute(
                """INSERT INTO chunks
                   (source_id, chunk_index, chunk_text, token_count,
                    heading_path, page_number, domain, embedding, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
                (
                    source_id, idx, chunk, len(chunk.split()),
                    heading or None,
                    page_num,
                    domain or None,
                    json.dumps(embedding),
                    json.dumps(meta),
                )
            )

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE source_files SET chunk_count = %s WHERE id = %s",
            (len(chunk_tuples), source_id)
        )


# ------------------------------------------------------------------
#  PROCESS ONE FILE
# ------------------------------------------------------------------
def process_file(conn, doc_path: Path, doc_type: str, db_info: Optional[dict],
                 chunk_size: int, chunk_overlap: int) -> tuple:
    path_str = str(doc_path)

    if is_excluded(path_str, doc_type):
        return ("excluded", 0, None)

    try:
        file_hash = sha256_file(doc_path)
    except Exception as e:
        return ("failed", 0, "hash error: {}".format(e))

    if db_info and db_info["hash"] == file_hash and db_info["chunk_count"] > 0:
        return ("skipped", 0, None)

    action = "ingested"
    if db_info and db_info["hash"] != file_hash:
        action = "updated"
    elif db_info and db_info["chunk_count"] == 0:
        action = "repair"

    try:
        fm = {}
        if doc_type == "markdown":
            chunk_tuples = chunk_markdown(doc_path)
            # Also parse frontmatter for source_files columns
            raw = doc_path.read_text(encoding="utf-8", errors="ignore")
            _, fm = parse_frontmatter(raw)
        else:
            pages = extract_pdf(doc_path)
            if not pages:
                return ("failed", 0, "empty or unreadable")
            chunk_tuples = chunk_pdf(doc_path, pages, chunk_size, chunk_overlap)

        if not chunk_tuples:
            return ("failed", 0, "no chunks produced")

    except Exception as e:
        return ("failed", 0, "extraction error: {}".format(e))

    content_date = None
    if is_timebound(path_str):
        # Daily Notes / Archives: force domain regardless of any frontmatter
        # domain value, and resolve a content_date for chronological sort
        # in rag_search_time_bound.
        domain = "timebound_context"
        content_date = resolve_content_date(doc_path, fm)
    else:
        domain = fm.get("domain", "") or ""

    try:
        source_id = upsert_source_record(conn, path_str, file_hash, doc_type, fm, content_date)
        write_chunks(conn, source_id, chunk_tuples, doc_path, domain)
        conn.commit()
        return (action, len(chunk_tuples), None)
    except Exception as e:
        conn.rollback()
        return ("failed", 0, str(e))

# ------------------------------------------------------------------
#  COLLECT DISK FILES
# ------------------------------------------------------------------
def collect_disk_files(subset: Optional[str] = None,
                       missing_roots: Optional[list] = None) -> list:
    """Returns unique (path, doc_type) tuples found on disk.

    If missing_roots is a list, every source root that does not exist (or
    exists but yields zero files, e.g. an unsynced Dropbox/OneDrive folder)
    is appended to it so the caller can refuse to purge DB rows under it."""
    files = []
    for source_type, root in SOURCE_ROOTS.items():
        if not root.exists():
            logger.warning("  Source root not found: {}".format(root))
            if missing_roots is not None:
                missing_roots.append(root)
            continue
        before = len(files)
        ext      = ".md"  if source_type == "vault" else ".pdf"
        doc_type = "markdown" if ext == ".md"  else "pdf"
        for f in root.rglob("*{}".format(ext)):
            if not f.is_file():
                continue
            if subset and subset not in str(f):
                continue
            if f.stem.endswith(' - Draft'):
                continue
            files.append((f, doc_type))
        if not subset and len(files) == before:
            logger.warning("  Source root is empty: {}".format(root))
            if missing_roots is not None:
                missing_roots.append(root)
    seen, unique = set(), []
    for f, t in files:
        if f not in seen:
            seen.add(f)
            unique.append((f, t))
    return unique

# ------------------------------------------------------------------
#  PURGE PLANNING  (safety guards -- see plan_purge docstring)
# ------------------------------------------------------------------
PURGE_MAX_FRACTION = 0.05   # refuse to purge more than ~5% of tracked files
PURGE_MIN_ALLOWED  = 3      # ...but always allow a handful on small DBs

def plan_purge(db_paths, disk_paths, subset: Optional[str] = None,
               missing_roots: Optional[list] = None,
               force: bool = False) -> tuple:
    """Decide which DB rows to delete because their file vanished from disk.

    Returns (to_delete: list[str], blocked_reason: Optional[str]).
    When blocked_reason is set, to_delete is empty and nothing may be purged.

    Guards (bug fixed 2026-10-29):
      1. --subset run: disk_paths only covers the subset, so every row
         outside it looked "deleted" and was purged. Cleanup is skipped.
      2. Missing/empty source root (e.g. Dropbox not mounted): every PDF row
         looked deleted and was wiped. Cleanup is skipped.
      3. Mass-deletion circuit breaker: if the purge would remove more than
         max(PURGE_MAX_FRACTION of tracked files, PURGE_MIN_ALLOWED), abort
         it unless force=True (--force-purge).
    """
    if subset:
        return [], "subset run (--subset {!r}); cleanup skipped".format(subset)
    if missing_roots:
        return [], "source root(s) missing or empty: {}; cleanup skipped".format(
            ", ".join(str(r) for r in missing_roots))

    disk_paths = set(disk_paths)
    db_paths = list(db_paths)
    to_delete = [p for p in db_paths if p not in disk_paths]
    limit = max(int(len(db_paths) * PURGE_MAX_FRACTION), PURGE_MIN_ALLOWED)
    if len(to_delete) > limit and not force:
        return [], ("purge would delete {} of {} tracked files (limit {}); "
                    "aborted -- re-run with --force-purge if intended".format(
                        len(to_delete), len(db_paths), limit))
    return to_delete, None

# ------------------------------------------------------------------
#  MAIN INGEST
# ------------------------------------------------------------------
def ingest(subset: Optional[str] = None, dry_run: bool = False,
           force_purge: bool = False) -> int:
    """Returns process exit code: 0 ok, 2 if the purge safety guard tripped."""
    logger.info("\n" + "=" * 60)
    logger.info("  RAG Ingest pgvector v3.0{}".format("  [DRY RUN]" if dry_run else ""))
    logger.info("  Embed: {}".format(EMBED_MODEL))
    logger.info("  {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    logger.info("=" * 60)

    conn       = get_db_conn()
    db_files   = get_all_db_files(conn)
    missing_roots = []
    disk_files = collect_disk_files(subset, missing_roots)

    logger.info("\n  DB state   : {} file(s) tracked".format(len(db_files)))
    logger.info("  Disk files : {} file(s) found".format(len(disk_files)))

    to_process = []
    skipped    = 0
    repairs    = 0

    for doc_path, doc_type in disk_files:
        if doc_path.name in SKIP_FILES:
            continue
        path_str = str(doc_path)
        if is_excluded(path_str, doc_type):
            continue
        db_info = db_files.get(path_str)
        if db_info is None:
            to_process.append((doc_path, doc_type, None))
        elif db_info["chunk_count"] == 0:
            to_process.append((doc_path, doc_type, db_info))
            repairs += 1
        else:
            try:
                if db_info["hash"] != sha256_file(doc_path):
                    to_process.append((doc_path, doc_type, db_info))
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    logger.info("  To process : {} file(s) ({} repairs)".format(len(to_process), repairs))
    logger.info("  Skipped    : {} (unchanged)\n".format(skipped))

    # Purge deleted files
    disk_paths = {str(f) for f, _ in disk_files}
    to_delete, purge_blocked = plan_purge(
        db_files.keys(), disk_paths, subset, missing_roots, force_purge)
    exit_code = 0
    if purge_blocked:
        if subset:
            logger.info("  Purge      : skipped -- {}".format(purge_blocked))
        else:
            logger.error("  PURGE BLOCKED: {}".format(purge_blocked))
            exit_code = 2
    purged = 0
    for path_str in to_delete:
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM source_files WHERE file_path = %s", (path_str,))
            conn.commit()
        logger.info("  [{}] {}".format("WOULD PURGE" if dry_run else "PURGED ",
                                       Path(path_str).name))
        purged += 1

    if not to_process:
        logger.info("  Everything up to date.")
        _print_summary(conn)
        conn.close()
        return exit_code

    if dry_run:
        for doc_path, doc_type, _ in to_process:
            logger.info("  [WOULD  ] {}".format(doc_path.name))
        _print_summary(conn)
        conn.close()
        return exit_code

    # Pass 1: default sizes
    total_chunks = 0
    failed = []
    logger.info("  Pass 1: MD {}/{} | PDF {}/{}\n".format(
        MD_CHUNK_MAX, MD_CHUNK_OVERLAP, PDF_CHUNK_MAX, PDF_CHUNK_OVERLAP))

    for doc_path, doc_type, db_info in to_process:
        logger.info("[{}] {}".format(doc_type, doc_path.name))
        size    = PDF_CHUNK_MAX     if doc_type == "pdf" else MD_CHUNK_MAX
        overlap = PDF_CHUNK_OVERLAP if doc_type == "pdf" else MD_CHUNK_OVERLAP
        status, n, err = process_file(conn, doc_path, doc_type, db_info, size, overlap)
        if status == "failed":
            logger.error("  Failed: {}".format(err))
            failed.append((doc_path, doc_type, db_info))
        else:
            total_chunks += n
            logger.info("  {} chunk(s) -- {}".format(n, status))

    # Auto-retry passes
    for pass_num, (retry_size, retry_overlap) in enumerate(RETRY_SCHEDULE, start=2):
        if not failed:
            break
        logger.info("\n  Pass {}: {} words / {} overlap ({} file(s))\n".format(
            pass_num, retry_size, retry_overlap, len(failed)))
        still_failed = []
        for doc_path, doc_type, db_info in failed:
            logger.info("[{}] {}".format(doc_type, doc_path.name))
            status, n, err = process_file(
                conn, doc_path, doc_type, db_info, retry_size, retry_overlap)
            if status == "failed":
                logger.error("  Failed: {}".format(err))
                still_failed.append((doc_path, doc_type, db_info))
            else:
                total_chunks += n
                logger.info("  {} chunk(s) -- {} (retry {})".format(n, status, pass_num))
        failed = still_failed

    logger.info("\n" + "=" * 60)
    logger.info("  Ingest complete!")
    logger.info("  Processed : {}".format(len(to_process) - len(failed)))
    logger.info("  Chunks    : {:,}".format(total_chunks))
    logger.info("  Failures  : {}".format(len(failed)))
    if purged:
        logger.info("  Purged    : {}".format(purged))
    logger.info("=" * 60)
    _print_summary(conn)
    conn.close()

    if failed:
        logger.warning("\n  Failed files:")
        for doc_path, _, _ in failed:
            logger.warning("   {}".format(doc_path))
    return exit_code

def _print_summary(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_type, COUNT(*) as files, COALESCE(SUM(chunk_count),0) as chunks "
            "FROM source_files GROUP BY doc_type ORDER BY doc_type"
        )
        rows = cur.fetchall()
    logger.info("\n  pgvector totals:")
    for doc_type, files, chunks in rows:
        logger.info("   {:<12}: {:>5} files  {:>7,} chunks".format(doc_type, files, chunks))

# ------------------------------------------------------------------
#  CLI HANDLERS
# ------------------------------------------------------------------
def handle_remove_file(filename: str):
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE source_files SET file_hash = '' WHERE file_path LIKE %s",
            ("%{}%".format(filename),)
        )
        count = cur.rowcount
    conn.commit()
    conn.close()
    logger.info("  Reset {} entry/entries matching '{}'.".format(count, filename))

def handle_clear_folder(folder: str):
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE source_files SET file_hash = '' WHERE file_path LIKE %s",
            ("%{}%".format(folder),)
        )
        count = cur.rowcount
    conn.commit()
    conn.close()
    logger.info("  Reset {} entries for folder '{}'.".format(count, folder))

# ------------------------------------------------------------------
#  ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG pgvector ingest v3.0")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--subset",       type=str, default=None)
    parser.add_argument("--remove-file",  metavar="FILENAME")
    parser.add_argument("--clear-folder", metavar="FOLDER")
    parser.add_argument("--force-purge",  action="store_true",
                        help="override the ~5%% mass-deletion guard "
                             "(subset/missing-root guards still apply)")
    args = parser.parse_args()

    try:
        if args.remove_file:
            handle_remove_file(args.remove_file)
        elif args.clear_folder:
            handle_clear_folder(args.clear_folder)
        else:
            sys.exit(ingest(subset=args.subset, dry_run=args.dry_run,
                            force_purge=args.force_purge))
    except KeyboardInterrupt:
        logger.info("\n  Cancelled.")
        sys.exit(0)
    except Exception as e:
        logger.error("\n  Fatal: {}".format(e))
        raise
