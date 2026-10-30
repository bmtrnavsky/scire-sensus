# config.py -- pgvector RAG configuration v3.0 (ported to Windows bare metal)
# Upgraded: nomic-embed-text (768d local) -> qwen3-embedding-8b (4096d OpenRouter)
import os
from pathlib import Path


def _load_dotenv(path=Path(__file__).with_name(".env")):
    """Minimal .env loader (no dependency). Real env vars take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _require(name):
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var {name} (set it or add it to rag/.env; see .env.example)")
    return val


_load_dotenv()

DB_DSN = _require("DB_DSN")

def _env_path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


SOURCE_ROOTS = {
    # Point these at your own folders via VAULT_ROOT / PDF_ROOT_DIR,
    # or edit the defaults below.
    "vault": _env_path("VAULT_ROOT", "~/documents/notes-vault"),
    "pdfs":  _env_path("PDF_ROOT_DIR", "~/documents/rag-sources"),
}

SUPPORTED_EXTS = {".md", ".pdf"}

EXCLUDE_MARKERS = [
    # NOTE: "/70 Archives/" and "/10 Daily Notes/" removed 2026-09-05 --
    # both now get ingested and tagged domain='timebound_context' instead
    # of being excluded (see TIMEBOUND_MARKERS check in
    # ingest_rag_pgvector_v3.py). Don't re-add them here.
    "/00 Meta/Templates/",
    "/99 System/",
    "/Council of Rivals Backup/",
    ".obsidian",
    "(conflict ",
]

# Paths that get ingested normally but tagged domain='timebound_context'
# instead of appearing in default rag_search/rag_semantic results.
# Queried separately via rag_search_time_bound. See ingest_rag_pgvector_v3.py.
TIMEBOUND_MARKERS = [
    "/70 Archives/",
    "/10 Daily Notes/",
]

EXCLUDE_PDF_MARKERS = []

# --- Embedding: OpenRouter Qwen3-Embedding-8B ---
OPENROUTER_API_KEY = _require("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL        = "qwen/qwen3-embedding-8b"
EMBED_DIM          = 4000   # halfvec(4000) -- Matryoshka truncation, fits pgvector HNSW limit
EMBED_BATCH_SIZE   = 5

# Qwen3-Embedding query instruction (documents get no prefix). Format per the
# model card: "Instruct: {task}\nQuery: {q}".
QUERY_INSTRUCT = ("Instruct: Given a question, retrieve relevant passages from a "
                  "personal knowledge vault that answer it\nQuery: ")

OLLAMA_URL  = "http://localhost:11434"  # kept for legacy reference
FLASK_PORT  = 5556

DEFAULT_TOP_K = 8
MAX_TOP_K     = 12

# --- Chunking ---
MD_CHUNK_MAX     = 400    # word cap per heading section
MD_CHUNK_OVERLAP = 50

PDF_CHUNK_MAX     = 500   # word cap per section (was 2000; ~500 dilutes embeddings less)
PDF_CHUNK_OVERLAP = 75

# --- PDF doc summaries (contextual retrieval, PDF-only) ---
# One short summary per PDF via OpenRouter chat, prepended to each chunk of
# that doc before embedding. MD notes are self-titled and need no prefix.
SUMMARY_MODEL      = "mistralai/mistral-small-3.2-24b-instruct"
SUMMARY_MAX_CHARS  = 6000   # head characters sent to the summarizer
SUMMARY_MAX_TOKENS = 100

# --- Reranker (Cohere via OpenRouter) ---
# Rerank 5 checked 2026-10-29: NOT on OpenRouter (400 "does not exist" for
# cohere/rerank-5, -5-pro, -5-fast, -v5). Stay on Rerank 4 Pro; flip
# RERANK_MODEL to "cohere/rerank-5-pro" when it lands. Rerank 4 scores are
# uncalibrated, so RERANK_MIN_SCORE is a garbage-trim (0.05), not a
# calibrated relevance threshold -- revisit with Rerank 5's calibrated scores.
RERANK_MODEL      = "cohere/rerank-4-pro"
RERANK_POOL       = 45    # RRF candidates fed to the reranker (was 20)
RERANK_MIN_SCORE  = 0.05  # drop reranked docs below this ...
RERANK_MIN_KEEP   = 3     # ... but always keep at least this many
RERANK_FALLBACK_LOG = str(Path(__file__).with_name("rerank_fallback.log"))
