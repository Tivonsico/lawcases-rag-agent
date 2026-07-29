"""RAG Agent 配置文件"""
import os
from pathlib import Path

# 加载 .env 文件（如存在），使环境变量在读 getenv 前就绪
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.is_file():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, val)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(PROJECT_ROOT)


def _env_path(name, default):
    """Resolve an optional path override without requiring the target to exist."""
    return os.path.abspath(os.path.expanduser(os.environ.get(name, default)))


def _has_chroma_data(path):
    """Detect an already-populated Chroma directory without opening the DB."""
    if not os.path.isdir(path):
        return False
    manifest = os.path.join(path, "index_manifest.json")
    if os.path.isfile(manifest):
        return True
    sqlite_path = os.path.join(path, "chroma.sqlite3")
    return os.path.isfile(sqlite_path) and os.path.getsize(sqlite_path) > 1024 * 1024


def _select_index_file(env_name, runtime_path, legacy_path):
    """Prefer an explicit override, then existing runtime or legacy index data."""
    if os.environ.get(env_name):
        return _env_path(env_name, runtime_path)
    if os.path.isfile(runtime_path):
        return os.path.abspath(runtime_path)
    if os.path.isfile(legacy_path):
        return os.path.abspath(legacy_path)
    return os.path.abspath(runtime_path)


# New generated state belongs under RUNTIME_DIR. Existing in-package artifacts are
# deliberately not moved or deleted; override these variables to keep using them.
DATA_DIR = _env_path("LEGAL_RAG_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
RUNTIME_DIR = _env_path("LEGAL_RAG_RUNTIME_DIR", os.path.join(REPOSITORY_ROOT, "runtime"))
INDEX_DIR = _env_path("LEGAL_RAG_INDEX_DIR", os.path.join(RUNTIME_DIR, "indexes"))
REPORT_DIR = _env_path("LEGAL_RAG_REPORT_DIR", os.path.join(RUNTIME_DIR, "reports"))
SESSION_DIR = _env_path("LEGAL_RAG_SESSION_DIR", os.path.join(RUNTIME_DIR, "sessions"))

# ── 文档路径 ──
DOC_DIR = _env_path("LEGAL_RAG_DOC_DIR", os.path.join(DATA_DIR, "legal_cases"))

# ── Chunk 参数 ──
CHUNK_SIZE = 200          # 基准字数
CHUNK_MIN = 50            # 最小chunk字数(小于此值合并)

# ── Embedding (千问3 / DashScope) ──
EMBEDDING_DIM = 1024      # 向量维度(可调)
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

# ── 向量数据库 ──
_CHROMA_RUNTIME_DIR = os.path.join(INDEX_DIR, "chroma")
_CHROMA_LEGACY_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
_CHROMA_OVERRIDE = os.environ.get("LEGAL_RAG_CHROMA_DIR")
if _CHROMA_OVERRIDE:
    CHROMA_PERSIST_DIR = _env_path("LEGAL_RAG_CHROMA_DIR", _CHROMA_RUNTIME_DIR)
elif _has_chroma_data(_CHROMA_RUNTIME_DIR):
    CHROMA_PERSIST_DIR = _CHROMA_RUNTIME_DIR
elif _has_chroma_data(_CHROMA_LEGACY_DIR):
    CHROMA_PERSIST_DIR = _CHROMA_LEGACY_DIR
else:
    CHROMA_PERSIST_DIR = _CHROMA_RUNTIME_DIR
_BM25_RUNTIME_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
_BM25_LEGACY_PATH = os.path.join(PROJECT_ROOT, "bm25_index.pkl")
BM25_INDEX_PATH = _select_index_file(
    "LEGAL_RAG_BM25_PATH", _BM25_RUNTIME_PATH, _BM25_LEGACY_PATH
)
CHROMA_COLLECTION = "legal_chunks"

# ── 检索 ──
HYBRID_TOP_K = 3          # BM25 和 Embedding 各取前 N 个，按倒数排名融合取前 2

# ── LLM (DeepSeek) ──
VECTOR_CANDIDATE_K = 200
BM25_CANDIDATE_K = 200
DOCUMENT_CANDIDATE_K = 50
FINAL_TOP_K = 5
RRF_K = 60
RRF_WEIGHTS = {"embedding_original": 2.0, "embedding_normalized": 0.5,
               "bm25_original": 0.6, "exact": 2.0}

LLM_API_URL = os.environ.get("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ── 记忆 ──
SHORT_MEMORY_SIZE = 5     # 短期记忆保留最近 N 轮

# ── Redis (短期记忆) ──
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# ── SQLite (长期记忆) ──
LONG_TERM_DB = _env_path("LEGAL_RAG_LONG_TERM_DB", os.path.join(RUNTIME_DIR, "long_term_memory.db"))

# ── 日志 ──
LOG_LEVEL = "INFO"
