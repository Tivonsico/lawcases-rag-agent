"""One-time, validated migration for the pre-manifest local indexes."""
import os

try:
    from .config import CHROMA_PERSIST_DIR, INDEX_DIR, PROJECT_ROOT
    from .embedding import EmbeddingService
    from .init_db import migrate_legacy_index
    from .vector_store import VectorStore
except ImportError:  # pragma: no cover - direct script compatibility
    from config import CHROMA_PERSIST_DIR, INDEX_DIR, PROJECT_ROOT
    from embedding import EmbeddingService
    from init_db import migrate_legacy_index
    from vector_store import VectorStore


def main():
    source = os.path.join(PROJECT_ROOT, "bm25_index.pkl")
    target = os.path.join(INDEX_DIR, "bm25_index.pkl")
    if not os.path.isfile(source):
        raise FileNotFoundError(f"旧 BM25 索引不存在: {source}")
    store = VectorStore(EmbeddingService(mock=True), CHROMA_PERSIST_DIR)
    result = migrate_legacy_index(source, target, store)
    print(f"迁移完成: {result['chunks']} chunks -> {result['target']}")


if __name__ == "__main__":
    main()
