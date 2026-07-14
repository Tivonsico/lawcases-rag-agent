"""Legal RAG package.

Public objects are loaded lazily so importing a lightweight module (notably the API
application factory) does not initialize optional retrieval dependencies.
"""
from importlib import import_module

_EXPORTS = {
    "EmbeddingService": ("embedding", "EmbeddingService"),
    "VectorStore": ("vector_store", "VectorStore"),
    "IntentProcessor": ("intent", "IntentProcessor"),
    "clean_query": ("intent", "clean_query"),
    "extract_keywords": ("intent", "extract_keywords"),
    "HybridRetriever": ("retriever", "HybridRetriever"),
    "BM25Index": ("retriever", "BM25Index"),
    "SummaryBufferMemory": ("memory", "SummaryBufferMemory"),
    "LongTermMemory": ("memory", "LongTermMemory"),
    "MemoryManager": ("memory", "MemoryManager"),
    "LegalAgent": ("agent", "LegalAgent"),
    "chunk_document": ("chunker", "chunk_document"),
    "chunk_all_documents": ("chunker", "chunk_all_documents"),
    "load_documents": ("chunker", "load_documents"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
