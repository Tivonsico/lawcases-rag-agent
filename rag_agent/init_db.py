"""初始化知识库：加载文档 → 切分 → 构建 BM25 → 存入向量库"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import chunk_all_documents
from embedding import EmbeddingService
from vector_store import VectorStore
from retriever import BM25Index
from config import DOC_DIR, BM25_INDEX_PATH


def load_bm25_index(path=BM25_INDEX_PATH, vector_store=None):
    """Load the trusted configured pickle and validate its index manifest.

    Pickle is executable input. Never pass an uploaded or request-controlled path.
    """
    import pickle

    trusted_path = os.path.abspath(path)
    if not os.path.isfile(trusted_path):
        raise FileNotFoundError(f"BM25 index is missing: {trusted_path}; rebuild the index")
    with open(trusted_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list) or not data.get("manifest"):
        raise RuntimeError("BM25 index has no valid manifest; rebuild the index")
    if vector_store is not None:
        vector_store.validate_manifest(data["manifest"])
    bm25 = BM25Index()
    bm25.build(data["chunks"])
    return bm25, data["chunks"], data["manifest"]


def migrate_legacy_index(source_path, target_path, vector_store):
    """Upgrade a trusted legacy BM25 pickle after matching it to Chroma IDs."""
    import pickle

    source = os.path.abspath(source_path)
    target = os.path.abspath(target_path)
    with open(source, "rb") as f:
        data = pickle.load(f)
    chunks = data.get("chunks") if isinstance(data, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("旧 BM25 索引不包含有效 chunks，无法迁移")
    upgraded_chunks = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or f"{chunk.get('doc_name', '')}_chunk{chunk.get('chunk_index')}"
        upgraded_chunks.append({**chunk, "chunk_id": chunk_id})
    chunk_ids = {c["chunk_id"] for c in upgraded_chunks}
    vector_ids = set(vector_store.collection.get(include=[])["ids"])
    if chunk_ids != vector_ids:
        raise RuntimeError(
            f"旧索引 ID 不一致，拒绝迁移: BM25={len(chunk_ids)}, Chroma={len(vector_ids)}"
        )
    manifest = vector_store.build_manifest(upgraded_chunks)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp_path = target + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            pickle.dump({"manifest": manifest, "chunks": upgraded_chunks}, f)
        vector_store.write_manifest(manifest)
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return {"chunks": len(upgraded_chunks), "manifest": manifest, "target": target}


def build_indexes(doc_dir=DOC_DIR, embed_service=None, vector_store=None, bm25_path=None):
    """两个 CLI 入口共享的幂等构建逻辑。"""
    chunks = chunk_all_documents(doc_dir)
    if not chunks:
        raise RuntimeError("没有成功切分的文档")
    embed = embed_service or EmbeddingService(mock=False)
    vs = vector_store or VectorStore(embed_service=embed)
    manifest = vs.build_manifest(chunks)
    vs.validate_manifest(manifest, allow_missing=True)
    vs.add_chunks(chunks)
    if vs.count() != len(chunks):
        raise RuntimeError(f"向量索引数量不一致: {vs.count()} != {len(chunks)}")
    vs.write_manifest(manifest)
    bm25 = BM25Index()
    bm25.build(chunks)
    import pickle
    path = os.path.abspath(bm25_path or BM25_INDEX_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"manifest": manifest, "chunks": chunks}, f)
    return {"chunks": chunks, "manifest": manifest, "vector_store": vs, "bm25": bm25}


def main():
    print("=" * 60)
    print("法律案例 RAG 知识库初始化")
    print("=" * 60)

    # 检查文档目录
    doc_dir = DOC_DIR
    if not os.path.isdir(doc_dir):
        print(f"[ERROR] 文档目录不存在: {doc_dir}")
        return

    result = build_indexes(doc_dir)
    chunks, vs = result["chunks"], result["vector_store"]

    print("\n" + "=" * 60)
    print("知识库初始化完成！")
    print(f"  文档数: {len(chunks)} chunks")
    print(f"  向量库: {vs.count()} 条")
    print("=" * 60)
    print("\n运行 python main.py 启动交互式问答")


if __name__ == "__main__":
    main()
