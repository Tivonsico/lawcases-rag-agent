"""
向量数据库模块
使用 ChromaDB 做持久化向量存储
以 chunk 标签作为索引
"""
import os
import json
import hashlib
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

try:
    from .config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_MIN
    from .embedding import EmbeddingService
except ImportError:  # pragma: no cover - direct script compatibility
    from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_MIN
    from embedding import EmbeddingService


class VectorStore:
    def __init__(self, embed_service: EmbeddingService = None, persist_dir: str = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.embed_service = embed_service or EmbeddingService(mock=True)

        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 collection
        self.collection = self._get_or_create_collection()
        self.manifest_path = os.path.join(self.persist_dir, "index_manifest.json")

    def build_manifest(self, chunks: List[Dict]) -> Dict:
        rows = sorted(f"{c['chunk_id']}\0{c['text']}" for c in chunks)
        return {"schema_version": "2", "corpus_hash": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
                "chunk_count": len(chunks), "chunk_size": CHUNK_SIZE, "chunk_min": CHUNK_MIN,
                "embedding_model": self.embed_service.model,
                "embedding_dimension": self.embed_service.dimension}

    def validate_manifest(self, expected: Dict, allow_missing: bool = False):
        if not os.path.exists(self.manifest_path):
            if allow_missing and self.count() == 0:
                return
            raise RuntimeError("索引缺少 manifest；请重建索引")
        with open(self.manifest_path, encoding="utf-8") as f:
            actual = json.load(f)
        mismatches = [k for k, v in expected.items() if actual.get(k) != v]
        if mismatches:
            raise RuntimeError(f"索引 manifest 不兼容 ({', '.join(mismatches)})；请重建索引")

    def write_manifest(self, manifest: Dict):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)

    def _get_or_create_collection(self):
        """获取已有 collection 或创建新的"""
        try:
            return self.client.get_collection(CHROMA_COLLECTION)
        except (ValueError, NotFoundError):
            return self.client.create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    def add_chunks(self, chunks: List[Dict], batch_size: int = 500):
        """
        将 chunks 存入向量数据库
        chunks: [{tag, text, doc_name, chunk_index, keywords, prev_context}]
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        tags = [c["tag"] for c in chunks]
        metadatas = [
            {
                "doc_name": c["doc_name"],
                "chunk_index": c["chunk_index"],
                "keywords": json.dumps(c.get("keywords", []), ensure_ascii=False),
                "prev_context": c.get("prev_context", "")[:200],
                "case_id": c["case_id"], "case_type": c.get("case_type", "unknown"),
                "case_number": c.get("case_number", "unknown"), "source_id": c.get("source_id", "unknown"),
                "cause": c.get("cause", "unknown"), "category": c.get("category", "unknown"),
                "authority": c.get("authority", "unknown"), "trial_level": c.get("trial_level", "unknown"),
                "publication_year": c.get("publication_year", "unknown"),
                "judgment_year": c.get("judgment_year", "unknown"),
                "legal_provisions": json.dumps(c.get("legal_provisions", []), ensure_ascii=False),
                "charge": c.get("charge", "unknown"),
                "entities": json.dumps(c.get("entities", []), ensure_ascii=False),
                "behaviors": json.dumps(c.get("behaviors", []), ensure_ascii=False),
                "dispute_focus": c.get("dispute_focus", "unknown"),
                "section_type": c.get("section_type", "body"),
                "validity_status": c.get("validity_status", "unknown"),
                "schema_version": c.get("schema_version", "2"),
            }
            for c in chunks
        ]

        # 生成唯一 ID
        ids = [c["chunk_id"] for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("chunk_id 冲突，拒绝写入索引")

        # 分批生成向量 + 分批写入 ChromaDB
        # embedding API 内部每批限制 10 条
        EMBED_BATCH = 10
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]
            batch_metas = metadatas[start:end]

            # 生成向量（内部再按10条分批调API）
            all_vecs = []
            for j in range(0, len(batch_texts), EMBED_BATCH):
                sub = batch_texts[j:j+EMBED_BATCH]
                sub_vecs = self.embed_service.embed(sub)
                all_vecs.extend(sub_vecs)

            # 写入 ChromaDB
            self.collection.upsert(
                ids=batch_ids,
                embeddings=all_vecs,
                metadatas=batch_metas,
                documents=batch_texts,
            )
            print(f"  [VECTOR] 进度: {end}/{total} ({(end/total*100):.0f}%)")

        print(f"[VECTOR] 已存入 {total} 个 chunks")

    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        向量相似度搜索，返回前 top_k 个结果
        """
        if not query_embedding or top_k <= 0:
            return []
        total = self.count()
        if total <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, top_k), total),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieved.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "tag": results["metadatas"][0][i].get("doc_name", "")
                             + f" chunk{results['metadatas'][0][i].get('chunk_index', '?')}",
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # 余弦距离转相似度
                })
        return retrieved

    def count(self) -> int:
        """返回库中 chunk 总数"""
        return self.collection.count()

    def clear(self):
        """清空 collection"""
        try:
            self.client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()
        print("[VECTOR] 已清空所有数据")


if __name__ == "__main__":
    from chunker import chunk_all_documents

    emb = EmbeddingService(mock=True)
    vs = VectorStore(embed_service=emb)

    print("切分文档...")
    chunks = chunk_all_documents()

    print("存入向量库...")
    vs.add_chunks(chunks)

    print(f"向量库总计: {vs.count()} chunks")
