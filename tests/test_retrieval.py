import sys
from pathlib import Path

import requests
sys.path.insert(0, str(Path(__file__).parents[1] / "rag_agent"))

from embedding import EmbeddingService
from retriever import BM25Index, HybridRetriever, RetrievalRequest, weighted_rrf
from vector_store import VectorStore


class FakeEmbed:
    def embed_one(self, text):
        return [float(len(text))]


class FakeVectors:
    def __init__(self):
        self.queries = []

    def search(self, text, vector, top_k):
        self.queries.append((text, top_k))
        key = {"原始": "a", "规范": "b", "假设": "c"}.get(text)
        return [] if not key else [{"id": key, "text": key, "metadata": {"case_id": "case:" + key, "doc_name": key, "chunk_index": 0}}]


def make_chunks():
    return [
        {
            "chunk_id": "a",
            "case_id": "case:a",
            "doc_name": "指导案例192号(A)",
            "chunk_index": 0,
            "text": "侵权争议",
            "tag": "a",
            "keywords": [],
            "case_number": "指导案例192号",
        },
        {
            "chunk_id": "d",
            "case_id": "case:d",
            "doc_name": "劳动案(D)",
            "chunk_index": 0,
            "text": "劳动合同解除",
            "tag": "d",
            "keywords": [],
            "case_number": "unknown",
        },
    ]


def test_rrf_missing_channel_gets_no_score():
    rows = {x["id"]: x for x in weighted_rrf({"x": [{"id": "a"}], "y": [{"id": "b"}]}, {"x": 2, "y": 1}, 60)}
    assert rows["a"]["rrf_score"] == 2 / 61
    assert rows["a"]["channel_ranks"] == {"x": 1}


def test_six_channels_contract_and_pool_sizes():
    bm25 = BM25Index()
    bm25.build(make_chunks())
    vectors = FakeVectors()
    request = RetrievalRequest("原始", "规范", "假设", ("劳动合同",), ("指导案例192号",), 123, 111, 40, 6)
    result = HybridRetriever(vectors, FakeEmbed(), bm25).search(request)
    assert set(result.channel_hits) == set(HybridRetriever.CHANNELS)
    assert vectors.queries == [("原始", 123), ("规范", 123), ("假设", 123)]
    assert result.channel_hits["exact"][0]["id"] == "a"
    assert result.documents[0]["case_id"] == "case:a"


def test_failures_fall_back_and_legacy_is_list_compatible():
    class Broken(FakeVectors):
        def search(self, *args):
            raise RuntimeError("offline")

    bm25 = BM25Index()
    bm25.build(make_chunks())

    def reranker(*args):
        raise RuntimeError("no model")

    result = HybridRetriever(Broken(), FakeEmbed(), bm25, reranker).search(
        RetrievalRequest("劳动合同", exact_terms=("指导案例192号",))
    )
    assert result.documents and "embedding_original" in result.errors and "reranker" in result.errors

    legacy = HybridRetriever(FakeVectors(), FakeEmbed(), bm25).search("原始", ["劳动合同"], top_k=1)
    assert len(legacy) == 1 and legacy[0] is legacy.documents[0]


def test_empty_vector_store_returns_empty_hits(tmp_path):
    store = VectorStore(embed_service=EmbeddingService(mock=True), persist_dir=str(tmp_path / "chroma"))
    store.clear()
    assert store.search("正当防卫", [1.0], top_k=5) == []


def test_embedding_retries_once_after_connect_timeout(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2]}]}

    calls = []

    def post(*_, **__):
        calls.append(True)
        if len(calls) == 1:
            raise requests.ConnectTimeout("temporary timeout")
        return Response()

    monkeypatch.setattr("embedding.requests.post", post)
    monkeypatch.setattr("embedding.time.sleep", lambda _: None)
    assert EmbeddingService(api_url="https://example.test", api_key="test", dim=2).embed_one("测试") == [0.1, 0.2]
    assert len(calls) == 2
