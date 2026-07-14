"""Unified six-channel retrieval and standard weighted RRF."""
from dataclasses import dataclass, field
import logging
import re
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from rank_bm25 import BM25Okapi
try:
    from .config import (BM25_CANDIDATE_K, DOCUMENT_CANDIDATE_K, FINAL_TOP_K,
                         RRF_K, RRF_WEIGHTS, VECTOR_CANDIDATE_K)
    from .embedding import EmbeddingService
    from .vector_store import VectorStore
except ImportError:  # pragma: no cover - direct script compatibility
    from config import (BM25_CANDIDATE_K, DOCUMENT_CANDIDATE_K, FINAL_TOP_K,
                        RRF_K, RRF_WEIGHTS, VECTOR_CANDIDATE_K)
    from embedding import EmbeddingService
    from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    normalized_query: str = ""
    exact_terms: Sequence[str] = field(default_factory=tuple)
    vector_candidate_k: int = VECTOR_CANDIDATE_K
    bm25_candidate_k: int = BM25_CANDIDATE_K
    document_k: int = DOCUMENT_CANDIDATE_K
    final_k: int = FINAL_TOP_K
    channels: Optional[Sequence[str]] = None


@dataclass
class RetrievalResult:
    request: RetrievalRequest
    documents: List[Dict]
    candidates: List[Dict]
    channel_hits: Dict[str, List[Dict]]
    errors: Dict[str, str] = field(default_factory=dict)
    def __iter__(self) -> Iterator[Dict]: return iter(self.documents)
    def __len__(self): return len(self.documents)
    def __getitem__(self, item): return self.documents[item]


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.documents, self.records = [], []

    def build(self, chunks: List[Dict]):
        self.documents, self.records = [], []
        for c in chunks:
            metadata = {k: v for k, v in c.items() if k not in {"text", "tag", "keywords"}}
            metadata.setdefault("doc_name", c.get("doc_name", ""))
            self.records.append({"id": c.get("chunk_id") or f"{c['doc_name']}_chunk{c['chunk_index']}",
                "tag": c.get("tag", ""), "text": c["text"], "doc_name": c.get("doc_name", ""),
                "metadata": metadata})
            self.documents.append(self._tokenize(c["text"] + " " + " ".join(c.get("keywords", []))))
        self.bm25 = BM25Okapi(self.documents) if self.documents else None

    @staticmethod
    def _tokenize(text):
        return ([text[i:i+2] for i in range(len(text)-1)
                 if all("\u4e00" <= c <= "\u9fff" for c in text[i:i+2])] +
                [w.lower() for w in re.findall(r"[a-zA-Z0-9_-]+", text)])

    def search(self, query, top_k=BM25_CANDIDATE_K):
        if not self.bm25 or not query.strip(): return []
        scores = self.bm25.get_scores(self._tokenize(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self.records[i], "score": float(scores[i]), "rank": rank}
                for rank, i in enumerate(indices, 1) if scores[i] > 0]

    def exact_search(self, terms, top_k=BM25_CANDIDATE_K):
        terms = [re.sub(r"\s+", "", x).lower() for x in terms if x]
        hits = []
        for record in self.records:
            meta = record["metadata"]
            haystack = re.sub(r"\s+", "", " ".join(str(meta.get(k, ""))
                for k in ("case_number", "source_id", "doc_name"))).lower()
            score = sum(term in haystack for term in terms)
            if score: hits.append({**record, "score": float(score)})
        hits.sort(key=lambda x: (-x["score"], x["id"]))
        for rank, hit in enumerate(hits[:top_k], 1): hit["rank"] = rank
        return hits[:top_k]


def weighted_rrf(channel_hits: Mapping[str, Sequence[Dict]], weights: Mapping[str, float], k=60):
    """Only present results score: sum(weight / (k + rank))."""
    fused = {}
    for channel, hits in channel_hits.items():
        for rank, hit in enumerate(hits, 1):
            chunk_id = hit.get("id")
            if not chunk_id: continue
            row = fused.setdefault(chunk_id, {**hit, "rrf_score": 0.0, "channel_ranks": {}})
            row["rrf_score"] += float(weights.get(channel, 1.0)) / (k + rank)
            row["channel_ranks"][channel] = rank
    return sorted(fused.values(), key=lambda x: (-x["rrf_score"], x["id"]))


class HybridRetriever:
    CHANNELS = ("embedding_original", "embedding_normalized",
                "bm25_original", "exact")
    def __init__(self, vector_store: VectorStore, embed_service: EmbeddingService = None,
                 bm25_index: BM25Index = None, reranker: Optional[Callable] = None,
                 rrf_k=RRF_K, channel_weights=None):
        self.vector_store = vector_store
        self.embed_service = embed_service or EmbeddingService(mock=True)
        self.bm25, self.reranker, self.rrf_k = bm25_index, reranker, rrf_k
        self.channel_weights = dict(channel_weights or RRF_WEIGHTS)

    def set_bm25_index(self, bm25): self.bm25 = bm25
    def _vector(self, text, top_k):
        if not text.strip(): return []
        return self.vector_store.search(text, self.embed_service.embed_one(text), top_k)
    @staticmethod
    def _case_id(hit):
        meta = hit.get("metadata", {})
        return meta.get("case_id") or hit.get("case_id") or meta.get("doc_name") or hit.get("doc_name", "")

    def search(self, request=None, expanded_keywords=None, top_k=None, **kwargs):
        if isinstance(request, RetrievalRequest): req = request
        else:
            query = request if isinstance(request, str) else kwargs.pop("query", "")
            for dead in ("synonyms", "hyde_query"): kwargs.pop(dead, None)
            req = RetrievalRequest(query=query,
                                   final_k=top_k or kwargs.pop("final_k", FINAL_TOP_K), **kwargs)
        enabled, hits, errors = set(req.channels or self.CHANNELS), {}, {}
        calls = {
            "embedding_original": lambda: self._vector(req.query, req.vector_candidate_k),
            "embedding_normalized": lambda: self._vector(req.normalized_query, req.vector_candidate_k),
            "bm25_original": lambda: self.bm25.search(req.query, req.bm25_candidate_k) if self.bm25 else [],
            "exact": lambda: self.bm25.exact_search(req.exact_terms, req.bm25_candidate_k) if self.bm25 else []}
        for channel, call in calls.items():
            if channel not in enabled: continue
            try: hits[channel] = call()
            except Exception as exc:
                logger.warning("channel %s failed: %s", channel, exc)
                hits[channel], errors[channel] = [], str(exc)
        grouped = {}
        for chunk in weighted_rrf(hits, self.channel_weights, self.rrf_k):
            case_id = self._case_id(chunk)
            if not case_id: continue
            doc = grouped.setdefault(case_id, {"case_id": case_id,
                "doc_name": chunk.get("metadata", {}).get("doc_name") or chunk.get("doc_name", ""),
                "chunks": [], "best_score": 0.0})
            doc["chunks"].append(chunk); doc["best_score"] = max(doc["best_score"], chunk["rrf_score"])
        candidates = sorted(grouped.values(), key=lambda x: (-x["best_score"], x["case_id"]))[:req.document_k]
        for doc in candidates:
            doc["chunks"].sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
            doc["display_name"] = doc["doc_name"].split("(")[0].strip()
            doc["text"] = "\n\n".join(dict.fromkeys(c.get("text", "").strip() for c in doc["chunks"] if c.get("text", "").strip()))
        documents = candidates
        if self.reranker:
            try: documents = list(self.reranker(req.query, list(candidates)))
            except Exception as exc: errors["reranker"] = str(exc)
        return RetrievalResult(req, documents[:req.final_k], candidates, hits, errors)
