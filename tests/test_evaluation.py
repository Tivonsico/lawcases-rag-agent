import inspect
from pathlib import Path

import pytest

from rag_agent.retriever import RetrievalResult
from rag_agent.test_evaluator import evaluate, evaluate_variants, load_split, load_variants, run_test_mode


class FakeRetriever:
    CHANNELS = ("embedding_original", "bm25_original")

    def __init__(self, candidates, documents=None, errors=None):
        self.candidates = candidates
        self.documents = documents or candidates
        self.errors = errors or {}
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return RetrievalResult(request, self.documents, self.candidates, {}, self.errors)


def row(case_id):
    return {"case_id": case_id, "doc_name": case_id}


def approved(kind, **overrides):
    row_data = {
        "schema_version": 1,
        "id": f"{kind}-001",
        "query": "如何判断相关法律责任？",
        "evaluation_type": kind,
        "relevant_case_ids": ["case:a"],
        "judgments_exhaustive": False,
        "review_status": f"human_verified_{kind}",
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-07-13T00:00:00Z",
    }
    if kind == "core":
        row_data["judgments"] = [
            {"case_id": "case:a", "relevance": 2},
            {"case_id": "case:x", "relevance": 0},
        ]
    else:
        row_data["target_leakage_checked"] = True
    row_data.update(overrides)
    return row_data


def test_fixed_splits_are_disjoint_and_keep_all_eleven_annotations():
    rows, digest = load_split()
    assert len(rows) == 22
    assert len(digest) == 64


def test_default_20_split_contains_original_and_generated_rows():
    rows, digest = load_split("20")
    assert len(rows) == 22
    assert len(digest) == 64
    assert sum(row["origin"] == "original_11" for row in rows) == 11
    assert sum(row["origin"] == "auto_generated" for row in rows) == 9
    assert sum(row.get("origin") == "manual_case_number" for row in rows) == 2


def test_interactive_test_runs_default_20_case_hybrid_mode(monkeypatch):
    retriever = FakeRetriever([row("case:a")])
    monkeypatch.setattr("builtins.input", lambda _: "")
    report = run_test_mode(retriever, None)
    assert report["split"] == "20"
    assert report["sample_count"] == 22
    assert retriever.requests[0].channels is None


def test_strict_hybrid_evaluation_rejects_partial_channel_failure():
    with pytest.raises(RuntimeError, match="完整混合检索评测已中止"):
        evaluate(
            FakeRetriever([row("case:a")], errors={"embedding_original": "timeout"}),
            test_cases=[{"query": "q", "relevant_case_ids": ["case:a"]}],
            strict_channels=True,
            verbose=False,
        )


def test_metrics_dedupe_case_id_stay_bounded_and_use_unified_search():
    relevant = "case:a"
    candidates = [row(relevant), row(relevant)] + [row(f"case:{i}") for i in range(1, 100)]
    retriever = FakeRetriever(candidates, [row(relevant), row(relevant), row("case:x")])
    report = evaluate(retriever, test_cases=[{"query": "q", "relevant_case_ids": [relevant]}], verbose=False)
    assert len(retriever.requests) == 1
    assert report["metrics"]["recall@20"] == report["metrics"]["recall@50"] == 1.0
    assert report["metrics"]["mrr@10"] == 1.0
    assert all(0 <= value <= 1 for value in report["metrics"].values())
    assert report["cases"][0]["candidate_case_ids"].count(relevant) == 1


def test_evaluator_has_no_private_retrieval_fork():
    """Evaluator must consume the public retriever result, not silently rerun retrieval."""
    source = inspect.getsource(evaluate)
    assert "_search_top_k" not in source


def test_report_metadata_and_empty_relevance_are_safe(tmp_path: Path):
    report = evaluate(FakeRetriever([]), test_cases=[{"query": "none", "relevant_case_ids": []}], split="20", verbose=False)
    assert report["split"] == "20"
    assert report["sample_count"] == 1
    assert report["exhaustive_judgments"] is False
    assert "探索性" in report["warning"]
    assert all(value == 0 for value in report["metrics"].values())


def test_core_protocol_reports_graded_metrics_and_error_counts():
    retriever = FakeRetriever(
        [row("case:a"), row("case:x")],
        [row("case:a"), row("case:x")],
        errors={"embedding_hyde": "offline"},
    )
    report = evaluate(retriever, test_cases=[approved("core")], protocol="core", verbose=False)
    assert report["metrics"]["pooled_recall@20"] == 1.0
    assert report["metrics"]["graded_ndcg@5"] == 1.0
    assert report["metrics"]["judgment_coverage@5"] == 1.0
    assert report["channel_errors"] == {"embedding_hyde": 1}
    assert "部分检索通道失败" in report["warning"]
    assert "target_hitrate@5" not in report["metrics"]


def test_core_ndcg_rewards_putting_highly_relevant_case_first():
    case = approved(
        "core",
        relevant_case_ids=["case:a", "case:b"],
        judgments=[
            {"case_id": "case:a", "relevance": 2},
            {"case_id": "case:b", "relevance": 1},
        ],
    )
    ideal = evaluate(
        FakeRetriever([row("case:a"), row("case:b")]),
        test_cases=[case],
        protocol="core",
        verbose=False,
    )
    reversed_order = evaluate(
        FakeRetriever([row("case:b"), row("case:a")]),
        test_cases=[case],
        protocol="core",
        verbose=False,
    )
    assert ideal["metrics"]["graded_ndcg@5"] == 1.0
    assert reversed_order["metrics"]["graded_ndcg@5"] < 1.0


def test_target_protocol_uses_metric_whitelist_and_passes_variant_channels():
    retriever = FakeRetriever([row("case:a")], [row("case:a")])
    report = evaluate(
        retriever,
        test_cases=[approved("target")],
        protocol="target",
        variant={"name": "bm25_only", "channels": ["bm25_original"]},
        verbose=False,
    )
    assert report["metrics"] == {
        "target_recall@5": 1.0,
        "target_recall@20": 1.0,
        "target_recall@50": 1.0,
        "target_hitrate@5": 1.0,
        "target_hitrate@10": 1.0,
        "target_mrr@10": 1.0,
    }
    assert retriever.requests[0].channels == ("bm25_original",)
    assert report["variant"]["name"] == "bm25_only"
    assert not any("precision" in key or "ndcg" in key or "map" in key for key in report["metrics"])


def test_protocol_rejects_unapproved_or_mixed_records():
    draft = approved("target", review_status="machine_checked")
    with pytest.raises(ValueError, match="not approved"):
        evaluate(FakeRetriever([]), test_cases=[draft], protocol="target", verbose=False)
    with pytest.raises(ValueError, match="expected target"):
        evaluate(FakeRetriever([]), test_cases=[approved("core")], protocol="target", verbose=False)


def test_ablation_config_has_unique_incremental_variants():
    variants = load_variants(Path("rag_agent/data/evaluation/ablation_variants.json"))
    assert len(variants) == len({variant["name"] for variant in variants}) == 9
    assert variants[0]["channels"] == ["embedding_original"]
    assert variants[1]["channels"] == ["bm25_original"]
    assert len(variants[6]["channels"]) == 6
    assert variants[-1]["requires_reranker"] is True


def test_variant_runner_uses_factory_configuration_and_rejects_missing_reranker():
    variants = [{
        "name": "vector_only", "channels": ["embedding_original"],
        "weights": {"embedding_original": 1.0}, "rrf_k": 42,
        "requires_reranker": False,
    }]

    def factory(variant):
        retriever = FakeRetriever([row("case:a")])
        retriever.rrf_k = variant["rrf_k"]
        retriever.channel_weights = variant["weights"]
        retriever.reranker = None
        return retriever

    reports = evaluate_variants(factory, [approved("target")], protocol="target", variants=variants)
    assert reports[0]["variant"] == {
        "name": "vector_only", "channels": ["embedding_original"],
        "rrf_k": 42, "weights": {"embedding_original": 1.0}, "reranker": False,
    }
    reranked = [{**variants[0], "name": "reranked", "requires_reranker": True}]
    with pytest.raises(ValueError, match="requires a reranker"):
        evaluate_variants(factory, [approved("target")], protocol="target", variants=reranked)


def test_variant_loader_rejects_unknown_channel(tmp_path: Path):
    path = tmp_path / "variants.json"
    path.write_text(
        '{"variants":[{"name":"bad","channels":["imaginary"],"weights":{}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid channels"):
        load_variants(path)
