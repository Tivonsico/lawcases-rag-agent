"""Protocol-aware offline evaluation using the production retrieval path."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable, Mapping, Sequence

try:
    from .config import FINAL_TOP_K
    from .evaluation_data import APPROVED_STATUS, canonical_jsonl, load_jsonl, validate_records
    from .intent import IntentProcessor
    from .retriever import HybridRetriever, RetrievalRequest
except ImportError:  # pragma: no cover - direct-script compatibility
    from config import FINAL_TOP_K
    from evaluation_data import APPROVED_STATUS, canonical_jsonl, load_jsonl, validate_records
    from intent import IntentProcessor
    from retriever import HybridRetriever, RetrievalRequest

EVALUATION_DIR = Path(__file__).resolve().parent / "data" / "evaluation"
RECALL_KS = (5, 20, 50)
RANKING_KS = (5, 10)
METRIC_LABELS = {
    "recall": "召回率", "precision": "准确率", "hitrate": "命中率",
    "map": "平均准确率", "ndcg": "归一化折损累计增益", "mrr": "倒数排名",
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _metric_label(key: str) -> str:
    prefix, _, suffix = key.partition("@")
    return f"{METRIC_LABELS.get(prefix.removeprefix('target_').removeprefix('pooled_').removeprefix('graded_'), prefix)}@{suffix}" if suffix else METRIC_LABELS.get(prefix, prefix)


def _print_report(report: Mapping) -> None:
    print("\n===== 检索评测结果 =====")
    print(f"数据集: {report['split']}（{report['sample_count']} 条）")
    print(f"检索模式: {report['variant']['name']}")
    for key, value in report["metrics"].items():
        print(f"{_metric_label(key)}: {value:.2%}")
    latency = report["latency_ms"]
    print(f"延迟: 平均 {latency['mean']:.1f}ms / P50 {latency['p50']:.1f}ms / P95 {latency['p95']:.1f}ms")
    if report["channel_errors"]:
        print(f"失败通道: {', '.join(report['channel_errors'])}")
    print(f"说明: {report['warning']}")


def load_split(split: str = "20", directory: Path = EVALUATION_DIR) -> tuple[list[dict], str]:
    """Load the evaluation dataset. Only '20' is available."""
    if split != "20":
        raise ValueError("only split '20' is available")
    path = directory / "evaluation_20.jsonl"
    raw = path.read_bytes()
    cases = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return cases, hashlib.sha256(raw).hexdigest()


def load_published(path: str | Path, *, protocol: str) -> tuple[list[dict], str]:
    rows, digest = load_jsonl(path)
    _require_protocol(rows, protocol)
    return rows, digest


def load_variants(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(variants, list) or not variants:
        raise ValueError("variant config requires a non-empty variants list")
    allowed, names = set(HybridRetriever.CHANNELS), set()
    for index, variant in enumerate(variants, 1):
        name, channels = variant.get("name"), variant.get("channels")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"variant {index} has invalid or duplicate name")
        if not isinstance(channels, list) or not channels or not set(channels) <= allowed:
            raise ValueError(f"variant {name} has invalid channels")
        if set(variant.get("weights", {})) - set(channels):
            raise ValueError(f"variant {name} weights include disabled channels")
        names.add(name)
    return [dict(variant) for variant in variants]


def evaluate_variants(retriever_factory, cases, *, protocol: str, variants, intent_processor=None):
    reports = []
    for variant in variants:
        retriever = retriever_factory(variant)
        if variant.get("requires_reranker") and not getattr(retriever, "reranker", None):
            raise ValueError(f"variant {variant['name']} requires a reranker")
        reports.append(
            evaluate(retriever, intent_processor, cases, protocol=protocol, variant=variant, verbose=False)
        )
    return reports


def _request(query: str, intent_processor, max_k: int, channels=None) -> RetrievalRequest:
    intent = intent_processor.process(query) if intent_processor else {}
    return RetrievalRequest(
        query=query,
        normalized_query=intent.get("normalized_query", ""),
        exact_terms=tuple(intent.get("exact_terms", ())),
        document_k=max_k,
        final_k=max(FINAL_TOP_K, max(RANKING_KS)),
        channels=tuple(channels) if channels else None,
    )


def _case_id(row: Mapping) -> str:
    return str(row.get("case_id") or row.get("metadata", {}).get("case_id") or "")


def _dedupe(rows: Iterable[Mapping]) -> list[dict]:
    seen, result = set(), []
    for row in rows:
        case_id = _case_id(row)
        if case_id and case_id not in seen:
            seen.add(case_id)
            result.append(dict(row))
    return result


def _average_precision(ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, case_id in enumerate(ids[:k], 1):
        if case_id in relevant:
            hits += 1
            total += hits / rank
    return _clamp(total / min(len(relevant), k))


def _binary_ndcg(ids: Sequence[str], relevant: set[str], k: int) -> float:
    grades = {case_id: 1 for case_id in relevant}
    return _graded_ndcg(ids, grades, k)


def _graded_ndcg(ids: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    dcg = sum(
        (2 ** grades.get(case_id, 0) - 1) / math.log2(rank + 1)
        for rank, case_id in enumerate(ids[:k], 1)
    )
    ideal_grades = sorted((grade for grade in grades.values() if grade > 0), reverse=True)[:k]
    ideal = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    return _clamp(dcg / ideal) if ideal else 0.0


def _require_protocol(rows: Sequence[Mapping], protocol: str) -> list[dict]:
    if protocol not in {"core", "target"}:
        raise ValueError("protocol must be core or target")
    validated = validate_records(rows, source=f"<{protocol}>")
    for index, row in enumerate(validated, 1):
        if row["evaluation_type"] != protocol:
            raise ValueError(f"row {index} is {row['evaluation_type']}, expected {protocol}")
        if row["review_status"] != APPROVED_STATUS[protocol]:
            raise ValueError(f"row {index} is not approved for {protocol}")
    return validated


def _protocol_metrics(protocol: str, candidate_ids: list[str], ranked_ids: list[str], case: Mapping) -> dict:
    relevant = set(case.get("relevant_case_ids", ()))
    first = next((rank for rank, cid in enumerate(ranked_ids[:10], 1) if cid in relevant), 0)
    if protocol == "target":
        metrics = {
            f"target_recall@{k}": _clamp(len(relevant.intersection(candidate_ids[:k])) / len(relevant))
            for k in RECALL_KS
        }
        metrics.update(
            {f"target_hitrate@{k}": float(bool(relevant.intersection(ranked_ids[:k]))) for k in RANKING_KS}
        )
        metrics["target_mrr@10"] = 1.0 / first if first else 0.0
        return metrics

    grades = {item["case_id"]: item["relevance"] for item in case["judgments"]}
    metrics = {
        f"pooled_recall@{k}": _clamp(len(relevant.intersection(candidate_ids[:k])) / len(relevant))
        for k in RECALL_KS
    }
    for k in RANKING_KS:
        top = ranked_ids[:k]
        hits = len(relevant.intersection(top))
        metrics[f"precision@{k}"] = _clamp(hits / k)
        metrics[f"hitrate@{k}"] = float(hits > 0)
        metrics[f"map@{k}"] = _average_precision(ranked_ids, relevant, k)
        metrics[f"graded_ndcg@{k}"] = _graded_ndcg(ranked_ids, grades, k)
        metrics[f"judgment_coverage@{k}"] = _clamp(sum(cid in grades for cid in top) / len(top)) if top else 0.0
    metrics["mrr@10"] = 1.0 / first if first else 0.0
    return metrics


def _legacy_metrics(candidate_ids: list[str], ranked_ids: list[str], relevant: set[str]) -> dict:
    metrics = {
        f"recall@{k}": _clamp(len(relevant.intersection(candidate_ids[:k])) / len(relevant)) if relevant else 0.0
        for k in RECALL_KS
    }
    for k in RANKING_KS:
        hits = len(relevant.intersection(ranked_ids[:k]))
        metrics.update(
            {
                f"precision@{k}": _clamp(hits / k),
                f"hitrate@{k}": float(hits > 0),
                f"map@{k}": _average_precision(ranked_ids, relevant, k),
                f"ndcg@{k}": _binary_ndcg(ranked_ids, relevant, k),
            }
        )
    first = next((rank for rank, cid in enumerate(ranked_ids[:10], 1) if cid in relevant), 0)
    metrics["mrr@10"] = 1.0 / first if first else 0.0
    return metrics


def evaluate(
    retriever: HybridRetriever,
    intent_processor=None,
    test_cases=None,
    *,
    split: str = "20",
    evaluation_dir: Path = EVALUATION_DIR,
    protocol: str | None = None,
    variant: Mapping | None = None,
    strict_channels: bool = False,
    verbose: bool = True,
) -> dict:
    """Evaluate legacy, core, or target data without mixing metric semantics."""
    if test_cases is None:
        cases, split_hash = load_split(split, evaluation_dir)
    else:
        cases = [dict(item) for item in test_cases]
        split_hash = hashlib.sha256(canonical_jsonl(cases)).hexdigest() if protocol else "ad-hoc"
    if protocol:
        cases = _require_protocol(cases, protocol)

    variant = dict(variant or {})
    channels = variant.get("channels")
    details, latencies, channel_errors = [], [], {}
    total_cases = len(cases)
    for idx, case in enumerate(cases, 1):
        if verbose:
            print(f"  [{idx}/{total_cases}] {case.get('id', case['query'][:30])}...", end=" ", flush=True)
        started = perf_counter()
        result = retriever.search(_request(case["query"], intent_processor, max(RECALL_KS), channels))
        latency_ms = (perf_counter() - started) * 1000
        if verbose:
            print(f"{latency_ms:.0f}ms")
        latencies.append(latency_ms)
        candidates, ranked = _dedupe(result.candidates), _dedupe(result.documents)
        candidate_ids = [_case_id(row) for row in candidates]
        ranked_ids = [_case_id(row) for row in ranked]
        if protocol:
            metrics = _protocol_metrics(protocol, candidate_ids, ranked_ids, case)
        else:
            metrics = _legacy_metrics(candidate_ids, ranked_ids, set(case.get("relevant_case_ids", ())))
        for channel in result.errors:
            channel_errors[channel] = channel_errors.get(channel, 0) + 1
        if strict_channels and result.errors:
            raise RuntimeError(f"以下检索通道失败，完整混合检索评测已中止：{', '.join(result.errors)}")
        details.append(
            {
                "id": case.get("id"),
                "query": case["query"],
                "candidate_case_ids": candidate_ids,
                "ranked_case_ids": ranked_ids,
                "latency_ms": round(latency_ms, 3),
                "errors": dict(result.errors),
                "metrics": metrics,
            }
        )

    metric_keys = list(details[0]["metrics"]) if details else []
    metrics = {
        key: round(_clamp(mean(row["metrics"][key] for row in details)), 4)
        for key in metric_keys
    }
    ordered = sorted(latencies)
    percentile = lambda p: ordered[min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1)] if ordered else 0.0
    variant_info = {
        "name": variant.get("name", "default"),
        "channels": list(channels) if channels else list(getattr(retriever, "CHANNELS", ())),
        "rrf_k": getattr(retriever, "rrf_k", None),
        "weights": dict(getattr(retriever, "channel_weights", {})),
        "reranker": bool(getattr(retriever, "reranker", None)),
    }
    report = {
        "split": split,
        "protocol": protocol or "legacy",
        "dataset_sha256": split_hash,
        "split_sha256": split_hash,
        "sample_count": len(details),
        "exhaustive_judgments": False,
        "variant": variant_info,
        "metrics": metrics,
        "channel_errors": channel_errors,
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(0.50), 3),
            "p95": round(percentile(0.95), 3),
        },
        "warning": (
            "探索性结果：legacy judgments are non-exhaustive."
            if protocol is None
            else "Core metrics are pool-relative, not exhaustive over the whole corpus."
            if protocol == "core"
            else "Target metrics measure known-item retrieval only; ranking-quality metrics are intentionally omitted."
        ),
        "cases": details,
    }
    if channel_errors:
        report["warning"] += " 部分检索通道失败，本次分数仅反映仍成功的通道。"
    if verbose:
        _print_report(report)
    return report


# ── 消融测试固定方案 ──
ABLATION_VARIANTS = [
    {"name": "emb", "channels": ["embedding_original"],
     "weights": {"embedding_original": 1.0}, "rrf_k": 60},
    {"name": "bm25", "channels": ["bm25_original"],
     "weights": {"bm25_original": 1.0}, "rrf_k": 60},
    {"name": "hybrid", "channels": ["embedding_original", "bm25_original"],
     "weights": {"embedding_original": 1.0, "bm25_original": 1.0}, "rrf_k": 60},
    {"name": "hybrid+jieba", "channels": ["embedding_original", "embedding_normalized", "bm25_original"],
     "weights": {"embedding_original": 1.0, "embedding_normalized": 0.8, "bm25_original": 1.0}, "rrf_k": 60},
    {"name": "plus+案号", "channels": ["embedding_original", "embedding_normalized", "bm25_original", "exact"],
     "rrf_k": 60},  # 无 weights → 继承 retriever.channel_weights（即 config.py RRF_WEIGHTS）
]


def _run_ablation(retriever: HybridRetriever, intent_processor=None) -> list[dict] | None:
    """Run 5-variant ablation comparison without loading external config."""
    def factory(variant):
        return HybridRetriever(
            vector_store=retriever.vector_store,
            embed_service=retriever.embed_service,
            bm25_index=retriever.bm25,
            channel_weights=variant.get("weights", retriever.channel_weights),
            rrf_k=variant.get("rrf_k", retriever.rrf_k),
        )

    cases, _ = load_split("20")
    reports = evaluate_variants(factory, cases, protocol=None, variants=ABLATION_VARIANTS, intent_processor=intent_processor)
    reports = [r for r in reports if r is not None]

    print()
    print("=" * 105)
    print("  消融测试（5 种方案对比，20 条评测数据）")
    print("=" * 105)
    print(f"{'方案':>15s}  {'召回率@5':>8s}  {'召回率@20':>8s}  {'召回率@50':>8s}  {'准确率@5':>8s}  {'MAP@5':>8s}  {'NDCG@5':>8s}  {'MRR@10':>8s}  {'延迟(ms)':>8s}")
    print("-" * 105)
    for report in reports:
        m = report["metrics"]
        name = report["variant"]["name"]
        lat = report["latency_ms"]
        print(f"{name:>15s}  {m.get('recall@5', 0):>7.1%}  {m.get('recall@20', 0):>7.1%}  {m.get('recall@50', 0):>7.1%}  {m.get('precision@5', 0):>7.1%}  {m.get('map@5', 0):>7.1%}  {m.get('ndcg@5', 0):>7.1%}  {m.get('mrr@10', 0):>7.1%}  {lat['mean']:>7.0f}")
        if "channel_errors" in report and report["channel_errors"]:
            print(f"{'':>15s}  ⚠ 失败通道: {', '.join(report['channel_errors'])}")
    print()
    return reports


def run_test_mode(retriever: HybridRetriever, intent_processor=None):
    """Run full evaluation or ablation comparison."""
    mode = input("评测模式 [full(全量) / ablation(消融对比)，默认 full]: ").strip().lower() or "full"
    if mode == "ablation":
        return _run_ablation(retriever, intent_processor)
    try:
        return evaluate(
            retriever,
            intent_processor or IntentProcessor(),
            split="20",
            variant={"name": "hybrid", "channels": ["embedding_original", "bm25_original", "exact"]},
            strict_channels=True,
        )
    except RuntimeError as exc:
        print(f"\n评测未完成：{exc}")
        return None


if __name__ == "__main__":
    raise SystemExit("请通过 rag_agent/main.py 的 /test 入口运行，以复用线上检索器。")
