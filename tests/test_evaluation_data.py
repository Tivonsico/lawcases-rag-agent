import json

import pytest

from rag_agent.evaluation_data import (
    EvaluationDataError,
    build_review_drafts,
    canonical_jsonl,
    deterministic_sample,
    load_jsonl,
    manifest_sha256,
    publish_records,
    validate_records,
)
from rag_agent.evaluation_generator import (
    build_prompt,
    generate_target_rows,
    parse_generated_query,
)


def target(**overrides):
    row = {
        "schema_version": 1,
        "id": "target-001",
        "query": "发生类似情况应当如何处理？",
        "evaluation_type": "target",
        "relevant_case_ids": ["case:a"],
        "judgments_exhaustive": False,
        "review_status": "generated",
    }
    row.update(overrides)
    return row


def core(**overrides):
    row = {
        "schema_version": 1,
        "id": "core-001",
        "query": "正当防卫的判断边界是什么？",
        "evaluation_type": "core",
        "relevant_case_ids": ["case:a"],
        "judgments": [
            {"case_id": "case:a", "relevance": 2},
            {"case_id": "case:b", "relevance": 0},
        ],
        "judgments_exhaustive": False,
        "review_status": "generated",
    }
    row.update(overrides)
    return row


def test_generated_record_is_valid_but_cannot_be_published(tmp_path):
    assert validate_records([target()])[0]["review_status"] == "generated"
    with pytest.raises(EvaluationDataError, match="requires exactly 1 approved target.*got 0"):
        publish_records([target()], tmp_path / "target.jsonl", evaluation_type="target", expected_count=1)
    assert not (tmp_path / "target.jsonl").exists()


def test_approved_target_requires_audit_fields_and_publishes_atomically(tmp_path):
    approved = target(
        review_status="human_verified_target",
        reviewer="reviewer-1",
        reviewed_at="2026-07-13T00:00:00Z",
        target_leakage_checked=True,
    )
    path = tmp_path / "target.jsonl"
    digest = publish_records([approved], path, evaluation_type="target", expected_count=1)
    loaded, loaded_digest = load_jsonl(path, known_case_ids={"case:a"})
    assert loaded == [approved]
    assert loaded_digest == digest
    assert path.read_bytes() == canonical_jsonl([approved])


@pytest.mark.parametrize("missing", ["reviewer", "reviewed_at", "target_leakage_checked"])
def test_approved_target_fails_closed_when_audit_field_is_missing(missing):
    row = target(
        review_status="human_verified_target",
        reviewer="reviewer-1",
        reviewed_at="2026-07-13T00:00:00Z",
        target_leakage_checked=True,
    )
    row.pop(missing)
    with pytest.raises(EvaluationDataError, match=missing):
        validate_records([row])


def test_core_positive_judgments_must_match_relevant_ids():
    with pytest.raises(EvaluationDataError, match="positive judgments must equal"):
        validate_records([core(relevant_case_ids=["case:b"])])


def test_target_rejects_multiple_targets_and_graded_judgments():
    with pytest.raises(EvaluationDataError, match="exactly one"):
        validate_records([target(relevant_case_ids=["case:a", "case:b"])])
    with pytest.raises(EvaluationDataError, match="must not contain graded"):
        validate_records([target(judgments=[{"case_id": "case:a", "relevance": 2}])])


def test_duplicate_and_unknown_case_ids_fail_with_source_line():
    rows = [target(), target(id="target-001", relevant_case_ids=["case:missing"])]
    with pytest.raises(EvaluationDataError, match=r"fixture.jsonl:2: unknown case ids: case:missing"):
        validate_records(rows, source="fixture.jsonl", known_case_ids={"case:a"})
    with pytest.raises(EvaluationDataError, match=r"fixture.jsonl:2: duplicate id"):
        validate_records([target(), target()], source="fixture.jsonl")


def test_load_jsonl_reports_path_and_physical_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationDataError, match=r"bad.jsonl:2: id must"):
        load_jsonl(path)


def test_canonical_jsonl_is_stable_across_input_order():
    first = target(id="target-001")
    second = target(id="target-002", relevant_case_ids=["case:b"])
    assert canonical_jsonl([second, first]) == canonical_jsonl([first, second])


def cases():
    return [
        {
            "case_id": f"case:{family}:{i}",
            "case_number": f"{family}{i}号",
            "source_family": family,
            "title": f"案例{i}",
            "keywords": ["合同", "履行"],
        }
        for family in ("最高法", "最高检")
        for i in range(8)
    ]


def test_sampling_is_seeded_quota_preserving_disjoint_and_excludes_cases():
    kwargs = {
        "core_quotas": {"最高法": 2, "最高检": 2},
        "target_quotas": {"最高法": 2, "最高检": 2},
        "excluded_case_ids": {"case:最高法:0"},
    }
    core_a, target_a = deterministic_sample(list(reversed(cases())), seed=7, **kwargs)
    core_b, target_b = deterministic_sample(cases(), seed=7, **kwargs)
    assert core_a == core_b and target_a == target_b
    assert {x["case_id"] for x in core_a}.isdisjoint(x["case_id"] for x in target_a)
    assert "case:最高法:0" not in {x["case_id"] for x in core_a + target_a}
    assert {x["source_family"] for x in core_a} == {"最高法", "最高检"}
    assert deterministic_sample(cases(), seed=8, **kwargs) != (core_a, target_a)


def test_review_drafts_are_valid_but_never_approved():
    core_cases, target_cases = deterministic_sample(
        cases(),
        seed=3,
        core_quotas={"最高法": 1, "最高检": 1},
        target_quotas={"最高法": 1, "最高检": 1},
    )
    digest = manifest_sha256(cases())
    core_rows, target_rows = build_review_drafts(
        core_cases, target_cases, corpus_sha256=digest
    )
    assert all(row["review_status"] == "generated" for row in core_rows + target_rows)
    assert all(row["metadata"]["corpus_sha256"] == digest for row in core_rows + target_rows)
    assert all(row["metadata"]["draft_purpose"] == "pooling_seed_only" for row in core_rows)


def generator_case():
    return {
        "case_id": "case:a",
        "case_number": "指导案例1号",
        "title": "张某诉李某合同纠纷案",
        "keywords": ["合同解除", "违约责任"],
        "summary": "当事人根本违约时，守约方可以依法主张解除合同。",
    }


def test_generation_prompt_uses_evidence_but_not_identity_fields():
    prompt = build_prompt(generator_case(), "scenario")
    assert "合同解除" in prompt and "根本违约" in prompt
    assert "指导案例1号" not in prompt and "张某诉李某" not in prompt


def test_generated_target_advances_only_to_machine_checked():
    row = target(metadata={"query_type": "scenario"})
    result, stats = generate_target_rows(
        [row],
        {"case:a": generator_case()},
        lambda _prompt: '{"query":"合同一方严重违约时，另一方通常可以怎样维护权益？"}',
        model="fake-model",
        max_records=1,
    )
    assert stats == {"generated": 1, "machine_checked": 1, "needs_revision": 0}
    assert result[0]["review_status"] == "machine_checked"
    assert result[0]["target_leakage_checked"] is False
    assert result[0]["metadata"]["generator"]["prompt_version"] == "target-query-v1"


def test_leaking_generation_requires_revision_and_never_approves():
    row = target(metadata={"query_type": "scenario"})
    result, stats = generate_target_rows(
        [row],
        {"case:a": generator_case()},
        lambda _prompt: '{"query":"请问指导案例1号主要说明了什么法律规则？"}',
        model="fake-model",
        max_records=1,
    )
    assert stats["needs_revision"] == 1
    assert result[0]["review_status"] == "needs_revision"
    assert result[0]["metadata"]["leakage_flags"] == ["case_number"]


def test_generator_rejects_bad_json_with_record_id_and_respects_limit():
    rows = [target(id="target-001"), target(id="target-002")]
    with pytest.raises(EvaluationDataError, match="target-001: generator returned invalid JSON"):
        generate_target_rows(
            rows,
            {"case:a": generator_case()},
            lambda _prompt: "not-json",
            model="fake-model",
            max_records=1,
        )
    assert parse_generated_query('```json\n{"query":"这是一个长度足够的法律问题吗？"}\n```')
