"""Validated, auditable datasets for offline retrieval evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
EVALUATION_TYPES = {"core", "target"}
REVIEW_STATUSES = {
    "generated",
    "machine_checked",
    "needs_revision",
    "rejected",
    "human_verified_core",
    "human_verified_target",
}
APPROVED_STATUS = {
    "core": "human_verified_core",
    "target": "human_verified_target",
}


class EvaluationDataError(ValueError):
    """Raised when evaluation data would make a report misleading."""


def _fail(source: str, message: str) -> None:
    raise EvaluationDataError(f"{source}: {message}")


def _string_list(value, source: str, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        _fail(source, f"{field} must be a non-empty string list")
    if len(value) != len(set(value)):
        _fail(source, f"{field} contains duplicates")
    return value


def validate_record(
    record: Mapping,
    *,
    source: str = "<record>",
    known_case_ids: set[str] | None = None,
) -> dict:
    """Validate one record without promoting its review status."""
    if not isinstance(record, Mapping):
        _fail(source, "record must be an object")
    row = dict(record)
    if row.get("schema_version") != SCHEMA_VERSION:
        _fail(source, f"schema_version must be {SCHEMA_VERSION}")
    for field in ("id", "query"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            _fail(source, f"{field} must be a non-empty string")
    kind = row.get("evaluation_type")
    if kind not in EVALUATION_TYPES:
        _fail(source, "evaluation_type must be core or target")
    status = row.get("review_status")
    if status not in REVIEW_STATUSES:
        _fail(source, f"invalid review_status: {status!r}")
    relevant = _string_list(row.get("relevant_case_ids"), source, "relevant_case_ids")
    if known_case_ids is not None:
        missing = sorted(set(relevant) - known_case_ids)
        if missing:
            _fail(source, f"unknown case ids: {', '.join(missing)}")

    exhaustive = row.get("judgments_exhaustive")
    if not isinstance(exhaustive, bool):
        _fail(source, "judgments_exhaustive must be boolean")
    judgments = row.get("judgments", [])
    if kind == "target":
        if len(relevant) != 1:
            _fail(source, "target records require exactly one target case")
        if judgments:
            _fail(source, "target records must not contain graded judgments")
        if exhaustive:
            _fail(source, "target judgments are never exhaustive")
    else:
        if not isinstance(judgments, list) or not judgments:
            _fail(source, "core records require graded judgments")
        grades: dict[str, int] = {}
        for index, judgment in enumerate(judgments, 1):
            if not isinstance(judgment, Mapping):
                _fail(source, f"judgments[{index}] must be an object")
            case_id, relevance = judgment.get("case_id"), judgment.get("relevance")
            if not isinstance(case_id, str) or not case_id:
                _fail(source, f"judgments[{index}].case_id is required")
            if case_id in grades:
                _fail(source, f"duplicate judgment for {case_id}")
            if relevance not in {0, 1, 2}:
                _fail(source, f"judgments[{index}].relevance must be 0, 1, or 2")
            if known_case_ids is not None and case_id not in known_case_ids:
                _fail(source, f"unknown judgment case id: {case_id}")
            grades[case_id] = relevance
        positives = {case_id for case_id, grade in grades.items() if grade > 0}
        if positives != set(relevant):
            _fail(source, "positive judgments must equal relevant_case_ids")

    if status == APPROVED_STATUS[kind]:
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            _fail(source, "approved records require reviewer")
        if not isinstance(row.get("reviewed_at"), str) or not row["reviewed_at"].strip():
            _fail(source, "approved records require reviewed_at")
        if kind == "target" and row.get("target_leakage_checked") is not True:
            _fail(source, "approved target records require target_leakage_checked=true")
    elif status.startswith("human_verified_"):
        _fail(source, f"review_status {status} does not match evaluation_type {kind}")
    return row


def validate_records(
    records: Iterable[Mapping],
    *,
    source: str = "<records>",
    known_case_ids: set[str] | None = None,
) -> list[dict]:
    rows, seen_ids = [], set()
    for line, record in enumerate(records, 1):
        row = validate_record(record, source=f"{source}:{line}", known_case_ids=known_case_ids)
        if row["id"] in seen_ids:
            _fail(f"{source}:{line}", f"duplicate id: {row['id']}")
        seen_ids.add(row["id"])
        rows.append(row)
    return rows


def load_jsonl(path: str | Path, *, known_case_ids: set[str] | None = None) -> tuple[list[dict], str]:
    target = Path(path)
    raw = target.read_bytes()
    rows, seen_ids = [], set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"{target}:{line_number}", f"invalid JSON: {exc.msg}")
        row = validate_record(
            parsed,
            source=f"{target}:{line_number}",
            known_case_ids=known_case_ids,
        )
        if row["id"] in seen_ids:
            _fail(f"{target}:{line_number}", f"duplicate id: {row['id']}")
        seen_ids.add(row["id"])
        rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()


def canonical_jsonl(records: Sequence[Mapping]) -> bytes:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(records, key=lambda item: item["id"])
    )
    return text.encode("utf-8")


def _first_section(text: str, heading: str) -> str:
    match = re.search(
        rf"【{re.escape(heading)}】\s*(.*?)(?=\n\s*【|\Z)",
        text,
        flags=re.DOTALL,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _case_identity(filename: str) -> tuple[str, str, str]:
    source = re.search(r"\((FBM-[^)]+)\)", Path(filename).stem, flags=re.IGNORECASE)
    number = re.search(r"((?:指导性?案例|检例第)\s*\d+\s*号)", filename)
    if not source or not number:
        raise EvaluationDataError(f"cannot extract stable case identity from {filename}")
    case_number = re.sub(r"\s+", "", number.group(1))
    family = "最高检" if case_number.startswith("检例") else "最高法"
    return f"case:{source.group(1).lower()}", case_number, family


def build_corpus_manifest(doc_dir: str | Path) -> list[dict]:
    """Build a byte-stable, case-level manifest from UTF-8 source files."""
    cases, seen = [], set()
    for path in sorted(Path(doc_dir).glob("*.txt"), key=lambda item: item.name):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        case_id, case_number, source_family = _case_identity(path.name)
        if case_id in seen:
            raise EvaluationDataError(f"duplicate corpus case_id: {case_id}")
        seen.add(case_id)
        title = path.stem.split("(FBM-")[0].strip()
        keywords = [
            value.strip()
            for value in re.split(r"[；;、，,\s]+", _first_section(text, "关键词"))
            if value.strip()
        ][:12]
        cases.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "source_family": source_family,
                "title": title,
                "filename": path.name,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "char_count": len(text),
                "keywords": keywords,
                "summary": (_first_section(text, "裁判要点") or _first_section(text, "要旨"))[:800],
            }
        )
    return sorted(cases, key=lambda row: row["case_id"])


def manifest_sha256(cases: Sequence[Mapping]) -> str:
    payload = json.dumps(
        list(cases), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_sample(
    cases: Sequence[Mapping],
    *,
    seed: int,
    core_quotas: Mapping[str, int],
    target_quotas: Mapping[str, int],
    excluded_case_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Select disjoint case-level cohorts with explicit source quotas."""
    excluded = excluded_case_ids or set()
    available: dict[str, list[dict]] = {}
    for case in sorted(cases, key=lambda row: row["case_id"]):
        if case["case_id"] not in excluded:
            available.setdefault(str(case["source_family"]), []).append(dict(case))
    selected_core, selected_target = [], []
    for family in sorted(set(core_quotas) | set(target_quotas)):
        pool = available.get(family, [])
        needed = int(core_quotas.get(family, 0)) + int(target_quotas.get(family, 0))
        if len(pool) < needed:
            raise EvaluationDataError(
                f"source family {family} needs {needed} cases; only {len(pool)} available"
            )
        random.Random(f"{seed}:{family}").shuffle(pool)
        core_n = int(core_quotas.get(family, 0))
        selected_core.extend(pool[:core_n])
        selected_target.extend(pool[core_n:needed])
    return (
        sorted(selected_core, key=lambda row: row["case_id"]),
        sorted(selected_target, key=lambda row: row["case_id"]),
    )


TARGET_QUERY_TYPES = (
    ["scenario"] * 24
    + ["abstract"] * 14
    + ["keywords"] * 9
    + ["procedure"] * 8
    + ["noisy"] * 8
    + ["exact_id"] * 7
)


def _draft_query(case: Mapping, query_type: str) -> str:
    concepts = "、".join(case.get("keywords", [])[:3]) or "相关法律争议"
    if query_type == "exact_id":
        return f"{case['case_number']}主要确立了什么裁判或办案规则？"
    templates = {
        "scenario": f"现实中遇到与{concepts}有关的情况，一般应重点分析哪些法律问题？",
        "abstract": f"司法实践中如何理解{concepts}的判断标准？",
        "keywords": f"{concepts}相关指导案例",
        "procedure": f"涉及{concepts}争议时，可以通过什么程序主张权利？",
        "noisy": f"想问下，碰到{concepts}这种事通常咋处理？",
    }
    return templates[query_type]


def leakage_flags(query: str, case: Mapping, *, query_type: str) -> list[str]:
    normalized = re.sub(r"\s+", "", query).lower()
    flags = []
    if "fbm-" in normalized:
        flags.append("source_id")
    case_number = re.sub(r"\s+", "", str(case.get("case_number", ""))).lower()
    if query_type != "exact_id" and case_number and case_number != "unknown" and case_number in normalized:
        flags.append("case_number")
    title = re.sub(r"\s+", "", str(case.get("title", ""))).lower()
    if title and title in normalized:
        flags.append("full_title")
    return flags


def build_review_drafts(
    core_cases: Sequence[Mapping],
    target_cases: Sequence[Mapping],
    *,
    corpus_sha256: str,
) -> tuple[list[dict], list[dict]]:
    core_rows = []
    for index, case in enumerate(core_cases, 1):
        query = _draft_query(case, "abstract")
        core_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"core-{index:03d}",
                "query": query,
                "evaluation_type": "core",
                "relevant_case_ids": [case["case_id"]],
                "judgments": [{"case_id": case["case_id"], "relevance": 2}],
                "judgments_exhaustive": False,
                "review_status": "generated",
                "metadata": {
                    "corpus_sha256": corpus_sha256,
                    "source_family": case["source_family"],
                    "query_type": "abstract",
                    "draft_purpose": "pooling_seed_only",
                    "target_title": case["title"],
                },
            }
        )
    target_rows = []
    for index, case in enumerate(target_cases, 1):
        query_type = TARGET_QUERY_TYPES[index - 1] if len(target_cases) == 70 else "abstract"
        query = _draft_query(case, query_type)
        target_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"target-{index:03d}",
                "query": query,
                "evaluation_type": "target",
                "relevant_case_ids": [case["case_id"]],
                "judgments_exhaustive": False,
                "review_status": "generated",
                "target_leakage_checked": False,
                "metadata": {
                    "corpus_sha256": corpus_sha256,
                    "source_family": case["source_family"],
                    "query_type": query_type,
                    "generation_method": "deterministic_template_v1",
                    "target_title": case["title"],
                    "leakage_flags": leakage_flags(query, case, query_type=query_type),
                },
            }
        )
    validate_records(core_rows + target_rows)
    return core_rows, target_rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def prepare_review_artifacts(
    doc_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260713,
    excluded_case_ids: set[str] | None = None,
) -> dict:
    cases = build_corpus_manifest(doc_dir)
    digest = manifest_sha256(cases)
    core_cases, target_cases = deterministic_sample(
        cases,
        seed=seed,
        core_quotas={"最高法": 16, "最高检": 14},
        target_quotas={"最高法": 37, "最高检": 33},
        excluded_case_ids=excluded_case_ids,
    )
    core_rows, target_rows = build_review_drafts(
        core_cases, target_cases, corpus_sha256=digest
    )
    destination = Path(output_dir)
    manifest_payload = json.dumps(
        {"schema_version": 1, "corpus_sha256": digest, "case_count": len(cases), "cases": cases},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    _atomic_write(destination / "corpus_manifest.json", manifest_payload)
    _atomic_write(destination / "core_review.jsonl", canonical_jsonl(core_rows))
    _atomic_write(destination / "target_review.jsonl", canonical_jsonl(target_rows))
    return {
        "corpus_sha256": digest,
        "case_count": len(cases),
        "core_count": len(core_rows),
        "target_count": len(target_rows),
    }


def collect_case_ids(paths: Sequence[str | Path]) -> set[str]:
    case_ids = set()
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"{path}:{line_number}", f"invalid JSON: {exc.msg}")
            relevant = row.get("relevant_case_ids", []) if isinstance(row, Mapping) else []
            if not isinstance(relevant, list):
                _fail(f"{path}:{line_number}", "relevant_case_ids must be a list")
            case_ids.update(value for value in relevant if isinstance(value, str))
    return case_ids


def publish_records(
    records: Iterable[Mapping],
    output_path: str | Path,
    *,
    evaluation_type: str,
    expected_count: int,
    known_case_ids: set[str] | None = None,
) -> str:
    """Atomically publish exactly the approved cohort, otherwise fail closed."""
    if evaluation_type not in EVALUATION_TYPES or expected_count < 1:
        raise EvaluationDataError("invalid publication contract")
    rows = validate_records(records, source="<publish>", known_case_ids=known_case_ids)
    approved = [
        row
        for row in rows
        if row["evaluation_type"] == evaluation_type
        and row["review_status"] == APPROVED_STATUS[evaluation_type]
    ]
    if len(approved) != expected_count:
        raise EvaluationDataError(
            f"publication requires exactly {expected_count} approved {evaluation_type} records; got {len(approved)}"
        )
    payload = canonical_jsonl(approved)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or validate evaluation data")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("path", type=Path)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--doc-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20260713)
    prepare.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    publish = commands.add_parser("publish")
    publish.add_argument("--input", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument("--protocol", choices=sorted(EVALUATION_TYPES), required=True)
    publish.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        rows, digest = load_jsonl(args.path)
        result = {"records": len(rows), "sha256": digest}
    elif args.command == "prepare":
        result = prepare_review_artifacts(
            args.doc_dir,
            args.output_dir,
            seed=args.seed,
            excluded_case_ids=collect_case_ids(args.exclude_jsonl),
        )
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        known = {row["case_id"] for row in manifest["cases"]}
        rows, _ = load_jsonl(args.input, known_case_ids=known)
        digest = publish_records(rows, args.output, evaluation_type=args.protocol,
                                 expected_count=args.expected_count, known_case_ids=known)
        result = {"published": args.expected_count, "protocol": args.protocol, "sha256": digest}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
