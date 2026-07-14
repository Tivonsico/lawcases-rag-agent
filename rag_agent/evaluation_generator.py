"""Bounded, auditable LLM generation for known-target query drafts."""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Mapping, Sequence

import requests

from .evaluation_data import (
    EvaluationDataError,
    canonical_jsonl,
    leakage_flags,
    load_jsonl,
    validate_records,
)

PROMPT_VERSION = "target-query-v1"


def build_prompt(case: Mapping, query_type: str) -> str:
    evidence = str(case.get("summary", ""))[:800]
    keywords = "、".join(case.get("keywords", [])[:8])
    return f"""你在构造中文法律案例检索评测问题。根据下列材料生成一个自然的用户问题。
问题类型：{query_type}
关键词：{keywords}
裁判要点或要旨：{evidence}

要求：
1. 只输出 JSON：{{"query":"..."}}。
2. 不输出答案，不捏造材料之外的事实。
3. 除 exact_id 类型外，不得出现案例编号、FBM 编码、完整标题或当事人姓名。
4. 问题长度 8～180 个字符，像真实用户表达，不照抄材料长句。
"""


def parse_generated_query(raw: str) -> str:
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"generator returned invalid JSON: {exc.msg}") from exc
    query = payload.get("query") if isinstance(payload, dict) else None
    if not isinstance(query, str) or not 8 <= len(query.strip()) <= 180:
        raise EvaluationDataError("generated query must contain 8-180 characters")
    return query.strip()


def generate_target_rows(
    rows: Sequence[Mapping],
    cases_by_id: Mapping[str, Mapping],
    llm_call: Callable[[str], str],
    *,
    model: str,
    max_records: int,
) -> tuple[list[dict], dict]:
    """Generate at most max_records without ever granting human approval."""
    if max_records < 1:
        raise EvaluationDataError("max_records must be positive")
    output, generated, needs_revision = [], 0, 0
    for original in rows:
        row = deepcopy(dict(original))
        if row.get("evaluation_type") != "target" or generated >= max_records:
            output.append(row)
            continue
        case_id = row["relevant_case_ids"][0]
        case = cases_by_id.get(case_id)
        if case is None:
            raise EvaluationDataError(f"{row['id']}: target case missing from manifest: {case_id}")
        query_type = row.get("metadata", {}).get("query_type", "abstract")
        try:
            query = parse_generated_query(llm_call(build_prompt(case, query_type)))
        except Exception as exc:
            if isinstance(exc, EvaluationDataError):
                raise EvaluationDataError(f"{row['id']}: {exc}") from exc
            raise EvaluationDataError(f"{row['id']}: generator call failed: {exc}") from exc
        flags = leakage_flags(query, case, query_type=query_type)
        row["query"] = query
        row["review_status"] = "needs_revision" if flags else "machine_checked"
        row["target_leakage_checked"] = False
        metadata = dict(row.get("metadata", {}))
        metadata.update(
            {
                "generation_method": "llm_from_case",
                "generator": {"model": model, "prompt_version": PROMPT_VERSION},
                "leakage_flags": flags,
            }
        )
        row["metadata"] = metadata
        generated += 1
        needs_revision += bool(flags)
        output.append(row)
    validate_records(output)
    return output, {
        "generated": generated,
        "machine_checked": generated - needs_revision,
        "needs_revision": needs_revision,
    }


def openai_compatible_call(api_url: str, api_key: str, model: str, timeout: float = 60):
    def call(prompt: str) -> str:
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return call


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a bounded number of target query drafts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--max-records", type=int, required=True)
    args = parser.parse_args(argv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing environment variable: {args.api_key_env}")
    rows, _ = load_jsonl(args.input)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = {row["case_id"]: row for row in manifest["cases"]}
    generated, stats = generate_target_rows(
        rows,
        cases,
        openai_compatible_call(args.api_url, api_key, args.model),
        model=args.model,
        max_records=args.max_records,
    )
    payload = canonical_jsonl(generated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, args.output)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
