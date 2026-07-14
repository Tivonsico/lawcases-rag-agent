import json
import gc
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag_agent"))

from chunker import chunk_document, extract_case_metadata
from embedding import EmbeddingService
from vector_store import VectorStore
from config import _select_index_file
from init_db import migrate_legacy_index


SAMPLE_NAME = "指导案例192号 李某侵犯公民个人信息案(FBM-CLI.C.123).txt"
SAMPLE_TEXT = "基本案情\n某人民法院审理本案。《中华人民共和国刑法》第二百五十三条规定相关责任。"


def test_stable_case_id_and_typed_metadata():
    first = extract_case_metadata(SAMPLE_NAME, SAMPLE_TEXT)
    second = extract_case_metadata(SAMPLE_NAME, SAMPLE_TEXT + "补充正文")
    assert first["case_id"] == second["case_id"] == "case:fbm-cli.c.123"
    assert first["case_number"] == "指导案例192号"
    assert first["case_type"] == "最高法指导案例"
    chunks = chunk_document(SAMPLE_NAME, SAMPLE_TEXT, chunk_size=30, chunk_min=5)
    assert all(chunk["case_id"] == first["case_id"] for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert {chunk["section_type"] for chunk in chunks} & {"facts", "body"}


def test_upsert_is_idempotent_and_manifest_rejects_mismatch(tmp_path):
    chunks = chunk_document(SAMPLE_NAME, SAMPLE_TEXT, chunk_size=30, chunk_min=5)
    store = VectorStore(EmbeddingService(mock=True, dim=8), persist_dir=str(tmp_path / "chroma"))
    manifest = store.build_manifest(chunks)
    store.validate_manifest(manifest, allow_missing=True)
    store.add_chunks(chunks)
    count = store.count()
    store.write_manifest(manifest)
    store.add_chunks(chunks)
    assert store.count() == count == len(chunks)
    with pytest.raises(RuntimeError, match="manifest 不兼容"):
        store.validate_manifest({**manifest, "embedding_dimension": 9})
    del store
    gc.collect()


def test_existing_index_without_manifest_is_rejected(tmp_path):
    chunks = chunk_document(SAMPLE_NAME, SAMPLE_TEXT, chunk_size=30, chunk_min=5)
    store = VectorStore(EmbeddingService(mock=True, dim=8), persist_dir=str(tmp_path / "legacy"))
    store.add_chunks(chunks)
    with pytest.raises(RuntimeError, match="缺少 manifest"):
        store.validate_manifest(store.build_manifest(chunks), allow_missing=True)
    del store
    gc.collect()


def test_bm25_path_falls_back_to_existing_legacy_index(tmp_path, monkeypatch):
    runtime_path = tmp_path / "runtime" / "bm25_index.pkl"
    legacy_path = tmp_path / "legacy" / "bm25_index.pkl"
    legacy_path.parent.mkdir()
    legacy_path.write_bytes(b"existing index")
    monkeypatch.delenv("TEST_BM25_PATH", raising=False)

    selected = _select_index_file(
        "TEST_BM25_PATH", str(runtime_path), str(legacy_path)
    )

    assert selected == str(legacy_path.resolve())


def test_bm25_path_override_wins_even_before_file_exists(tmp_path, monkeypatch):
    override = tmp_path / "custom" / "bm25.pkl"
    monkeypatch.setenv("TEST_BM25_PATH", str(override))

    selected = _select_index_file(
        "TEST_BM25_PATH",
        str(tmp_path / "runtime" / "bm25_index.pkl"),
        str(tmp_path / "legacy" / "bm25_index.pkl"),
    )

    assert selected == str(override.resolve())


def test_legacy_migration_requires_exact_vector_id_match(tmp_path):
    source = tmp_path / "legacy.pkl"
    target = tmp_path / "runtime" / "bm25.pkl"
    chunks = [{"chunk_id": "chunk-1", "doc_name": "案例", "chunk_index": 0,
               "text": "正文", "tag": "案情", "keywords": []}]
    source.write_bytes(pickle.dumps({"chunks": chunks}))

    class Collection:
        def get(self, include):
            return {"ids": ["different-id"]}

    class Store:
        collection = Collection()

    with pytest.raises(RuntimeError, match="ID 不一致"):
        migrate_legacy_index(source, target, Store())
    assert not target.exists()


def test_legacy_migration_materializes_implicit_chunk_ids(tmp_path):
    source = tmp_path / "legacy.pkl"
    target = tmp_path / "runtime" / "bm25.pkl"
    chunks = [{"doc_name": "案例", "chunk_index": 0, "text": "正文",
               "tag": "案情", "keywords": []}]
    source.write_bytes(pickle.dumps({"chunks": chunks}))

    class Collection:
        def get(self, include):
            return {"ids": ["案例_chunk0"]}

    class Store:
        collection = Collection()

        def build_manifest(self, migrated):
            assert migrated[0]["chunk_id"] == "案例_chunk0"
            return {"chunk_count": 1}

        def write_manifest(self, manifest):
            assert manifest == {"chunk_count": 1}

    result = migrate_legacy_index(source, target, Store())
    migrated = pickle.loads(target.read_bytes())
    assert result["chunks"] == 1
    assert migrated["chunks"][0]["chunk_id"] == "案例_chunk0"
