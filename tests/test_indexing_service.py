"""Unit tests for core/indexing_service.py — no external API calls required.

These pin down the property the indexing redesign exists for: parsing happens
once, and app startup costs zero LLM calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.index_store import IndexStore, file_fingerprint
from core.indexing_service import add_resumes, build_index, index_status, load_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeParser:
    """Stands in for ResumeParser; records every call so we can count them."""

    def __init__(self, calls):
        self.calls = calls

    def parse_resume(self, text, candidate_id):
        self.calls.append(candidate_id)
        return {
            "name": f"Name {candidate_id}",
            "skills": ["Python"],
            "location": "NYC",
            "experience": 5,
        }


@pytest.fixture
def llm_calls():
    """Collects candidate_ids sent to the (fake) LLM during a test."""
    return []


@pytest.fixture(autouse=True)
def offline(llm_calls):
    """Replace PDF reading and Gemini parsing with local stand-ins."""
    def fake_load_pdf(path):
        return Path(path).read_bytes().decode()

    parser = _FakeParser(llm_calls)
    with patch("core.ingestion.load_pdf", fake_load_pdf), \
         patch("core.ingestion._get_parser", lambda: parser), \
         patch("core.indexing_service._get_parser", lambda: parser):
        yield


@pytest.fixture
def resumes_dir(tmp_path):
    d = tmp_path / "resumes"
    d.mkdir()
    _write(d, "alice.pdf", "Alice python developer sql")
    _write(d, "bob.pdf", "Bob java spring engineer")
    _write(d, "carol.pdf", "Carol data scientist tensorflow")
    return d


@pytest.fixture
def store(tmp_path):
    return IndexStore(tmp_path / "index")


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_bytes(body.encode())
    return path


def _make_indexer():
    """HybridIndexer with Pinecone stubbed out."""
    with patch("core.hybrid_indexer.VectorStore") as MockVS:
        vs = MockVS.return_value
        vs.initialize.return_value = True
        vs.is_ready.return_value = True
        vs.add_resumes.return_value = True
        vs.embeddings = MagicMock(model_name="all-MiniLM-L6-v2")
        from core.hybrid_indexer import HybridIndexer
        return HybridIndexer()


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_first_build_parses_every_resume(self, resumes_dir, store, llm_calls):
        report = build_index(_make_indexer(), resumes_dir, store=store)

        assert report.parsed == 3
        assert report.total_indexed == 3
        assert len(llm_calls) == 3
        assert report.persisted and store.exists()

    def test_second_build_parses_nothing(self, resumes_dir, store, llm_calls):
        build_index(_make_indexer(), resumes_dir, store=store)
        llm_calls.clear()

        report = build_index(_make_indexer(), resumes_dir, store=store)

        assert llm_calls == []
        assert report.parsed == 0
        assert report.total_indexed == 3

    def test_only_new_files_are_parsed(self, resumes_dir, store, llm_calls):
        build_index(_make_indexer(), resumes_dir, store=store)
        llm_calls.clear()
        _write(resumes_dir, "dave.pdf", "Dave devops kubernetes")

        report = build_index(_make_indexer(), resumes_dir, store=store)

        assert llm_calls == ["c_dave"]
        assert report.parsed == 1
        assert report.skipped == 3
        assert report.total_indexed == 4

    def test_only_new_files_are_upserted(self, resumes_dir, store):
        build_index(_make_indexer(), resumes_dir, store=store)
        _write(resumes_dir, "dave.pdf", "Dave devops kubernetes")

        indexer = _make_indexer()
        build_index(indexer, resumes_dir, store=store)

        upserted = indexer.vector_store.add_resumes.call_args[0][0]
        assert [d.metadata["candidate_id"] for d in upserted] == ["c_dave"]

    def test_changed_file_is_reparsed(self, resumes_dir, store, llm_calls):
        build_index(_make_indexer(), resumes_dir, store=store)
        llm_calls.clear()
        _write(resumes_dir, "bob.pdf", "Bob now a golang engineer")

        indexer = _make_indexer()
        report = build_index(indexer, resumes_dir, store=store)

        assert llm_calls == ["c_bob"]
        assert report.total_indexed == 3
        assert any("golang" in t for t in indexer.resume_texts)

    def test_deleted_file_is_dropped(self, resumes_dir, store, llm_calls):
        build_index(_make_indexer(), resumes_dir, store=store)
        llm_calls.clear()
        (resumes_dir / "carol.pdf").unlink()

        indexer = _make_indexer()
        report = build_index(indexer, resumes_dir, store=store)

        assert llm_calls == []
        assert report.total_indexed == 2
        filenames = {m["filename"] for m in indexer.resume_metadata}
        assert "carol.pdf" not in filenames

    def test_force_reparses_everything(self, resumes_dir, store, llm_calls):
        build_index(_make_indexer(), resumes_dir, store=store)
        llm_calls.clear()

        report = build_index(_make_indexer(), resumes_dir, store=store, force=True)

        assert len(llm_calls) == 3
        assert report.forced
        assert report.skipped == 0
        assert report.total_indexed == 3

    def test_missing_directory_is_not_an_error(self, tmp_path, store):
        report = build_index(_make_indexer(), tmp_path / "nope", store=store)
        assert report.total_indexed == 0
        assert report.parsed == 0

    def test_bm25_is_usable_after_build(self, resumes_dir, store):
        indexer = _make_indexer()
        indexer.vector_store.search_resumes.return_value = []
        build_index(indexer, resumes_dir, store=store)

        assert indexer.bm25_resumes is not None
        assert indexer.search_resumes("python developer", top_k=2)


# ---------------------------------------------------------------------------
# load_index — the app-startup path
# ---------------------------------------------------------------------------

class TestLoadIndex:
    def test_startup_makes_no_llm_calls(self, resumes_dir, store, llm_calls):
        build_index(_make_indexer(), resumes_dir, store=store)
        llm_calls.clear()

        indexer = _make_indexer()
        assert load_index(indexer, store=store) is True
        assert llm_calls == []
        assert len(indexer.resume_texts) == 3

    def test_startup_makes_no_pinecone_writes(self, resumes_dir, store):
        build_index(_make_indexer(), resumes_dir, store=store)

        indexer = _make_indexer()
        load_index(indexer, store=store)

        assert indexer.vector_store.add_resumes.call_count == 0

    def test_search_works_after_load(self, resumes_dir, store):
        build_index(_make_indexer(), resumes_dir, store=store)

        indexer = _make_indexer()
        indexer.vector_store.search_resumes.return_value = []
        load_index(indexer, store=store)

        results = indexer.search_resumes("python developer sql", top_k=2)
        assert results
        assert results[0]["candidate_id"].startswith("c_")

    def test_returns_false_without_an_index(self, store):
        indexer = _make_indexer()
        assert load_index(indexer, store=store) is False
        assert indexer.bm25_resumes is None

    def test_corrupt_corpus_is_treated_as_empty(self, store):
        store.index_dir.mkdir(parents=True, exist_ok=True)
        store.corpus_path.write_bytes(b"not a pickle")
        assert load_index(_make_indexer(), store=store) is False


# ---------------------------------------------------------------------------
# add_resumes — the upload path
# ---------------------------------------------------------------------------

class TestAddResumes:
    def test_upload_survives_restart(self, resumes_dir, store, llm_calls):
        from core.ingestion import build_resume_document

        build_index(_make_indexer(), resumes_dir, store=store)
        path = _write(resumes_dir, "eve.pdf", "Eve product manager agile")
        doc = build_resume_document(str(path), "eve.pdf", parser=_FakeParser(llm_calls))

        indexer = _make_indexer()
        load_index(indexer, store=store)
        assert add_resumes(indexer, [doc], store=store, resumes_dir=resumes_dir)

        restarted = _make_indexer()
        load_index(restarted, store=store)
        assert "eve.pdf" in {m["filename"] for m in restarted.resume_metadata}

    def test_uploaded_file_is_not_reparsed_later(self, resumes_dir, store, llm_calls):
        from core.ingestion import build_resume_document

        build_index(_make_indexer(), resumes_dir, store=store)
        path = _write(resumes_dir, "eve.pdf", "Eve product manager agile")
        doc = build_resume_document(str(path), "eve.pdf", parser=_FakeParser(llm_calls))

        indexer = _make_indexer()
        load_index(indexer, store=store)
        add_resumes(indexer, [doc], store=store, resumes_dir=resumes_dir)

        llm_calls.clear()
        report = build_index(_make_indexer(), resumes_dir, store=store)
        assert llm_calls == []
        assert report.total_indexed == 4

    def test_reupload_replaces_instead_of_duplicating(self, resumes_dir, store, llm_calls):
        from core.ingestion import build_resume_document

        build_index(_make_indexer(), resumes_dir, store=store)
        indexer = _make_indexer()
        load_index(indexer, store=store)

        path = _write(resumes_dir, "alice.pdf", "Alice updated rust developer")
        doc = build_resume_document(str(path), "alice.pdf", parser=_FakeParser(llm_calls))
        add_resumes(indexer, [doc], store=store, resumes_dir=resumes_dir)

        ids = [m["candidate_id"] for m in indexer.resume_metadata]
        assert ids.count("c_alice") == 1
        assert len(indexer.resume_texts) == 3

    def test_empty_list_is_a_no_op(self, store):
        assert add_resumes(_make_indexer(), [], store=store) is False


# ---------------------------------------------------------------------------
# index_status
# ---------------------------------------------------------------------------

class TestIndexStatus:
    def test_reports_counts_and_timestamp(self, resumes_dir, store):
        build_index(_make_indexer(), resumes_dir, store=store)

        status = index_status(store=store)
        assert status["index_exists"] is True
        assert status["indexed_resumes"] == 3
        assert status["updated_at"]

    def test_reports_nothing_when_unbuilt(self, store):
        status = index_status(store=store)
        assert status["index_exists"] is False
        assert status["indexed_resumes"] == 0


# ---------------------------------------------------------------------------
# IndexStore
# ---------------------------------------------------------------------------

class TestIndexStore:
    def test_fingerprint_tracks_content_not_mtime(self, tmp_path):
        a = _write(tmp_path, "a.pdf", "same bytes")
        b = _write(tmp_path, "b.pdf", "same bytes")
        c = _write(tmp_path, "c.pdf", "different bytes")

        assert file_fingerprint(a) == file_fingerprint(b)
        assert file_fingerprint(a) != file_fingerprint(c)

    def test_round_trip(self, store):
        files = {"a.pdf": {"sha256": "abc", "candidate_id": "c_a"}}
        assert store.save(["text a"], [{"candidate_id": "c_a"}], files)

        texts, metadata = store.load_corpus()
        assert texts == ["text a"]
        assert metadata[0]["candidate_id"] == "c_a"
        assert store.indexed_files() == files

    def test_mismatched_lengths_are_rejected(self, store):
        assert store.save(["a", "b"], [{"candidate_id": "c_a"}], {}) is False

    def test_clear_removes_the_index(self, store):
        store.save(["text"], [{"candidate_id": "c_a"}], {})
        assert store.exists()
        store.clear()
        assert not store.exists()

    def test_version_mismatch_is_ignored(self, store):
        import json
        store.save(["text"], [{"candidate_id": "c_a"}], {})
        manifest = json.loads(store.manifest_path.read_text())
        manifest["version"] = 999
        store.manifest_path.write_text(json.dumps(manifest))

        assert store.indexed_files() == {}
