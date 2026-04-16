"""Unit tests for core/hybrid_indexer.py — no external API calls required."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.schema import Document
from core.hybrid_indexer import HybridIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(candidate_id: str, name: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"candidate_id": candidate_id, "name": name, "source": f"{candidate_id}.pdf"},
    )


def _make_indexer() -> HybridIndexer:
    """Return a HybridIndexer with VectorStore completely mocked out."""
    with patch("core.hybrid_indexer.VectorStore") as MockVS:
        instance = MockVS.return_value
        instance.initialize.return_value = True
        instance.is_ready.return_value = False  # skip Pinecone in most tests
        instance.add_resumes.return_value = True
        indexer = HybridIndexer()
        indexer.vector_store = instance
    return indexer


# ---------------------------------------------------------------------------
# index_resumes
# ---------------------------------------------------------------------------

class TestIndexResumes:
    def test_returns_false_for_empty_list(self):
        indexer = _make_indexer()
        assert indexer.index_resumes([]) is False

    def test_builds_bm25_model(self):
        indexer = _make_indexer()
        docs = [
            _make_doc("c1", "Alice", "python developer with django experience"),
            _make_doc("c2", "Bob", "java spring boot microservices"),
        ]
        result = indexer.index_resumes(docs)
        assert result is True
        assert indexer.bm25_resumes is not None
        assert len(indexer.resume_texts) == 2

    def test_stores_lowercased_texts(self):
        indexer = _make_indexer()
        docs = [_make_doc("c1", "Alice", "Python Developer")]
        indexer.index_resumes(docs)
        assert indexer.resume_texts[0] == "python developer"

    def test_calls_vector_store_when_ready(self):
        indexer = _make_indexer()
        indexer.vector_store.is_ready.return_value = True
        docs = [_make_doc("c1", "Alice", "python developer")]
        indexer.index_resumes(docs)
        indexer.vector_store.add_resumes.assert_called_once_with(docs)

    def test_skips_vector_store_when_not_ready(self):
        indexer = _make_indexer()
        indexer.vector_store.is_ready.return_value = False
        docs = [_make_doc("c1", "Alice", "python developer")]
        indexer.index_resumes(docs)
        indexer.vector_store.add_resumes.assert_not_called()


# ---------------------------------------------------------------------------
# combine_results (core RRF logic — no I/O)
# ---------------------------------------------------------------------------

class TestCombineResults:
    def _setup_indexer(self, texts: list) -> HybridIndexer:
        indexer = _make_indexer()
        indexer.resume_texts = texts
        return indexer

    def test_empty_inputs_return_empty(self):
        indexer = self._setup_indexer([])
        result = indexer.combine_results([], [], top_k=5)
        assert result == []

    def test_bm25_only_results_ranked_by_score(self):
        indexer = self._setup_indexer([
            "python django developer",
            "java spring developer",
            "react frontend engineer",
        ])
        # c_0 gets highest BM25 score
        bm25_scores = [3.0, 1.0, 0.5]
        result = indexer.combine_results(bm25_scores, [], top_k=3)
        assert result[0]["candidate_id"] == "c_0"
        assert result[1]["candidate_id"] == "c_1"

    def test_combined_score_present_in_results(self):
        indexer = self._setup_indexer(["python developer"])
        result = indexer.combine_results([2.0], [], top_k=1)
        assert "combined_score" in result[0]
        assert "bm25_score" in result[0]
        assert "vector_score" in result[0]

    def test_top_k_limits_results(self):
        texts = [f"candidate {i}" for i in range(10)]
        indexer = self._setup_indexer(texts)
        bm25_scores = [float(i) for i in range(10)]
        result = indexer.combine_results(bm25_scores, [], top_k=3)
        assert len(result) == 3

    def test_vector_results_merged_by_candidate_id(self):
        indexer = self._setup_indexer(["python developer"])
        bm25_scores = [1.0]
        vector_results = [
            {
                "page_content": "python developer",
                "metadata": {
                    "candidate_id": "c_0",
                    "name": "Alice",
                    "skills": ["Python"],
                    "location": "New York",
                    "experience": 5,
                },
                "score": 0.9,
            }
        ]
        result = indexer.combine_results(bm25_scores, vector_results, top_k=5)
        # c_0 appears in both lists — should only show up once
        ids = [r["candidate_id"] for r in result]
        assert ids.count("c_0") == 1
        # Should have a higher combined score than if it appeared in only one list
        combined = result[0]["combined_score"]
        assert combined > 1 / (60 + 1)  # greater than a single-list RRF score

    def test_bm25_scores_normalized(self):
        indexer = self._setup_indexer(["doc one", "doc two"])
        bm25_scores = [10.0, 5.0]
        result = indexer.combine_results(bm25_scores, [], top_k=2)
        # Top result's bm25_score should be 1.0 (normalized)
        assert result[0]["bm25_score"] == 1.0
        assert result[1]["bm25_score"] == 0.5


# ---------------------------------------------------------------------------
# search_resumes (integration path — mocked BM25 + vector store)
# ---------------------------------------------------------------------------

class TestSearchResumes:
    def test_returns_empty_when_bm25_not_built(self):
        indexer = _make_indexer()
        # bm25_resumes is None by default
        result = indexer.search_resumes("python developer")
        assert result == []

    def test_calls_combine_results(self):
        indexer = _make_indexer()
        # Prime a real BM25 model
        docs = [_make_doc("c1", "Alice", "python developer with experience")]
        indexer.index_resumes(docs)
        result = indexer.search_resumes("python developer", top_k=1)
        assert isinstance(result, list)
