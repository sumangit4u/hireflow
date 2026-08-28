"""
HireFlow FastAPI backend.

Exposes three endpoints:
    POST /index   — triggers full indexing of resumes in data/resumes/
    POST /search  — hybrid search returning ranked candidates
    GET  /status  — returns current index stats

Run with:
    python start_backend.py
or:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bootstrap sys.path so imports work whether run from repo root or api/
# ---------------------------------------------------------------------------
import sys
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.hybrid_indexer import HybridIndexer
from core.indexing_service import build_index, index_status, load_index

# ---------------------------------------------------------------------------
# Application-level singleton — shared across requests
# ---------------------------------------------------------------------------
app = FastAPI(title="HireFlow API", version="1.0.0")

_indexer: Optional[HybridIndexer] = None
_DATA_RESUMES_DIR = _project_root / "data" / "resumes"


def get_indexer() -> HybridIndexer:
    """Return (and lazily initialise) the shared HybridIndexer instance.

    The persisted index is loaded once, on first use. That is a local file read
    plus an in-memory BM25 rebuild — no PDF parsing, no Gemini calls — so the
    backend comes up searchable without re-running POST /index after a restart.
    """
    global _indexer
    if _indexer is None:
        _indexer = HybridIndexer()
        load_index(_indexer)
    return _indexer


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CandidateResult(BaseModel):
    candidate_id: str
    name: str
    bm25_score: float
    vector_score: float
    combined_score: float
    skills: List[str]
    location: str
    experience: Optional[int]


class SearchResponse(BaseModel):
    results: List[CandidateResult]
    total: int


class IndexResponse(BaseModel):
    indexed: int
    message: str


class StatusResponse(BaseModel):
    resumes_ready: bool
    vector_store_ready: bool
    hybrid_ready: bool
    pinecone_vector_count: int
    indexed_resumes: int
    last_indexed: Optional[str]
    unindexed_pdfs: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/index", response_model=IndexResponse, summary="Index resumes in data/resumes/")
def index_resumes(force: bool = False):
    """Index the PDFs in data/resumes/ into BM25 + Pinecone and persist the result.

    Incremental by default — only new or changed PDFs are parsed. Pass
    `?force=true` to re-parse every resume (one Gemini call each).
    """
    indexer = get_indexer()
    report = build_index(indexer, resumes_dir=str(_DATA_RESUMES_DIR), force=force)

    if report.total_indexed == 0:
        raise HTTPException(status_code=404, detail="No resume PDFs found in data/resumes/")
    if not report.persisted:
        raise HTTPException(status_code=500, detail="Indexing failed — check server logs")

    return IndexResponse(
        indexed=report.total_indexed,
        message=(
            f"Index ready: {report.summary()}"
            + (f"; {len(report.failed)} failed" if report.failed else "")
        ),
    )


@app.post("/search", response_model=SearchResponse, summary="Search for matching candidates")
def search(request: SearchRequest):
    """Run a hybrid BM25 + vector search and return ranked candidates."""
    indexer = get_indexer()
    if not indexer.bm25_resumes:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Run 'python index_resumes.py' or call POST /index first."
        )
    raw_results = indexer.search_resumes(request.query, top_k=request.top_k)
    results = [
        CandidateResult(
            candidate_id=r.get("candidate_id", ""),
            name=r.get("name", "Unknown"),
            bm25_score=round(r.get("bm25_score", 0.0), 4),
            vector_score=round(r.get("vector_score", 0.0), 4),
            combined_score=round(r.get("combined_score", 0.0), 4),
            skills=r.get("skills", []),
            location=r.get("location", "Unknown"),
            experience=r.get("experience"),
        )
        for r in raw_results
    ]
    return SearchResponse(results=results, total=len(results))


@app.get("/status", response_model=StatusResponse, summary="Get current index status")
def status():
    """Return BM25 and Pinecone readiness along with vector count."""
    indexer = get_indexer()
    stats = indexer.get_index_stats()
    index_info = index_status(indexer)
    return StatusResponse(
        resumes_ready=stats["resumes_ready"],
        vector_store_ready=stats["vector_store_ready"],
        hybrid_ready=stats["hybrid_ready"],
        pinecone_vector_count=index_info.get("pinecone_vector_count", 0),
        indexed_resumes=index_info["indexed_resumes"],
        last_indexed=index_info["updated_at"],
        unindexed_pdfs=len(index_info["unindexed_pdfs"]),
    )
