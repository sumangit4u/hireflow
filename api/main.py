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
from core.ingestion import load_resumes

# ---------------------------------------------------------------------------
# Application-level singleton — shared across requests
# ---------------------------------------------------------------------------
app = FastAPI(title="HireFlow API", version="1.0.0")

_indexer: Optional[HybridIndexer] = None
_DATA_RESUMES_DIR = _project_root / "data" / "resumes"


def get_indexer() -> HybridIndexer:
    """Return (and lazily initialise) the shared HybridIndexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = HybridIndexer()
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/index", response_model=IndexResponse, summary="Index all resumes in data/resumes/")
def index_resumes():
    """Load all PDFs from data/resumes/ and (re-)index them in BM25 + Pinecone."""
    indexer = get_indexer()
    resumes = load_resumes(str(_DATA_RESUMES_DIR))
    if not resumes:
        raise HTTPException(status_code=404, detail="No resume PDFs found in data/resumes/")
    ok = indexer.index_resumes(resumes)
    if not ok:
        raise HTTPException(status_code=500, detail="Indexing failed — check server logs")
    return IndexResponse(indexed=len(resumes), message=f"Successfully indexed {len(resumes)} resumes")


@app.post("/search", response_model=SearchResponse, summary="Search for matching candidates")
def search(request: SearchRequest):
    """Run a hybrid BM25 + vector search and return ranked candidates."""
    indexer = get_indexer()
    if not indexer.bm25_resumes:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Call POST /index first."
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
    pinecone_count = 0
    if indexer.vector_store.is_ready():
        pinecone_stats = indexer.vector_store.get_stats()
        pinecone_count = pinecone_stats.get("total_vector_count", 0)
    return StatusResponse(
        resumes_ready=stats["resumes_ready"],
        vector_store_ready=stats["vector_store_ready"],
        hybrid_ready=stats["hybrid_ready"],
        pinecone_vector_count=pinecone_count,
    )
