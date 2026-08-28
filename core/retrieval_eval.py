"""
Offline evaluation of retrieval quality: the metrics, and the runner that
applies them to HireFlow's search.

*Offline* means the correct answers are known in advance, so no human judge and
no LLM are involved. This measures **retrieval** — did the right resumes come
back, and were they near the top — not answer quality.

A test case names a query and the skill(s) a candidate must have to count as
relevant:

    {"query": "tax preparation specialist", "relevant_skills": ["tax preparation"]}

Ground truth is then read off the indexed metadata: every candidate whose
parsed skills contain all the listed skills. Worth understanding before
trusting the numbers — retrieval searches the resume *text*, while relevance is
judged on the *skills Gemini extracted*. Both come from the same document by
different routes, so the metrics ask whether text search surfaces the people who
genuinely hold a skill.

A case may instead pin exact ids with "relevant_candidate_ids" for hand-curated
ground truth.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from utils.utils import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = _PROJECT_ROOT / "data" / "eval" / "test_cases.json"


# ===========================================================================
#  Metrics
#
#  Each takes a ranked list of retrieved ids plus the set of ids that are
#  actually relevant, and returns a score in [0, 1]. Relevance is binary: a
#  candidate either matches the requirement or does not.
#
#  The three answer different questions:
#      MRR   How far down is the FIRST good result? Blind to the rest.
#      MAP   Are ALL the good results near the top?
#      NDCG  Same, log-discounted by position and normalised against the
#            best possible ordering.
# ===========================================================================

def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none was found.

        first relevant at position 1 -> 1.000
        first relevant at position 2 -> 0.500
        first relevant at position 5 -> 0.200

    Positions are 1-based, matching how the metric is normally written.
    """
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """Mean of the precision values measured at each relevant hit.

    Walking the ranking, every time a relevant item appears we record
    precision@that position, then average those records:

        ranked:    [good, bad, good, bad]   relevant total = 2
        hit at 1 -> precision = 1/1 = 1.000
        hit at 3 -> precision = 2/3 = 0.667
        AP = (1.000 + 0.667) / 2 = 0.833

    The divisor is the number of relevant items that could have appeared in a
    list this long — min(total relevant, list length) — so a query with 30
    relevant candidates is not automatically penalised when only 10 are shown.
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / position

    attainable = min(len(relevant_ids), len(ranked_ids))
    return precision_sum / attainable if attainable else 0.0


def dcg(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """Discounted cumulative gain with binary relevance.

        DCG = sum over positions of  relevance / log2(position + 1)

    A hit at position 1 contributes 1/log2(2) = 1.0, at position 2
    1/log2(3) = 0.63, at position 3 0.5 — later hits are worth steadily less,
    but never nothing.
    """
    return sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked_ids, start=1)
        if doc_id in relevant_ids
    )


def ndcg(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """DCG divided by the DCG of a perfect ranking (hence "normalised").

    The ideal ranking puts every relevant item first, so IDCG is the DCG of
    min(relevant, retrieved) hits in the leading positions. Dividing by it
    makes scores comparable across queries with different numbers of relevant
    candidates.
    """
    if not relevant_ids or not ranked_ids:
        return 0.0

    ideal_hits = min(len(relevant_ids), len(ranked_ids))
    ideal_dcg = sum(1.0 / math.log2(position + 1)
                    for position in range(1, ideal_hits + 1))
    return dcg(ranked_ids, relevant_ids) / ideal_dcg if ideal_dcg else 0.0


def evaluate_ranking(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> Dict[str, float]:
    """All three metrics for a single query."""
    return {
        "reciprocal_rank": reciprocal_rank(ranked_ids, relevant_ids),
        "average_precision": average_precision(ranked_ids, relevant_ids),
        "ndcg": ndcg(ranked_ids, relevant_ids),
    }


def aggregate(per_query: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Average per-query metrics into MRR, MAP, and mean NDCG.

    The 'M' in MRR and MAP is exactly this mean across queries — one query
    alone has a reciprocal rank, not a *mean* reciprocal rank.
    """
    rows = list(per_query)
    if not rows:
        return {"mrr": 0.0, "map": 0.0, "ndcg": 0.0, "queries": 0}

    def mean(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    return {
        "mrr": mean("reciprocal_rank"),
        "map": mean("average_precision"),
        "ndcg": mean("ndcg"),
        "queries": len(rows),
    }


# ===========================================================================
#  Test cases and the evaluation run
# ===========================================================================

@dataclass
class EvalCase:
    """One query plus its definition of a relevant result."""
    query: str
    relevant_skills: List[str] = field(default_factory=list)
    relevant_candidate_ids: List[str] = field(default_factory=list)
    note: str = ""

    def resolve_relevant(self, corpus_metadata: Sequence[Dict[str, Any]]) -> Set[str]:
        """Return the candidate ids that count as relevant for this case."""
        if self.relevant_candidate_ids:
            return set(self.relevant_candidate_ids)

        wanted = {s.strip().lower() for s in self.relevant_skills if s.strip()}
        if not wanted:
            return set()

        relevant = set()
        for meta in corpus_metadata:
            skills = {str(s).strip().lower() for s in (meta.get("skills") or [])}
            if wanted <= skills:  # candidate has every required skill
                relevant.add(meta.get("candidate_id"))
        relevant.discard(None)
        return relevant


@dataclass
class CaseResult:
    """Outcome of running one test case."""
    query: str
    note: str
    relevant_count: int
    retrieved: List[Dict[str, Any]]     # the ranked hits, with their scores
    hits: List[bool]                    # was each retrieved id relevant?
    reciprocal_rank: float
    average_precision: float
    ndcg: float

    @property
    def first_hit_rank(self) -> int | None:
        """1-based position of the first relevant result, if any."""
        for position, hit in enumerate(self.hits, start=1):
            if hit:
                return position
        return None


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> List[EvalCase]:
    """Read test cases from JSON. Returns [] if the file is missing."""
    path = Path(path)
    if not path.is_file():
        logger.warning(f"No evaluation cases found at {path}")
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Could not read evaluation cases: {e}")
        return []

    entries = raw.get("cases", []) if isinstance(raw, dict) else raw
    return [
        EvalCase(
            query=entry["query"].strip(),
            relevant_skills=entry.get("relevant_skills", []),
            relevant_candidate_ids=entry.get("relevant_candidate_ids", []),
            note=entry.get("note", ""),
        )
        for entry in entries
        if (entry.get("query") or "").strip()
    ]


def run_case(indexer, case: EvalCase, top_k: int = 10) -> CaseResult | None:
    """Run one query through hybrid search and score the ranking.

    Returns None when the case has no relevant candidates in this index —
    scoring against empty ground truth would just report a misleading zero.
    """
    relevant = case.resolve_relevant(indexer.resume_metadata)
    if not relevant:
        logger.warning(f"No relevant candidates for {case.query!r} — skipping")
        return None

    results = indexer.search_resumes(case.query, top_k=top_k)
    ranked_ids = [r.get("candidate_id") for r in results]
    metrics = evaluate_ranking(ranked_ids, relevant)

    return CaseResult(
        query=case.query,
        note=case.note,
        relevant_count=len(relevant),
        retrieved=[
            {
                "candidate_id": r.get("candidate_id"),
                "name": r.get("name", "Unknown"),
                "bm25_score": round(r.get("bm25_score", 0.0), 4),
                "vector_score": round(r.get("vector_score", 0.0), 4),
                "combined_score": round(r.get("combined_score", 0.0), 4),
            }
            for r in results
        ],
        hits=[doc_id in relevant for doc_id in ranked_ids],
        **metrics,
    )


def run_evaluation(indexer, cases: Sequence[EvalCase] = None,
                   top_k: int = 10) -> Dict[str, Any]:
    """Run every case and return per-case results plus the aggregate metrics."""
    cases = list(cases) if cases is not None else load_cases()

    results: List[CaseResult] = []
    skipped = 0
    for case in cases:
        result = run_case(indexer, case, top_k=top_k)
        if result is None:
            skipped += 1
        else:
            results.append(result)

    summary = aggregate([
        {
            "reciprocal_rank": r.reciprocal_rank,
            "average_precision": r.average_precision,
            "ndcg": r.ndcg,
        }
        for r in results
    ])

    return {"results": results, "summary": summary, "top_k": top_k, "skipped": skipped}
