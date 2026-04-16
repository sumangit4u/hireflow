"""Unit tests for core/re_ranker.py — no external API calls required."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.re_ranker import ReRanker
from utils.schemas import Resume, SearchQuery, CandidateEvaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_resume(candidate_id: str = "c1", skills: list = None, experience: int = 3) -> Resume:
    return Resume(
        candidate_id=candidate_id,
        name="Test Candidate",
        email="test@example.com",
        phone="+1-555-0000",
        skills=skills or [],
        experience=experience,
        text="Experienced Python developer with Django and REST API skills.",
    )


def _make_query(title: str = "Python Developer", required_skills: list = None) -> SearchQuery:
    return SearchQuery(
        title=title,
        text="Looking for a Python developer with Django experience.",
        required_skills=required_skills or ["Python", "Django"],
    )


def _make_reranker_no_llm() -> ReRanker:
    """Return a ReRanker with LLM disabled (forces rule-based path)."""
    with patch("core.re_ranker.GOOGLE_API_KEY", ""):
        reranker = ReRanker()
    reranker.llm = None
    return reranker


# ---------------------------------------------------------------------------
# simple_evaluation
# ---------------------------------------------------------------------------

class TestSimpleEvaluation:
    def test_returns_candidate_evaluation(self):
        reranker = _make_reranker_no_llm()
        result = reranker.simple_evaluation(_make_resume(), _make_query())
        assert isinstance(result, CandidateEvaluation)

    def test_score_in_valid_range(self):
        reranker = _make_reranker_no_llm()
        result = reranker.simple_evaluation(_make_resume(), _make_query())
        assert 0 <= result.fit_score <= 100

    def test_skill_match_increases_score(self):
        reranker = _make_reranker_no_llm()
        resume_with_skills = _make_resume(skills=["Python", "Django"])
        result = reranker.simple_evaluation(resume_with_skills, _make_query())
        assert result.fit_score > 50
        assert len(result.strengths) > 0

    def test_skill_mismatch_decreases_score(self):
        reranker = _make_reranker_no_llm()
        resume_no_skills = _make_resume(skills=["Kotlin", "Android"])
        result = reranker.simple_evaluation(resume_no_skills, _make_query())
        assert result.fit_score < 50
        assert len(result.gaps) > 0

    def test_no_required_skills_neutral_score(self):
        reranker = _make_reranker_no_llm()
        query_no_skills = SearchQuery(title="Any Role", text="Open role", required_skills=[])
        resume = _make_resume(skills=["Python"])
        result = reranker.simple_evaluation(resume, query_no_skills)
        # Has skills but no required skills to match against — slight positive
        assert result.fit_score >= 50

    def test_correct_candidate_id_in_result(self):
        reranker = _make_reranker_no_llm()
        resume = _make_resume(candidate_id="test_id_123")
        result = reranker.simple_evaluation(resume, _make_query())
        assert result.candidate_id == "test_id_123"


# ---------------------------------------------------------------------------
# extract_section
# ---------------------------------------------------------------------------

class TestExtractSection:
    SAMPLE_LLM_OUTPUT = """
Strengths:
- Strong Python background
- Experience with Django REST framework
- Good communication skills

Gaps:
- No cloud experience (AWS/GCP)
- Limited leadership experience

Risks:
- May require ramp-up time

Summary:
Solid candidate with strong Python skills but limited cloud exposure.
"""

    def test_extracts_strengths(self):
        reranker = _make_reranker_no_llm()
        items = reranker.extract_section(self.SAMPLE_LLM_OUTPUT, "strengths", 3)
        assert len(items) == 3
        assert any("Python" in item for item in items)

    def test_extracts_gaps(self):
        reranker = _make_reranker_no_llm()
        items = reranker.extract_section(self.SAMPLE_LLM_OUTPUT, "gaps", 3)
        assert len(items) == 2

    def test_extracts_risks(self):
        reranker = _make_reranker_no_llm()
        items = reranker.extract_section(self.SAMPLE_LLM_OUTPUT, "risks", 5)
        assert len(items) == 1

    def test_max_items_respected(self):
        reranker = _make_reranker_no_llm()
        items = reranker.extract_section(self.SAMPLE_LLM_OUTPUT, "strengths", 1)
        assert len(items) == 1

    def test_missing_section_returns_empty(self):
        reranker = _make_reranker_no_llm()
        items = reranker.extract_section(self.SAMPLE_LLM_OUTPUT, "certifications", 5)
        assert items == []

    def test_preserves_original_case(self):
        reranker = _make_reranker_no_llm()
        items = reranker.extract_section(self.SAMPLE_LLM_OUTPUT, "strengths", 3)
        # Items should NOT be all lowercase
        assert any(c.isupper() for item in items for c in item)


# ---------------------------------------------------------------------------
# extract_summary
# ---------------------------------------------------------------------------

class TestExtractSummary:
    def test_extracts_summary_text(self):
        reranker = _make_reranker_no_llm()
        text = "Strengths: ...\nGaps: ...\nSummary: Great candidate with strong Python skills."
        summary = reranker.extract_summary(text)
        assert "Python" in summary

    def test_returns_default_when_no_summary(self):
        reranker = _make_reranker_no_llm()
        summary = reranker.extract_summary("No summary marker here")
        assert summary == "No summary marker here"

    def test_strips_newlines(self):
        reranker = _make_reranker_no_llm()
        text = "Summary:\nThis is the summary.\n"
        summary = reranker.extract_summary(text)
        assert "\n" not in summary


# ---------------------------------------------------------------------------
# re_rank_candidates
# ---------------------------------------------------------------------------

class TestReRankCandidates:
    def test_returns_empty_for_empty_input(self):
        reranker = _make_reranker_no_llm()
        result = reranker.re_rank_candidates([], _make_query())
        assert result == []

    def test_results_sorted_by_fit_score_descending(self):
        reranker = _make_reranker_no_llm()
        candidates = [
            {"candidate_id": "c1", "name": "Alice", "skills": ["Python", "Django"],
             "experience": 5, "text": "Python developer"},
            {"candidate_id": "c2", "name": "Bob", "skills": ["Java"],
             "experience": 2, "text": "Java developer"},
        ]
        results = reranker.re_rank_candidates(candidates, _make_query())
        scores = [r.fit_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_candidate_evaluation_objects(self):
        reranker = _make_reranker_no_llm()
        candidates = [
            {"candidate_id": "c1", "name": "Alice", "skills": ["Python"],
             "experience": 3, "text": "Python developer"},
        ]
        results = reranker.re_rank_candidates(candidates, _make_query())
        assert all(isinstance(r, CandidateEvaluation) for r in results)

    def test_accepts_resume_objects_directly(self):
        reranker = _make_reranker_no_llm()
        resume = _make_resume(candidate_id="c1", skills=["Python"])
        results = reranker.re_rank_candidates([resume], _make_query())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# _get_jd_skill_tokens
# ---------------------------------------------------------------------------

class TestGetJdSkillTokens:
    def test_returns_required_skills_when_present(self):
        reranker = _make_reranker_no_llm()
        query = _make_query(required_skills=["Python", "Django", "SQL"])
        tokens = reranker._get_jd_skill_tokens(query)
        assert tokens == ["python", "django", "sql"]

    def test_fallback_to_text_parsing_when_no_skills(self):
        reranker = _make_reranker_no_llm()
        query = SearchQuery(title="Dev", text="Required skills: Python, Django, SQL", required_skills=[])
        tokens = reranker._get_jd_skill_tokens(query)
        assert len(tokens) > 0
