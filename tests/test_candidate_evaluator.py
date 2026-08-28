"""Unit tests for core/candidate_evaluator.py — no external API calls required."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.candidate_evaluator import CandidateEvaluator
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


def _make_evaluator_no_llm() -> CandidateEvaluator:
    """Return a CandidateEvaluator with the LLM disabled (rule-based path)."""
    with patch("core.candidate_evaluator.GOOGLE_API_KEY", ""):
        evaluator = CandidateEvaluator()
    evaluator.llm = None
    return evaluator


# ---------------------------------------------------------------------------
# _rule_based_evaluation
# ---------------------------------------------------------------------------

class TestSimpleEvaluation:
    def test_returns_candidate_evaluation(self):
        evaluator = _make_evaluator_no_llm()
        result = evaluator._rule_based_evaluation(_make_resume(), _make_query())
        assert isinstance(result, CandidateEvaluation)

    def test_carries_no_numeric_score(self):
        """Ranking is expressed by BM25/vector/RRF, never by a fit score."""
        evaluator = _make_evaluator_no_llm()
        result = evaluator._rule_based_evaluation(_make_resume(), _make_query())
        assert not hasattr(result, "fit_score")
        assert "fit_score" not in CandidateEvaluation.model_fields

    def test_skill_match_reported_as_strength(self):
        evaluator = _make_evaluator_no_llm()
        resume_with_skills = _make_resume(skills=["Python", "Django"])
        result = evaluator._rule_based_evaluation(resume_with_skills, _make_query())
        assert len(result.strengths) > 0
        assert not result.gaps

    def test_skill_mismatch_reported_as_gap(self):
        evaluator = _make_evaluator_no_llm()
        resume_no_skills = _make_resume(skills=["Kotlin", "Android"])
        result = evaluator._rule_based_evaluation(resume_no_skills, _make_query())
        assert len(result.gaps) > 0

    def test_no_required_skills_still_reports_strengths(self):
        evaluator = _make_evaluator_no_llm()
        query_no_skills = SearchQuery(title="Any Role", text="Open role", required_skills=[])
        resume = _make_resume(skills=["Python"])
        result = evaluator._rule_based_evaluation(resume, query_no_skills)
        # Has skills but nothing to match against — reported, not scored
        assert len(result.strengths) > 0
        assert not result.gaps

    def test_correct_candidate_id_in_result(self):
        evaluator = _make_evaluator_no_llm()
        resume = _make_resume(candidate_id="test_id_123")
        result = evaluator._rule_based_evaluation(resume, _make_query())
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
        evaluator = _make_evaluator_no_llm()
        items = evaluator.extract_section(self.SAMPLE_LLM_OUTPUT, "strengths", 3)
        assert len(items) == 3
        assert any("Python" in item for item in items)

    def test_extracts_gaps(self):
        evaluator = _make_evaluator_no_llm()
        items = evaluator.extract_section(self.SAMPLE_LLM_OUTPUT, "gaps", 3)
        assert len(items) == 2

    def test_extracts_risks(self):
        evaluator = _make_evaluator_no_llm()
        items = evaluator.extract_section(self.SAMPLE_LLM_OUTPUT, "risks", 5)
        assert len(items) == 1

    def test_max_items_respected(self):
        evaluator = _make_evaluator_no_llm()
        items = evaluator.extract_section(self.SAMPLE_LLM_OUTPUT, "strengths", 1)
        assert len(items) == 1

    def test_missing_section_returns_empty(self):
        evaluator = _make_evaluator_no_llm()
        items = evaluator.extract_section(self.SAMPLE_LLM_OUTPUT, "certifications", 5)
        assert items == []

    def test_preserves_original_case(self):
        evaluator = _make_evaluator_no_llm()
        items = evaluator.extract_section(self.SAMPLE_LLM_OUTPUT, "strengths", 3)
        # Items should NOT be all lowercase
        assert any(c.isupper() for item in items for c in item)


# ---------------------------------------------------------------------------
# extract_summary
# ---------------------------------------------------------------------------

class TestExtractSummary:
    def test_extracts_summary_text(self):
        evaluator = _make_evaluator_no_llm()
        text = "Strengths: ...\nGaps: ...\nSummary: Great candidate with strong Python skills."
        summary = evaluator.extract_summary(text)
        assert "Python" in summary

    def test_returns_default_when_no_summary(self):
        evaluator = _make_evaluator_no_llm()
        summary = evaluator.extract_summary("No summary marker here")
        assert summary == "No summary marker here"

    def test_strips_newlines(self):
        evaluator = _make_evaluator_no_llm()
        text = "Summary:\nThis is the summary.\n"
        summary = evaluator.extract_summary(text)
        assert "\n" not in summary


# ---------------------------------------------------------------------------
# evaluate_candidates
# ---------------------------------------------------------------------------

class TestReRankCandidates:
    def test_returns_empty_for_empty_input(self):
        evaluator = _make_evaluator_no_llm()
        result = evaluator.evaluate_candidates([], _make_query())
        assert result == []

    def test_preserves_input_order(self):
        """Callers pair evaluations with candidates by position.

        The candidates arrive already ranked by combined RRF score, so
        reordering here would attach each evaluation to the wrong candidate.
        """
        evaluator = _make_evaluator_no_llm()
        candidates = [
            {"candidate_id": "c1", "name": "Alice", "skills": ["Java"],
             "experience": 2, "text": "Java developer"},
            {"candidate_id": "c2", "name": "Bob", "skills": ["Python", "Django"],
             "experience": 5, "text": "Python developer"},
        ]
        results = evaluator.evaluate_candidates(candidates, _make_query())
        assert [r.candidate_id for r in results] == ["c1", "c2"]

    def test_returns_candidate_evaluation_objects(self):
        evaluator = _make_evaluator_no_llm()
        candidates = [
            {"candidate_id": "c1", "name": "Alice", "skills": ["Python"],
             "experience": 3, "text": "Python developer"},
        ]
        results = evaluator.evaluate_candidates(candidates, _make_query())
        assert all(isinstance(r, CandidateEvaluation) for r in results)

    def test_accepts_resume_objects_directly(self):
        evaluator = _make_evaluator_no_llm()
        resume = _make_resume(candidate_id="c1", skills=["Python"])
        results = evaluator.evaluate_candidates([resume], _make_query())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# The scoring helpers (_get_jd_skill_tokens, _skill_match_weight,
# _aggregate_score) existed only to compute fit_score and were removed with it.
# ---------------------------------------------------------------------------

class TestNoScoringMachinery:
    def test_scoring_helpers_are_gone(self):
        evaluator = _make_evaluator_no_llm()
        for removed in ("_get_jd_skill_tokens", "_skill_match_weight", "_aggregate_score"):
            assert not hasattr(evaluator, removed)
