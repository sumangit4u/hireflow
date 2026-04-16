"""Unit tests for core/filters.py — no external API calls required."""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.filters import filter_by_skills, filter_by_location, filter_by_experience, apply_filters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(candidate_id: str, skills: list, location: str, experience: float) -> dict:
    """Nested metadata format (as returned by vector store)."""
    return {
        "candidate_id": candidate_id,
        "name": candidate_id,
        "metadata": {
            "skills": skills,
            "location": location,
            "experience": experience,
        },
    }


def _make_flat_candidate(candidate_id: str, skills: list, location: str, experience: float) -> dict:
    """Flat format (as returned by hybrid_indexer.combine_results)."""
    return {
        "candidate_id": candidate_id,
        "name": candidate_id,
        "skills": skills,
        "location": location,
        "experience": experience,
    }


CANDIDATES = [
    _make_candidate("c1", ["Python", "Django", "SQL"], "New York, USA", 5),
    _make_candidate("c2", ["Java", "Spring", "SQL"], "London, UK", 3),
    _make_candidate("c3", ["Python", "FastAPI"], "Remote", 1),
    _make_candidate("c4", ["Python", "Django", "React"], "New York, USA", 8),
    _make_candidate("c5", [], "Berlin, Germany", 2),
]


# ---------------------------------------------------------------------------
# filter_by_skills
# ---------------------------------------------------------------------------

class TestFilterBySkills:
    def test_returns_all_when_no_required_skills(self):
        result = filter_by_skills(CANDIDATES, [])
        assert result == CANDIDATES

    def test_single_skill_match(self):
        result = filter_by_skills(CANDIDATES, ["Python"])
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids
        assert "c3" in ids
        assert "c4" in ids
        assert "c2" not in ids

    def test_multiple_skills_all_required(self):
        result = filter_by_skills(CANDIDATES, ["Python", "Django"])
        ids = [c["candidate_id"] for c in result]
        assert ids == ["c1", "c4"]

    def test_case_insensitive_matching(self):
        result = filter_by_skills(CANDIDATES, ["python", "django"])
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids
        assert "c4" in ids

    def test_no_match_returns_empty(self):
        result = filter_by_skills(CANDIDATES, ["Kotlin"])
        assert result == []

    def test_candidate_with_empty_skills_excluded(self):
        result = filter_by_skills(CANDIDATES, ["Python"])
        ids = [c["candidate_id"] for c in result]
        assert "c5" not in ids


# ---------------------------------------------------------------------------
# filter_by_location
# ---------------------------------------------------------------------------

class TestFilterByLocation:
    def test_returns_all_when_no_target_locations(self):
        result = filter_by_location(CANDIDATES, [])
        assert result == CANDIDATES

    def test_exact_city_match(self):
        result = filter_by_location(CANDIDATES, ["New York"])
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids
        assert "c4" in ids
        assert "c2" not in ids

    def test_partial_location_match(self):
        result = filter_by_location(CANDIDATES, ["UK"])
        ids = [c["candidate_id"] for c in result]
        assert "c2" in ids

    def test_case_insensitive_location(self):
        result = filter_by_location(CANDIDATES, ["new york"])
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids
        assert "c4" in ids

    def test_multiple_target_locations(self):
        result = filter_by_location(CANDIDATES, ["London", "Berlin"])
        ids = [c["candidate_id"] for c in result]
        assert "c2" in ids
        assert "c5" in ids
        assert "c1" not in ids

    def test_no_match_returns_empty(self):
        result = filter_by_location(CANDIDATES, ["Tokyo"])
        assert result == []


# ---------------------------------------------------------------------------
# filter_by_experience
# ---------------------------------------------------------------------------

class TestFilterByExperience:
    def test_returns_all_when_none(self):
        result = filter_by_experience(CANDIDATES, None)
        assert result == CANDIDATES

    def test_minimum_experience_boundary(self):
        result = filter_by_experience(CANDIDATES, 3)
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids   # 5 >= 3
        assert "c2" in ids   # 3 >= 3
        assert "c4" in ids   # 8 >= 3
        assert "c3" not in ids  # 1 < 3
        assert "c5" not in ids  # 2 < 3

    def test_high_threshold_returns_subset(self):
        result = filter_by_experience(CANDIDATES, 6)
        ids = [c["candidate_id"] for c in result]
        assert ids == ["c4"]

    def test_zero_threshold_returns_all_with_experience(self):
        # c5 has experience=2, so all candidates with experience set are returned
        result = filter_by_experience(CANDIDATES, 0)
        assert len(result) == len(CANDIDATES)

    def test_candidate_with_none_experience_excluded(self):
        candidates = [_make_candidate("cx", ["Python"], "NYC", None)]
        result = filter_by_experience(candidates, 1)
        assert result == []


# ---------------------------------------------------------------------------
# apply_filters (combined)
# ---------------------------------------------------------------------------

class TestApplyFilters:
    def test_no_filters_returns_all(self):
        result = apply_filters(CANDIDATES)
        assert result == CANDIDATES

    def test_skills_and_location_combined(self):
        result = apply_filters(CANDIDATES, required_skills=["Python"], target_locations=["New York"])
        ids = [c["candidate_id"] for c in result]
        assert ids == ["c1", "c4"]

    def test_skills_location_experience_combined(self):
        result = apply_filters(
            CANDIDATES,
            required_skills=["Python"],
            target_locations=["New York"],
            min_experience=6,
        )
        ids = [c["candidate_id"] for c in result]
        assert ids == ["c4"]

    def test_overly_strict_filters_return_empty(self):
        result = apply_filters(
            CANDIDATES,
            required_skills=["Python", "Java"],
            min_experience=10,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Flat-dict format (as returned by hybrid_indexer)
# ---------------------------------------------------------------------------

FLAT_CANDIDATES = [
    _make_flat_candidate("c1", ["Python", "Django", "SQL"], "New York, USA", 5),
    _make_flat_candidate("c2", ["Java", "Spring", "SQL"], "London, UK", 3),
    _make_flat_candidate("c3", ["Python", "FastAPI"], "Remote", 1),
]


class TestFlatDictFormat:
    def test_skills_filter_on_flat_dict(self):
        result = filter_by_skills(FLAT_CANDIDATES, ["Python"])
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids
        assert "c3" in ids
        assert "c2" not in ids

    def test_location_filter_on_flat_dict(self):
        result = filter_by_location(FLAT_CANDIDATES, ["London"])
        ids = [c["candidate_id"] for c in result]
        assert ids == ["c2"]

    def test_experience_filter_on_flat_dict(self):
        result = filter_by_experience(FLAT_CANDIDATES, 3)
        ids = [c["candidate_id"] for c in result]
        assert "c1" in ids
        assert "c2" in ids
        assert "c3" not in ids

    def test_apply_filters_on_flat_dict(self):
        result = apply_filters(FLAT_CANDIDATES, required_skills=["Python"], min_experience=3)
        ids = [c["candidate_id"] for c in result]
        assert ids == ["c1"]
