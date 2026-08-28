"""
Qualitative candidate assessment with Gemini.

Given a candidate and a job description, this produces strengths, gaps, risks,
and a one-line summary — prose that explains *why* a candidate showed up.

It produces no score and does not reorder anything. Ranking belongs to the
retrieval layer (BM25, vector similarity, and their RRF combination); a second
number derived from LLM prose only made those harder to read.
"""

import sys
sys.path.append(".")
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import CandidateEvaluation, Resume, SearchQuery
from utils.utils import get_logger

logger = get_logger(__name__)

_PROMPT = """Analyze this candidate for the job:

JOB: {jd}
CANDIDATE: {resume_text}

Give me:
1. 3 key strengths
2. 3 areas where there are gaps
3. Any risks
4. A brief summary

Format:
Strengths: <strengths>
Gaps: <gaps>
Risks: <risks>
Summary: <summary>
"""

_SECTION_HEADERS = ("strengths:", "gaps:", "risks:", "summary:")


class CandidateEvaluator:
    """Explains candidate fit in words. Falls back to rules without an LLM."""

    def __init__(self):
        self.llm = None
        if not GOOGLE_API_KEY:
            logger.info("GOOGLE_API_KEY not set — using rule-based evaluation.")
            return
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,  # low, so repeated runs read consistently
            )
        except Exception as e:
            logger.warning(f"LLM unavailable, using rule-based evaluation: {e}")

    def is_available(self) -> bool:
        """True when the Gemini path is usable."""
        return self.llm is not None

    # -- public API ---------------------------------------------------------

    def evaluate_candidates(self, candidates: List[Dict[str, Any]],
                            jd: SearchQuery) -> List[CandidateEvaluation]:
        """Evaluate every candidate, preserving input order.

        Order matters: callers pair each evaluation with its candidate by
        position, and candidates arrive already ranked by combined RRF score.
        """
        evaluations = []
        for resume in (self._as_resume(c) for c in candidates):
            if resume is not None:
                evaluations.append(self.evaluate_candidate(resume, jd))
        return evaluations

    def evaluate_candidate(self, resume: Resume, jd: SearchQuery) -> CandidateEvaluation:
        """Assess one candidate, falling back to rules if the LLM fails."""
        if not self.llm:
            return self._rule_based_evaluation(resume, jd)

        try:
            response = self.llm.invoke(
                _PROMPT.format(jd=jd, resume_text=resume.text)
            )
            text = response.content
            return CandidateEvaluation(
                candidate_id=resume.candidate_id,
                strengths=self.extract_section(text, "strengths", 3),
                gaps=self.extract_section(text, "gaps", 3),
                risks=self.extract_section(text, "risks", 5),
                summary=self.extract_summary(text),
            )
        except Exception as e:
            logger.error(f"LLM evaluation failed for {resume.candidate_id}: {e}")
            return self._rule_based_evaluation(resume, jd)

    # -- LLM response parsing ----------------------------------------------

    def extract_section(self, text: str, section_name: str, max_items: int) -> List[str]:
        """Return the bullet points under a named section of the LLM response.

        Scans for the section header, collects bullet lines (-, •, *, or 1./2./3.)
        until the next header, and preserves the original casing.
        """
        items: List[str] = []
        in_section = False

        for line in text.split("\n"):
            stripped = line.strip()
            lower = stripped.lower()

            if section_name in lower:
                in_section = True
                continue
            if in_section and lower.startswith(_SECTION_HEADERS):
                break
            if in_section and stripped.startswith(("-", "•", "*", "1", "2", "3")):
                item = stripped.lstrip("-•*1234567890. ").strip()
                if item:
                    items.append(item)
                    if len(items) == max_items:
                        break

        return items

    def extract_summary(self, text: str) -> str:
        """Return the text after the `Summary:` header."""
        summary = text.split("Summary:")[-1].replace("\n", "").strip()
        return summary or "Evaluation completed"

    # -- fallback -----------------------------------------------------------

    def _rule_based_evaluation(self, resume: Resume, jd: SearchQuery) -> CandidateEvaluation:
        """Skill-overlap summary, used when Gemini is unavailable."""
        strengths: List[str] = []
        gaps: List[str] = []

        if resume.skills and jd.required_skills:
            required = {s.lower() for s in jd.required_skills}
            matching = [s for s in resume.skills if s.lower() in required]
            if matching:
                strengths.append(f"Has required skills: {', '.join(matching)}")
            else:
                gaps.append("Missing required skills")
        elif resume.skills:
            strengths.append(f"Has {len(resume.skills)} listed skills")

        return CandidateEvaluation(
            candidate_id=resume.candidate_id,
            strengths=strengths[:3],
            gaps=gaps[:3],
            risks=[],
            summary=f"Rule-based evaluation: {len(strengths)} strengths, {len(gaps)} gaps",
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _as_resume(candidate: Any) -> Resume | None:
        """Accept either a search-result dict or a Resume."""
        if isinstance(candidate, Resume):
            return candidate
        if isinstance(candidate, dict):
            return Resume(
                candidate_id=candidate.get("candidate_id", "unknown"),
                name=candidate.get("name", "Unknown"),
                text=candidate.get("page_content") or candidate.get("text") or "",
                skills=candidate.get("skills", []),
                experience=candidate.get("experience"),
            )
        return None


if __name__ == "__main__":
    sys.path.append(".")

    evaluator = CandidateEvaluator()
    print("LLM available:", evaluator.is_available())

    sample_output = """
    Strengths:
    - Strong Python expertise with 6 years of experience
    - AWS certified professional
    Gaps:
    - No mention of Docker
    Risks:
    - May be overqualified
    Summary: Strong candidate with relevant backend skills.
    """
    print("\n=== response parsing ===")
    print("Strengths:", evaluator.extract_section(sample_output, "strengths", 3))
    print("Gaps:     ", evaluator.extract_section(sample_output, "gaps", 3))
    print("Summary:  ", evaluator.extract_summary(sample_output))

    print("\n=== evaluate_candidates ===")
    jd = SearchQuery(title="Python Developer", text="Backend role",
                     required_skills=["Python", "SQL", "AWS"])
    candidates = [
        {"candidate_id": "c_001", "name": "Alice Johnson",
         "text": "Senior Python developer.", "skills": ["Python", "AWS", "SQL"],
         "experience": 6},
        {"candidate_id": "c_002", "name": "Bob Smith",
         "text": "Java developer, minimal Python.", "skills": ["Java", "Spring"],
         "experience": 3},
    ]
    for ev in evaluator.evaluate_candidates(candidates, jd):
        print(f"  {ev.candidate_id}: {ev.summary}")
