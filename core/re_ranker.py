"""
AI-powered candidate evaluation and ranking system.
Uses LLM to assess candidate fit and provide detailed feedback.
"""

import sys
sys.path.append(".")
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import Resume, SearchQuery, CandidateEvaluation
from utils.utils import get_logger

logger = get_logger(__name__)

class ReRanker:
    """LLM-powered candidate evaluation with detailed fit analysis"""
    
    def __init__(self):
        """Initialize LLM for candidate evaluation or fall back to rule-based"""
        self.llm = None
        try:
            if not GOOGLE_API_KEY:
                return
                
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3  # Low temperature for consistent evaluation
            )
        except Exception as e:
            self.llm = None

    # ------------------------------------------------------------------ #
    #  Private helpers for scoring                                        #
    # ------------------------------------------------------------------ #

    def _get_jd_skill_tokens(self, jd: SearchQuery) -> List[str]:
        """Extract an ordered list of skill tokens from the query.

        Skills that appear earlier in `required_skills` are considered
        higher priority and will receive a larger match weight in scoring.
        Falls back to comma-separated parsing of the query text when
        `required_skills` is empty.
        """
        if jd.required_skills:
            return [s.lower().strip() for s in jd.required_skills if s.strip()]

        # Best-effort extraction from free-text query
        text = (jd.text or "").lower()
        parts: List[str] = []
        for line in text.splitlines():
            if ":" in line and ("skill" in line or "required" in line):
                parts += [p.strip() for p in line.split(":", 1)[1].split(",")]
            elif "," in line:
                parts += [p.strip() for p in line.split(",")]
        return [p for p in (parts or [w for w in text.split(",") if w.strip()]) if p]

    def _skill_match_weight(self, text: str, jd_skills: List[str]) -> float:
        """Return a positional match weight in [0, 1] for a strength/gap text item.

        Skills that appear earlier in the JD's required_skills list are
        given a higher weight: weight = (n - idx) / n * length_factor.
        This rewards candidates who match the most important skills first.
        """
        if not jd_skills:
            return 0.0
        t = text.lower()
        n = len(jd_skills)
        best = 0.0
        for idx, skill in enumerate(jd_skills):
            if not skill:
                continue
            tokens = t.split()
            # Match if the skill appears as a substring or shares a token with the text
            if skill in t or any(tok in skill or skill in tok for tok in tokens):
                # Skills earlier in the list (lower idx) get higher weight;
                # longer skill strings (more specific) are rewarded slightly more.
                weight = (n - idx) / n * min(1.0, len(skill) / 20 + 0.1)
                best = max(best, weight)
        return best

    def _aggregate_score(self, items: List[str], max_total: float,
                         jd_skills: List[str], is_gap: bool = False) -> float:
        """Compute a weighted aggregate score for a list of strengths or gaps.

        Each item is weighted by:
        - Positional weight: items listed first by the LLM carry more weight.
        - Skill-match weight: items that reference high-priority JD skills score higher.
        - Experience bonus/penalty: items mentioning 'experience', 'years', or 'senior'
          add a bonus to strengths and a penalty to gaps.

        Returns a value in [0, max_total].
        """
        if not items:
            return 0.0
        total = 0.0
        for i, item in enumerate(items):
            # Items listed first by the LLM are assumed more significant
            positional_weight = (len(items) - i) / len(items)
            match = self._skill_match_weight(item, jd_skills)
            experience_bonus = 0.2 if any(
                k in item.lower() for k in ("experience", "years", "senior")
            ) else 0.0
            if is_gap:
                # For gaps: skill match amplifies the penalty; experience mention also hurts
                item_score = positional_weight * (0.6 + 0.6 * match - experience_bonus)
            else:
                # For strengths: skill match amplifies the bonus; experience mention helps
                item_score = positional_weight * (0.6 + 0.4 * match + experience_bonus)
            total += item_score
        return (total / len(items)) * max_total

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #

    def evaluate_candidate(self, resume: Resume, jd: SearchQuery) -> CandidateEvaluation:
        """Evaluate a candidate's fit using the LLM, falling back to rule-based scoring.

        Scoring formula:
            fit_score = base(50) + strength_score(max 30) - gap_penalty(max 40)
            clamped to [0, 100].

        The LLM produces up to 3 strengths, 3 gaps, and any risks.
        Each list is then scored via `_aggregate_score`, which weighs items
        by their position and by how closely they match the required skills.
        """
        if not self.llm:
            return self.simple_evaluation(resume, jd)

        try:
            prompt = f"""
            Analyze this candidate for the job:

            JOB: {jd}
            CANDIDATE: {resume.text}

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
            resp = self.llm.invoke(prompt)
            txt = resp.content

            strengths = self.extract_section(txt, "strengths", 3)
            gaps = self.extract_section(txt, "gaps", 3)
            risks = self.extract_section(txt, "risks", 5)
            summary = self.extract_summary(txt)

            jd_skills = self._get_jd_skill_tokens(jd)

            # Base score is 50 (neutral starting point)
            base = 50.0
            # Strengths can add up to 30 points
            strength_score = self._aggregate_score(strengths, 30.0, jd_skills, is_gap=False)
            # Gaps can subtract up to 40 points
            gap_penalty = self._aggregate_score(gaps, 40.0, jd_skills, is_gap=True)

            score = int(max(0, min(100, round(base + strength_score - gap_penalty))))
            return CandidateEvaluation(
                candidate_id=resume.candidate_id,
                fit_score=score,
                strengths=strengths,
                gaps=gaps,
                risks=risks,
                summary=summary,
            )

        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return self.simple_evaluation(resume, jd)

    def extract_section(self, text: str, section_name: str, max_items: int) -> List[str]:
        """Parse an LLM response and return bullet-point items for a named section.

        Strategy:
        - Scan lines until a line containing `section_name` is found (case-insensitive).
        - Collect lines that start with common bullet markers (-, •, *, or 1./2./3.).
        - Stop when the next section header is detected.
        - Original casing is preserved in the returned items.
        """
        items: List[str] = []
        section_headers = {'strengths:', 'gaps:', 'risks:', 'summary:'}
        in_section = False

        for line in text.split('\n'):
            lower = line.strip().lower()

            # Detect section start
            if section_name in lower:
                in_section = True
                continue

            # Detect section end (another header begins)
            if in_section and any(lower.startswith(h) for h in section_headers):
                break

            # Collect bullet items — preserve original case
            if in_section and line.strip().startswith(('-', '•', '*', '1', '2', '3')):
                clean_item = line.strip().lstrip('-•*1234567890. ').strip()
                if clean_item and len(items) < max_items:
                    items.append(clean_item)

        return items

    def extract_summary(self, text: str) -> str:
        """Extract the summary section from the LLM evaluation response."""
        summary = text.split('Summary:')[-1].replace("\n", "").strip()
        return summary if summary else "Evaluation completed"

    def simple_evaluation(self, resume: Resume, jd: SearchQuery) -> CandidateEvaluation:
        """Rule-based fallback evaluation used when the LLM is unavailable.

        Scoring formula:
            fit_score = 50 + (20 * n_strengths) - (15 * n_gaps)
            clamped to [0, 100].

        Checks performed:
        - Experience: does the candidate meet the query's implied requirements?
        - Skills: does the candidate have any of the required skills listed in the query?
        """
        strengths: List[str] = []
        gaps: List[str] = []

        # Check experience if available on both sides
        if resume.experience is not None and jd.required_skills:
            # We don't have min_experience in SearchQuery; skip experience check
            pass

        # Check skill overlap
        if resume.skills and jd.required_skills:
            required_lower = [r.lower() for r in jd.required_skills]
            matching = [s for s in resume.skills if s.lower() in required_lower]
            if matching:
                strengths.append(f"Has required skills: {', '.join(matching)}")
            else:
                gaps.append("Missing required skills")
        elif resume.skills:
            # No specific skills required; presence of any skills is a mild positive
            strengths.append(f"Has {len(resume.skills)} listed skills")

        # fit_score = 50 + 20 per strength - 15 per gap, clamped to [0, 100]
        fit_score = max(0, min(100, 50 + len(strengths) * 20 - len(gaps) * 15))

        return CandidateEvaluation(
            candidate_id=resume.candidate_id,
            fit_score=fit_score,
            strengths=strengths[:3],
            gaps=gaps[:3],
            risks=[],
            summary=f"Rule-based evaluation: {len(strengths)} strengths, {len(gaps)} gaps",
        )

    def re_rank_candidates(self, candidates: List[Dict[str, Any]], jd: SearchQuery) -> List[CandidateEvaluation]:
        """Evaluate and rank all candidates by job fit score"""
        if not candidates:
            return []
        
        resume_objects = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                resume_data = {
                    "candidate_id": candidate.get("candidate_id", "unknown"),
                    "name": candidate.get("name", "Unknown"),
                    "text": candidate.get("page_content") or candidate.get("text") or "",
                    "skills": candidate.get("skills", []),
                    "experience": candidate.get("experience", None)
                }
                resume_objects.append(Resume(**resume_data))
            elif isinstance(candidate, Resume):
                resume_objects.append(candidate)
        
        evaluations = []
        for resume in resume_objects:
            evaluation = self.evaluate_candidate(resume, jd)
            if evaluation:
                evaluations.append(evaluation)
        
        # Sort by score (highest first)
        evaluations.sort(key=lambda x: x.fit_score, reverse=True)
        
        return evaluations

    def is_available(self) -> bool:
        """Check if LLM is available for evaluation"""
        return self.llm is not None

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utils.schemas import Resume, SearchQuery

    reranker = ReRanker()

    sample_resume = Resume(
        candidate_id="c_001",
        name="Alice Johnson",
        text="Senior Python developer with 6 years of experience. Expert in Python, SQL, and AWS.",
        skills=["Python", "SQL", "AWS", "Django"],
        experience=6,
    )
    sample_jd = SearchQuery(
        title="Backend Engineer",
        text="Looking for a Python developer with SQL and AWS experience.",
        required_skills=["Python", "SQL", "AWS"],
    )

    print("=== is_available ===")
    print("LLM available:", reranker.is_available())

    print("\n=== _get_jd_skill_tokens ===")
    tokens = reranker._get_jd_skill_tokens(sample_jd)
    print("JD skill tokens:", tokens)

    print("\n=== _skill_match_weight ===")
    weight = reranker._skill_match_weight("Expert in Python and AWS", tokens)
    print("Match weight for 'Expert in Python and AWS':", round(weight, 4))

    print("\n=== _aggregate_score (strengths) ===")
    strengths_list = ["Strong Python expertise", "AWS certified", "SQL database experience"]
    score = reranker._aggregate_score(strengths_list, max_total=30.0, jd_skills=tokens, is_gap=False)
    print("Aggregate strength score (max 30):", round(score, 2))

    print("\n=== _aggregate_score (gaps) ===")
    gaps_list = ["Missing Docker knowledge", "No cloud infrastructure experience"]
    penalty = reranker._aggregate_score(gaps_list, max_total=40.0, jd_skills=tokens, is_gap=True)
    print("Aggregate gap penalty (max 40):", round(penalty, 2))

    print("\n=== extract_section & extract_summary ===")
    sample_llm_output = """
    Strengths:
    - Strong Python expertise with 6 years of experience
    - AWS certified professional
    - Proficient in SQL databases
    Gaps:
    - No mention of Docker
    - Limited frontend experience
    Risks:
    - May be overqualified
    Summary: Strong candidate with relevant backend skills.
    """
    print("Strengths:", reranker.extract_section(sample_llm_output, "strengths", 3))
    print("Gaps:", reranker.extract_section(sample_llm_output, "gaps", 3))
    print("Summary:", reranker.extract_summary(sample_llm_output))

    print("\n=== simple_evaluation ===")
    evaluation = reranker.simple_evaluation(sample_resume, sample_jd)
    print("Fit score:", evaluation.fit_score)
    print("Strengths:", evaluation.strengths)
    print("Gaps:", evaluation.gaps)
    print("Summary:", evaluation.summary)

    print("\n=== re_rank_candidates ===")
    candidates = [
        {"candidate_id": "c_001", "name": "Alice Johnson", "text": "Senior Python developer.", "skills": ["Python", "AWS", "SQL"], "experience": 6},
        {"candidate_id": "c_002", "name": "Bob Smith",    "text": "Java developer, minimal Python.", "skills": ["Java", "Spring"],  "experience": 3},
    ]
    evaluations = reranker.re_rank_candidates(candidates, sample_jd)
    for ev in evaluations:
        print(f"  {ev.candidate_id} ({ev.candidate_id}): fit_score={ev.fit_score}, summary={ev.summary}")
