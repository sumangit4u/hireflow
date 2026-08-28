"""HireFlow - Clean Architecture Implementation"""

import streamlit as st
from pathlib import Path
import sys
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.candidate_evaluator import CandidateEvaluator
from core.hybrid_indexer import HybridIndexer
from core.indexing_service import add_resumes, build_index, index_status, load_index
from core.ingestion import build_resume_document
from core.parsing import ResumeParser
from core.retrieval_eval import DEFAULT_CASES_PATH, load_cases, run_evaluation
from core.vector_store import VectorStore
from utils.schemas import SearchQuery

# Data directory
DATA_RESUMES_DIR = project_root / "data" / "resumes"

# How many top results get an LLM write-up. Each one is a Gemini call, so this
# stays small — the rest of the results still show their retrieval scores.
EVALUATED_COUNT = 5

# Page config
st.set_page_config(page_title="HireFlow", page_icon="🎯", layout="wide")

# ============================================================================
# SYSTEM INITIALIZATION (Clean, minimal)
# ============================================================================

class SystemManager:
    """Manages system components without business logic"""
    
    def __init__(self):
        self._components = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize core system components"""
        if self._initialized:
            return True
            
        try:
            vector_store = VectorStore()
            vector_store_ready = vector_store.initialize()

            hybrid_indexer = HybridIndexer() if vector_store_ready else None
            resume_parser = ResumeParser()
            candidate_evaluator = CandidateEvaluator()

            # Load the index that `index_resumes.py` already built.
            #
            # Startup deliberately does NO indexing: reading every PDF and
            # calling Gemini once per resume is a one-time cost, not a
            # per-launch one. This path only reads a local file and rebuilds
            # BM25 in memory — no PDF parsing, no LLM calls, no Pinecone writes.
            index_loaded = False
            if hybrid_indexer:
                try:
                    index_loaded = load_index(hybrid_indexer)
                except Exception as e:
                    st.warning(f"Could not load the resume index: {e}")


            self._components = {
                'vector_store': vector_store,
                'hybrid_indexer': hybrid_indexer,
                'resume_parser': resume_parser,
                'candidate_evaluator': candidate_evaluator,
                'vector_store_ready': vector_store_ready,
                'index_loaded': index_loaded,
            }
            
            self._initialized = True
            return True
            
        except Exception as e:
            st.error(f"System initialization failed: {e}")
            return False

    def get_component(self, name: str) -> Any:
        """Get component by name"""
        if not self._initialized:
            raise RuntimeError("System not initialized")
        return self._components.get(name)
    
    def is_ready(self) -> bool:
        """Check if system is ready"""
        return self._initialized and self._components.get('vector_store_ready', False)

# ============================================================================
# UI LAYER (Only UI logic, no business logic)
# ============================================================================

class HireFlowUI:
    """Clean UI layer that delegates to core modules"""
    
    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager
    
    def render_upload_section(self):
        """Render resume upload section"""
        st.header("Add Resumes")
        st.info("Upload PDF resumes to search through")
        
        resume_files = st.file_uploader("Select PDF Resumes", type="pdf", accept_multiple_files=True)
        
        if resume_files:
            st.success(f"Selected {len(resume_files)} resume(s)")
            if st.button("Process & Index Resumes", type="primary"):
                self._handle_resume_upload(resume_files)
    
    def render_search_section(self):
        """Render candidate search section"""
        st.header("Search Candidates")
        st.info("Enter job details to find matching candidates")

        with st.form("search_form"):
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Accountant")
            job_description = st.text_area("Job Description", placeholder="Enter detailed job requirements...", height=100)
            required_skills = st.text_area("Required Skills (one per line)", placeholder="Python\nJavaScript\nReact")
            col_loc, col_exp = st.columns(2)
            with col_loc:
                preferred_location = st.text_input("Location Filter (optional)", placeholder="e.g., New York")
            with col_exp:
                min_experience = st.number_input("Min. Experience (years)", min_value=0, value=0, step=1)
            top_k = st.slider("Number of Results", 3, 10, 5)

            submitted = st.form_submit_button("Find Candidates", type="primary")

        if submitted and job_description:
            self._handle_search(job_title, job_description, required_skills, top_k,
                                preferred_location, int(min_experience))
    
    def render_index_warning(self):
        """Tell the user how to build the index when there isn't one yet."""
        try:
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
        except RuntimeError:
            return  # initialization failed; the error is already on screen
        if hybrid_indexer and hybrid_indexer.resume_texts:
            return

        st.warning("No resume index found — search will return nothing.")
        st.markdown(
            "Indexing is a separate one-time step, because it reads every PDF and "
            "calls Gemini once per resume. Build it from the `Hireflow/` directory:"
        )
        st.code("uv run python index_resumes.py", language="bash")
        st.caption(
            "Then restart the app, or use **Rebuild index** in the sidebar. "
            "Later runs only parse resumes that are new or changed."
        )

    def render_status_sidebar(self):
        """Render system status in sidebar"""
        st.sidebar.header("System Status")

        hybrid_indexer = self.system_manager.get_component('hybrid_indexer')

        if self.system_manager.is_ready():
            if hybrid_indexer:
                st.sidebar.write(f"**Resumes Indexed:** {len(hybrid_indexer.resume_texts)}")
            st.sidebar.write("**System:** Ready")
        else:
            st.sidebar.write("**System:** Initializing...")

        # Index provenance — makes it obvious the index is loaded from disk
        # rather than rebuilt on this launch.
        try:
            status = index_status(hybrid_indexer)
            if status["index_exists"]:
                st.sidebar.caption(f"Index built: {status['updated_at'] or 'unknown'}")
                if status["unindexed_pdfs"]:
                    st.sidebar.info(
                        f"{len(status['unindexed_pdfs'])} new PDF(s) in data/resumes/ "
                        "are not indexed yet — use **Index new resumes** below."
                    )
            else:
                st.sidebar.warning("No index on disk. Run `python index_resumes.py`.")
        except Exception as e:
            st.sidebar.caption(f"Index status unavailable: {e}")

        # Indexing controls. Both are explicit, user-triggered actions — nothing
        # here runs on its own at startup.
        st.sidebar.markdown("---")
        st.sidebar.write("**Indexing:**")

        if st.sidebar.button("Index new resumes", use_container_width=True,
                             help="Parse only PDFs that are new or changed since the last run."):
            self._run_indexing(force=False)

        if st.sidebar.button("Force full rebuild", use_container_width=True,
                             help="Re-parse every resume from scratch. One Gemini call per resume."):
            self._run_indexing(force=True)

        st.sidebar.markdown("---")
        st.sidebar.write("**Navigation:**")
        if st.sidebar.button("Search", use_container_width=True):
            st.session_state.page = "main"
        if st.sidebar.button("Retrieval Evaluation", use_container_width=True):
            st.session_state.page = "evaluation"
    
    def _run_indexing(self, force: bool = False):
        """Run the shared indexing service from the sidebar, with live progress."""
        hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
        if not hybrid_indexer:
            st.sidebar.error("Indexer not available — check Pinecone configuration.")
            return

        label = "Rebuilding the whole index" if force else "Indexing new resumes"
        status_box = st.sidebar.status(f"{label}...", expanded=True)
        try:
            report = build_index(
                hybrid_indexer,
                resumes_dir=str(DATA_RESUMES_DIR),
                force=force,
                progress=lambda msg: status_box.write(msg),
            )
        except Exception as e:
            status_box.update(label=f"Indexing failed: {e}", state="error")
            return

        if report.total_indexed == 0:
            status_box.update(label="No resumes found in data/resumes/", state="error")
            return

        status_box.update(label=f"Index ready — {report.summary()}", state="complete")
        if report.failed:
            st.sidebar.warning(f"{len(report.failed)} resume(s) failed — see logs.")
        st.rerun()

    def render_evaluation_page(self):
        """Offline retrieval evaluation: MRR, MAP, and NDCG over fixed test cases."""
        st.header("Retrieval Evaluation")
        st.markdown(
            "Offline evaluation runs a fixed set of queries whose correct answers "
            "are known in advance, then scores the ranking the search produced. "
            "No LLM is involved — this measures **retrieval**, not answer quality."
        )

        hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
        if not hybrid_indexer or not hybrid_indexer.resume_texts:
            st.warning("No index loaded. Build one with `uv run python index_resumes.py`.")
            return

        cases = load_cases()
        if not cases:
            st.error(f"No test cases found at `{DEFAULT_CASES_PATH}`.")
            return

        with st.expander("How the three metrics differ", expanded=False):
            st.markdown(
                "- **MRR** — mean reciprocal rank. Looks only at the *first* correct "
                "result: position 1 scores 1.00, position 2 scores 0.50, position 4 "
                "scores 0.25. Answers \"how soon does the user see something useful?\"\n"
                "- **MAP** — mean average precision. Averages the precision measured "
                "at *every* correct hit, so it rewards getting all the good "
                "candidates up top, not just one.\n"
                "- **NDCG** — normalised discounted cumulative gain. Like MAP, but "
                "discounts each position on a log scale and divides by the best "
                "possible ranking, so 1.00 means \"perfect order\".\n\n"
                "A case counts a candidate as relevant when its indexed skills "
                "contain the skills the case asks for. Retrieval searches resume "
                "*text* while relevance is judged on *extracted skills*, so these "
                "numbers measure whether text search finds the people who really "
                "hold a skill."
            )

        col_k, col_run = st.columns([2, 1])
        with col_k:
            top_k = st.slider("Results scored per query (k)", 3, 20, 10)
        with col_run:
            st.write("")
            run = st.button("Run evaluation", type="primary", use_container_width=True)

        if not run:
            st.caption(f"{len(cases)} test cases ready. Same run from the terminal: "
                       "`uv run python evaluate_retrieval.py --detail`")
            return

        with st.spinner(f"Running {len(cases)} queries..."):
            report = run_evaluation(hybrid_indexer, cases, top_k=top_k)

        summary = report["summary"]
        if not summary["queries"]:
            st.error("No cases could be scored — none had relevant candidates in this index.")
            return

        st.subheader("Overall")
        c1, c2, c3 = st.columns(3)
        c1.metric("MRR", f"{summary['mrr']:.3f}", help="Mean reciprocal rank of the first hit")
        c2.metric("MAP", f"{summary['map']:.3f}", help="Mean average precision across all hits")
        c3.metric("NDCG", f"{summary['ndcg']:.3f}", help="Position-discounted gain vs. a perfect ranking")
        st.caption(f"Averaged over {summary['queries']} queries, scoring the top {report['top_k']}."
                   + (f" {report['skipped']} skipped for having no relevant candidates."
                      if report["skipped"] else ""))

        st.subheader("Per query")
        st.dataframe(
            [
                {
                    "Query": r.query,
                    "Relevant in corpus": r.relevant_count,
                    "First hit at": r.first_hit_rank or "—",
                    "RR": round(r.reciprocal_rank, 3),
                    "AP": round(r.average_precision, 3),
                    "NDCG": round(r.ndcg, 3),
                }
                for r in report["results"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Rankings")
        st.caption("The retrieval scores behind each result — this is what the metrics score.")
        for result in report["results"]:
            hits = sum(result.hits)
            with st.expander(f"{result.query}  —  {hits}/{len(result.hits)} relevant in top {report['top_k']}"):
                if result.note:
                    st.caption(result.note)
                st.dataframe(
                    [
                        {
                            "#": position,
                            "Relevant": "yes" if hit else "no",
                            "Candidate": doc["name"],
                            "BM25": doc["bm25_score"],
                            "Vector": doc["vector_score"],
                            "Combined (RRF)": doc["combined_score"],
                        }
                        for position, (hit, doc) in enumerate(
                            zip(result.hits, result.retrieved), start=1
                        )
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    
    def _handle_resume_upload(self, resume_files):
        """Handle resume upload using core modules"""
        try:
            resume_parser = self.system_manager.get_component('resume_parser')
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')

            if not all([resume_parser, hybrid_indexer]):
                st.error("System not ready for resume processing")
                return
            
            st.info(f"Processing {len(resume_files)} resume(s)...")

            import os
            os.makedirs(DATA_RESUMES_DIR, exist_ok=True)

            new_docs = []
            for file in resume_files:
                try:
                    saved_path = str(DATA_RESUMES_DIR / file.name)
                    with open(saved_path, "wb") as f:
                        f.write(file.getbuffer())

                    # Same parse path as bulk indexing — one Gemini call for
                    # this file only, reusing the already-built parser.
                    resume_doc = build_resume_document(
                        saved_path, filename=file.name, parser=resume_parser
                    )
                    if resume_doc is None:
                        st.error(f"{file.name} - Could not extract text")
                        continue

                    new_docs.append(resume_doc)
                    st.success(f"{file.name} - Parsed")
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")

            if new_docs:
                # Append to the live index AND persist, so the upload survives
                # a restart instead of being re-parsed next launch.
                if add_resumes(hybrid_indexer, new_docs, resumes_dir=str(DATA_RESUMES_DIR)):
                    st.success(f"Indexed and saved {len(new_docs)} resume(s).")
                else:
                    st.warning(
                        f"Indexed {len(new_docs)} resume(s) in memory, but saving to "
                        "disk failed — they will need re-indexing after a restart."
                    )
                st.rerun()

        except Exception as e:
            st.error(f"Resume upload failed: {e}")
    
    def _handle_search(self, job_title: str, job_description: str, required_skills: str,
                       top_k: int, preferred_location: str = "", min_experience: int = 0):
        """Handle candidate search using core modules, then apply post-search filters."""
        from core.filters import apply_filters

        try:
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            evaluator = self.system_manager.get_component('candidate_evaluator')

            if not hybrid_indexer:
                st.error("Search service not available")
                return

            with st.spinner("Searching candidates..."):
                skills_list = [s.strip() for s in required_skills.split('\n') if s.strip()] if required_skills else []
                search_query = f"{job_title} {' '.join(skills_list)} {job_description}"

                candidates_data = hybrid_indexer.search_resumes(search_query, top_k=top_k)

                # Apply post-search filters (skills, location, experience)
                candidates_data = apply_filters(
                    candidates_data,
                    required_skills=skills_list if skills_list else None,
                    target_locations=[preferred_location] if preferred_location.strip() else None,
                    min_experience=min_experience if min_experience > 0 else None,
                )

                if candidates_data:
                    st.success(f"Found {len(candidates_data)} candidates!")
                    self._display_search_results(
                        candidates_data, job_title or "Position", job_description, evaluator
                    )
                else:
                    st.warning("No matching candidates found. Try relaxing the filters.")

        except Exception as e:
            st.error(f"Search failed: {e}")
    
    def _display_search_results(self, candidates_data: list, job_title: str,
                                job_description: str, evaluator: CandidateEvaluator):
        """Render the ranked candidates, with a written assessment of the top few."""
        st.header(f"Top Matches for: {job_title}")

        jd = SearchQuery(title=job_title or "Position", text=job_description)

        # Only the leading candidates get an LLM write-up — one call each.
        evaluated = candidates_data[:EVALUATED_COUNT]
        evaluations = []
        if evaluator and evaluated:
            try:
                evaluations = evaluator.evaluate_candidates(evaluated, jd)
            except Exception as e:
                st.warning(f"Candidate evaluation unavailable: {e}")

        for i, candidate in enumerate(candidates_data):
            with st.expander(f"{candidate.get('name', f'Candidate {i+1}')}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Skills:** {', '.join(candidate.get('skills', []))}")
                    st.write(f"**Experience:** {candidate.get('experience', 'N/A')}")
                    st.write(f"**Location:** {candidate.get('location', 'N/A')}")

                    text = candidate.get('text', '')
                    if text:
                        st.markdown("**Resume Preview:**")
                        st.text(text[:300] + "..." if len(text) > 300 else text)

                    # evaluations is positionally aligned with candidates_data.
                    if i < len(evaluations):
                        self._render_evaluation(evaluations[i])

                with col2:
                    # The three retrieval scores — the only ranking signals.
                    st.metric("Combined (RRF)", f"{candidate.get('combined_score', 0.0):.4f}")
                    st.metric("BM25 Score", f"{candidate.get('bm25_score', 0.0):.3f}")
                    st.metric("Vector Score", f"{candidate.get('vector_score', 0.0):.3f}")

    @staticmethod
    def _render_evaluation(evaluation):
        """Show one candidate's written assessment. No score, by design."""
        if evaluation is None:
            return

        st.markdown("**AI Evaluation:**")
        if evaluation.strengths:
            st.write("**Strengths:**")
            for item in evaluation.strengths[:2]:
                st.write(f"• {item}")
        if evaluation.gaps:
            st.write("**Gaps:**")
            for item in evaluation.gaps[:2]:
                st.write(f"• {item}")
        if evaluation.summary:
            st.write("**Summary:**")
            st.write(evaluation.summary)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
def get_or_create_system_manager():
    # Use session_state to persist across reruns
    if "system_manager" not in st.session_state:
        st.session_state["system_manager"] = SystemManager()
        ok = st.session_state["system_manager"].initialize()
        if not ok:
            st.error("System initialization failed. Please check configuration.")
    return st.session_state["system_manager"]

def main():
    st.title("HireFlow - AI Resume Search")

    st.session_state.setdefault("page", "main")

    ui = HireFlowUI(get_or_create_system_manager())
    ui.render_status_sidebar()

    if st.session_state["page"] == "evaluation":
        ui.render_evaluation_page()
        return

    # Surface the "you haven't indexed yet" case before anything else.
    ui.render_index_warning()

    col1, col2 = st.columns([1, 1])
    with col1:
        ui.render_upload_section()
    with col2:
        ui.render_search_section()


if __name__ == "__main__":
    main()
