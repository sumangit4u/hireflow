"""HireFlow - Clean Architecture Implementation"""

import streamlit as st
from pathlib import Path
import sys
import numpy as np
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from core.re_ranker import ReRanker
from core.ingestion import load_resumes
from utils.utils import load_pdf
from core.parsing import ResumeParser
from utils.schemas import SearchQuery, Resume
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from core.memory_rag import MemoryRAG

# Data directory
DATA_RESUMES_DIR = project_root / "data" / "resumes"

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
            # Initialize VectorStore
            vector_store = VectorStore()
            vector_store_ready = vector_store.initialize()
            
            # Create LLM
            llm = None
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                )
            except Exception as e:
                st.warning(f"LLM initialization failed: {e}")
            
            # Create core components
            hybrid_indexer = HybridIndexer() if vector_store_ready else None
            resume_parser = ResumeParser()
            # job_parser = JobParser()
            reranker = ReRanker()
            memory_rag = MemoryRAG()
            
            # Auto-index existing resumes using core module.
            # If Pinecone already contains vectors, skip re-uploading to avoid
            # redundant embedding work on every Streamlit cold start.
            # BM25 is always rebuilt from local PDFs because it lives in memory.
            if hybrid_indexer and vector_store_ready:
                try:
                    existing_resumes = load_resumes(str(DATA_RESUMES_DIR))
                    if existing_resumes:
                        pinecone_stats = vector_store.get_stats()
                        already_indexed = pinecone_stats.get("total_vector_count", 0) > 0
                        if already_indexed:
                            # Rebuild BM25 only; skip Pinecone upsert
                            from rank_bm25 import BM25Okapi
                            hybrid_indexer.resume_texts = []
                            hybrid_indexer.resume_metadata = []
                            for r in existing_resumes:
                                if r.page_content.strip():
                                    hybrid_indexer.resume_texts.append(r.page_content.lower())
                                    hybrid_indexer.resume_metadata.append(r.metadata)
                            tokenized = [t.split() for t in hybrid_indexer.resume_texts]
                            hybrid_indexer.bm25_resumes = BM25Okapi(tokenized)
                            st.info(f"BM25 rebuilt for {len(existing_resumes)} resumes (Pinecone already populated)")
                        else:
                            hybrid_indexer.index_resumes(existing_resumes)
                            st.success(f"Indexed {len(existing_resumes)} resumes into Pinecone and BM25")
                except Exception as e:
                    st.warning(f"Auto-indexing failed: {e}")
            
            # Store components
            self._components = {
                'vector_store': vector_store,
                'hybrid_indexer': hybrid_indexer,
                'llm': llm,
                'resume_parser': resume_parser,
                # 'job_parser': job_parser,
                'reranker': reranker,
                'memory_rag': memory_rag,
                'vector_store_ready': vector_store_ready
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
    
    def render_status_sidebar(self):
        """Render system status in sidebar"""
        st.sidebar.header("System Status")
        
        if self.system_manager.is_ready():
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            if hybrid_indexer:
                st.sidebar.write(f"**Resumes Indexed:** {len(hybrid_indexer.resume_texts)}")
            st.sidebar.write("**System:** Ready")
        else:
            st.sidebar.write("**System:** Initializing...")
        
        # Add Memory & Evaluation section
        st.sidebar.markdown("---")
        st.sidebar.write("**Memory & Evaluation:**")
        
        memory_rag = self.system_manager.get_component('memory_rag')
        if memory_rag:
            stats = memory_rag.get_memory_stats()
            st.sidebar.write(f"• Total Interactions: {stats['total_messages']}")
            st.sidebar.write(f"• Searches: {stats['search_count']}")
            st.sidebar.write(f"• Candidate Views: {stats['candidate_views']}")
        
        # Force re-index button — triggers a full Pinecone + BM25 rebuild
        st.sidebar.markdown("---")
        if st.sidebar.button("Force Re-index Resumes", use_container_width=True):
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            if hybrid_indexer:
                try:
                    resumes = load_resumes(str(DATA_RESUMES_DIR))
                    if resumes:
                        hybrid_indexer.index_resumes(resumes)
                        st.sidebar.success(f"Re-indexed {len(resumes)} resumes")
                    else:
                        st.sidebar.warning("No resumes found in data/resumes/")
                except Exception as e:
                    st.sidebar.error(f"Re-index failed: {e}")

        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.write("**Navigation:**")
        if st.sidebar.button("Memory & Evaluation", use_container_width=True):
            st.session_state.page = "memory_eval"
        if st.sidebar.button("Main Page", use_container_width=True):
            st.session_state.page = "main"
    
    def render_memory_evaluation_page(self):
        """Render Memory & Evaluation page"""
        st.header("🧠 Memory & Evaluation Dashboard")
        
        # Memory RAG Section
        st.subheader("📝 Search Memory")
        memory_rag = self.system_manager.get_component('memory_rag')
        
        if memory_rag:
            # Show memory stats
            col1, col2, col3 = st.columns(3)
            stats = memory_rag.get_memory_stats()
            
            with col1:
                st.metric("Total Interactions", stats['total_messages'])
            with col2:
                st.metric("Searches", stats['search_count'])
            with col3:
                st.metric("Candidate Views", stats['candidate_views'])
            
            # Show recent search history
            st.subheader("🔍 Recent Search History")
            search_history = memory_rag.get_search_history()
            if search_history:
                for i, query in enumerate(search_history, 1):
                    st.write(f"{i}. **{query}**")
            else:
                st.info("No search history yet. Perform some searches to see them here!")
            
            # Show recent interactions
            st.subheader("👥 Recent Interactions")
            messages = memory_rag.memory.chat_memory.messages[-10:]  # Last 10 messages
            for msg in messages:
                if hasattr(msg, 'content'):
                    if "Search:" in msg.content:
                        st.write(f"🔍 **{msg.content}**")
                    elif "Viewed candidate:" in msg.content:
                        st.write(f"👀 **{msg.content}**")
                    else:
                        st.write(f"💬 {msg.content}")
        else:
            st.warning("Memory RAG not available")
        
        # RAG Evaluation Section
        st.subheader("📊 RAG Quality Evaluation")
        st.info("Evaluate the quality of your search results using RAGAS metrics")
        
        # Simple evaluation form
        with st.form("evaluation_form"):
            eval_query = st.text_input("Query to Evaluate", placeholder="e.g., Senior Accountant with QuickBooks")
            expected_skills = st.text_area("Expected Skills (one per line)", placeholder="QuickBooks\nAccounting\nExcel")
            eval_top_k = st.slider("Number of Results to Evaluate", 3, 10, 5)
            
            if st.form_submit_button("Evaluate Search Quality"):
                if eval_query and expected_skills:
                    self._run_evaluation(eval_query, expected_skills.split('\n'), eval_top_k)
            else:
                    st.warning("Please provide both query and expected skills")
    
    def _run_evaluation(self, query: str, expected_skills: list, top_k: int):
        """Run RAG evaluation"""
        try:
            from core.evaluator import RAGEvaluator
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            vector_store = self.system_manager.get_component('vector_store')
            evaluator = RAGEvaluator(vector_store, hybrid_indexer)
            st.info("Running evaluation... This may take a moment.")
            
            # Clean up skills list
            skills = [s.strip() for s in expected_skills if s.strip()]
            
            # Run evaluation
            metrics = evaluator.evaluate_search_quality(query, skills, "deep", top_k)
            
            if metrics:
                st.success("Evaluation completed!")
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Answer Relevancy", f"{np.mean(metrics.answer_relevancy):.2f}")
                    st.metric("Context Precision", f"{np.mean(metrics.context_precision):.2f}")
                
                with col2:
                    st.metric("Faithfulness", f"{np.mean(metrics.faithfulness):.2f}")
                    st.metric("Answer Correctness", f"{np.mean(metrics.answer_correctness):.2f}")
                
                st.metric("Overall Score", f"{metrics.overall_score:.2f}", delta=f"{metrics.overall_score - 5:.2f}")
                
                # Interpretation
                if metrics.overall_score >= 8:
                    st.success("Excellent search quality! 🎉")
                elif metrics.overall_score >= 6:
                    st.info("Good search quality. Room for improvement.")
                else:
                    st.warning("Search quality needs improvement. Consider refining your query or adding more relevant resumes.")
            else:
                st.warning("Evaluation returned no results. Check your query and available data.")
                
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.info("This might be due to missing dependencies or configuration issues.")
    
    def _handle_resume_upload(self, resume_files):
        """Handle resume upload using core modules"""
        try:
            resume_parser = self.system_manager.get_component('resume_parser')
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')

            if not all([resume_parser, hybrid_indexer]):
                st.error("System not ready for resume processing")
                return
            
            st.info(f"Processing {len(resume_files)} resume(s)...")
            
            processed = 0
            for file in resume_files:
                try:
                    # Use core module to process resume
                    import os
                    os.makedirs(DATA_RESUMES_DIR, exist_ok=True)
                    temp_path = str(DATA_RESUMES_DIR / file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    text = load_pdf(temp_path)
                    if text:
                        # Create proper Document object for indexing
                        from langchain.schema import Document
                        # Use core ResumeParser for metadata extraction
                        candidate_id = f"c_{file.name.replace('.pdf', '')}"
                        parsed_resume = resume_parser.parse_resume(text, candidate_id)
                        resume_doc = Document(
                            page_content=text,
                            metadata={
                                'source': temp_path,
                                'filename': file.name,
                                'candidate_id': f"c_{file.name.replace('.pdf', '')}",
                                'name': file.name.replace('.pdf', '').replace('_', ' ').title(),
                                'location': parsed_resume.get('location', 'N/A'),
                                'experience': parsed_resume.get('experience', 0),
                                'skills': parsed_resume.get('skills', []),
                            }
                        )                        
                        
                        # Use core HybridIndexer with proper Document object
                        indexing_result = hybrid_indexer.index_resumes([resume_doc])
                        
                        if indexing_result:
                            processed += 1
                            st.success(f"{file.name} - Indexed successfully")
                        else:
                            st.warning(f"{file.name} - Indexing failed")
                    else:
                        st.error(f"{file.name} - Could not extract text")
                        
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")
            
            if processed > 0:
                st.success(f"Successfully processed and indexed {processed} resumes!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Resume upload failed: {e}")
    
    def _handle_search(self, job_title: str, job_description: str, required_skills: str,
                       top_k: int, preferred_location: str = "", min_experience: int = 0):
        """Handle candidate search using core modules, then apply post-search filters."""
        from core.filters import apply_filters

        try:
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            reranker = self.system_manager.get_component('reranker')

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

                memory_rag = self.system_manager.get_component('memory_rag')
                if memory_rag and candidates_data:
                    memory_rag.record_search(search_query, len(candidates_data))

                if candidates_data:
                    st.success(f"Found {len(candidates_data)} candidates!")
                    if memory_rag:
                        for candidate in candidates_data[:3]:
                            memory_rag.record_candidate_view(candidate.get('name', 'Unknown Candidate'))
                    self._display_search_results(candidates_data, job_title or "Position", job_description, reranker)
                else:
                    st.warning("No matching candidates found. Try relaxing the filters.")

        except Exception as e:
            st.error(f"Search failed: {e}")
    
    def _display_search_results(self, candidates_data: list, job_title: str, job_description: str, reranker: ReRanker):
        """Display search results from core module data (optimized reranker call)."""
        st.header(f"Top Matches for: {job_title}")

        # Prepare job description object once
        jd = SearchQuery(
            title=job_title or "Position",
            text=job_description
        )

        # Prepare resumes for top candidates that should be re-ranked/evaluated (top 3)
        resumes_for_evaluation = []
        eval_indices = []  # keep track of which candidate indices we included
        for i, candidate in enumerate(candidates_data[:5]):  # only top 3 as before
            resume_obj = Resume(
                candidate_id=candidate.get('candidate_id', f"unknown_{i}"),
                name=candidate.get('name', 'Unknown'),
                email="candidate@example.com",
                phone="+1-555-0000",
                experience=candidate.get('experience', 5) or 5,
                skills=candidate.get('skills', []),
                education="Bachelor's Degree",
                text=candidate.get('text', '')
            )
            resumes_for_evaluation.append(resume_obj)
            eval_indices.append(i)

        # Call reranker once with a list (if available)
        evaluations = None
        if reranker and resumes_for_evaluation:
            try:
                # Expect reranker.re_rank_candidates(list_of_resumes, jd) -> list of evaluations
                evaluations = reranker.re_rank_candidates(resumes_for_evaluation, jd)
            except TypeError:
                # If the reranker has a different signature, attempt to call the older single-resume signature
                # but avoid calling it inside the display loop to prevent repeated calls: call serially here.
                evaluations = []
                for r in resumes_for_evaluation:
                    try:
                        evaluations.append(reranker.re_rank_candidates(r, jd))
                    except Exception:
                        evaluations.append(None)
            except Exception:
                evaluations = None

        # Normalize evaluations into a list aligned with resumes_for_evaluation
        normalized_evals = None
        if evaluations is None:
            normalized_evals = [None] * len(resumes_for_evaluation)
        elif isinstance(evaluations, list):
            normalized_evals = evaluations
        else:
            # If a single object returned (for whatever reason), replicate or try to map by candidate_id
            try:
                # If returned is a dict candidate_id -> eval
                if isinstance(evaluations, dict):
                    normalized_evals = []
                    for r in resumes_for_evaluation:
                        normalized_evals.append(evaluations.get(r.candidate_id))
                else:
                    # fallback: put single evaluation for first
                    normalized_evals = [evaluations] + [None] * (len(resumes_for_evaluation) - 1)
            except Exception:
                normalized_evals = [None] * len(resumes_for_evaluation)

        # Display each candidate as before, and attach corresponding evaluation (if present)
        for i, candidate in enumerate(candidates_data):
            with st.expander(f"{candidate.get('name', f'Candidate {i+1}')}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Skills:** {', '.join(candidate.get('skills', []))}")
                    st.write(f"**Experience:** {candidate.get('experience', 'N/A')}")
                    st.write(f"**Location:** {candidate.get('location', 'N/A')}")

                    if candidate.get('text'):
                        st.markdown("**Resume Preview:**")
                        text = candidate.get('text', '')
                        st.text(text[:300] + "..." if len(text) > 300 else text)

                    # Attach AI evaluation for top 3 candidates using pre-fetched evaluations
                    if i in eval_indices and reranker:
                        idx = eval_indices.index(i)
                        eval_obj = normalized_evals[idx] if normalized_evals and idx < len(normalized_evals) else None

                        # Try to get model_dump if present, otherwise assume dict-like
                        eval_data = None
                        if eval_obj is None:
                            st.info("AI evaluation not available")
                        else:
                            try:
                                # If it's a pydantic/model object with model_dump
                                if hasattr(eval_obj, "model_dump"):
                                    eval_data = eval_obj.model_dump()
                                elif hasattr(eval_obj, "dict"):
                                    eval_data = eval_obj.dict()
                                elif isinstance(eval_obj, dict):
                                    eval_data = eval_obj
                                else:
                                    # last resort: attempt attribute access
                                    eval_data = {}
                                    for attr in ("fit_score", "strengths", "gaps", "summary"):
                                        if hasattr(eval_obj, attr):
                                            eval_data[attr] = getattr(eval_obj, attr)
                            except Exception:
                                eval_data = None

                            if eval_data:
                                fit_score = eval_data.get('fit_score', 0)
                                st.markdown("**AI Evaluation:**")
                                if fit_score >= 80:
                                    st.success(f"**AI Score: {fit_score}/100**")
                                elif fit_score >= 60:
                                    st.info(f"**AI Score: {fit_score}/100**")
                                else:
                                    st.warning(f"**AI Score: {fit_score}/100**")

                                strengths = eval_data.get('strengths', [])
                                if strengths:
                                    st.write("**Strengths:**")
                                    for s in strengths[:2]:
                                        st.write(f"• {s}")

                                gaps = eval_data.get('gaps', [])
                                if gaps:
                                    st.write("**Gaps:**")
                                    for g in gaps[:2]:
                                        st.write(f"• {g}")

                                summary = eval_data.get('summary', '')
                                if summary:
                                    st.write("**Summary:**")
                                    st.write(summary)
                            else:
                                st.info("AI evaluation returned no structured data")

                with col2:
                    # Show all three score components for transparency
                    combined = candidate.get('combined_score', 0.0)
                    bm25 = candidate.get('bm25_score', 0.0)
                    vector = candidate.get('vector_score', 0.0)
                    st.metric("Combined (RRF)", f"{combined:.4f}")
                    st.metric("BM25 Score", f"{bm25:.3f}")
                    st.metric("Vector Score", f"{vector:.3f}")



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

    # ensure page exists
    if "page" not in st.session_state:
        st.session_state["page"] = "main"

    # get persistent SystemManager
    system_manager = get_or_create_system_manager()

    # Create UI with dependency injection
    ui = HireFlowUI(system_manager)

    # Layout
    col1, col2 = st.columns([1, 1])
    with col1:
        ui.render_upload_section()
    with col2:
        ui.render_search_section()

    ui.render_status_sidebar()

    if st.session_state.get("page") == "memory_eval":
        ui.render_memory_evaluation_page()


if __name__ == "__main__":
    main()
