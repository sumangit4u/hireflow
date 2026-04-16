"""
Intelligent search routing system using LangChain RunnableBranch.
Routes queries to shallow or deep search based on complexity and LLM analysis.
"""

from typing import Dict, Any, List, Optional
import re
from langchain_core.runnables import RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI
from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger

logger = get_logger(__name__)

class SearchRouter:
    """Routes search queries to appropriate search strategy (shallow vs deep)"""
    
    def __init__(self, vector_store, hybrid_indexer):
        self.hybrid_indexer = hybrid_indexer
        self.vector_store = vector_store
        
        self.llm = None
        if GOOGLE_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.1
                )
            except Exception as e:
                pass
        
        self.search_chain = self.build_search_chain()
    
    def build_search_chain(self):
        """Build the RunnableBranch search chain"""
        
        def shallow_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Fast vector-only search"""
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            
            pass
            
            try:
                if self.vector_store.is_ready():
                    results = self.vector_store.search_resumes(query, top_k, filters)
                    
                    # Format results for consistency
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'candidate_id': result.get('metadata', {}).get('candidate_id', 'unknown'),
                            'name': result.get('metadata', {}).get('name', 'Unknown'),
                            'score': result.get('score', 0.0),
                            'metadata': result.get('metadata', {}),
                            'search_type': 'shallow_vector',
                            'source_query': query
                        })
                    
                    return {
                        "results": formatted_results,
                        "search_type": "shallow_vector",
                        "query": query,
                        "top_k": top_k
                    }
                else:
                    return deep_search(inputs)
                    
            except Exception as e:
                logger.error(f"Shallow search failed: {e}")
                return deep_search(inputs)
        
        def deep_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Comprehensive hybrid search with multi-query and reranking"""
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            jd_context = inputs.get("jd_context", "")
            
            pass
            
            try:
                # Perform hybrid search
                results = self.hybrid_indexer.search_resumes(query, top_k)
                  
                if results and self.llm:
                    # LLM reranking
                    reranked_results = self._llm_rerank(results, query, jd_context, top_k)
                    results = reranked_results

                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'candidate_id': result.get('candidate_id', 'unknown'),
                        'name': result.get('name', 'Unknown'),
                        'score': result.get('combined_rrf_score', result.get('rrf_score', 0.0)),
                        'metadata': result.get('metadata', {}),
                        'search_type': 'deep_hybrid_reranked',
                        'source_query': query,
                        'query_count': result.get('query_count', 1),
                        'llm_reranked': 'llm_reranked' in result
                    })
                
                return {
                    "results": formatted_results,
                    "search_type": "deep_hybrid_reranked",
                    "query": query,
                    "top_k": top_k
                }
                
            except Exception as e:
                logger.error(f"Deep search failed: {e}")
                # Fallback to shallow search
                return shallow_search(inputs)
        
        def route_search(inputs: Dict[str, Any]) -> str:
            """Route to appropriate search strategy based on query complexity and user preference"""
            query = inputs.get("query", "")
            search_mode = inputs.get("search_mode", "auto")
            
            if search_mode == "shallow":
                return "shallow"
            elif search_mode == "deep":
                return "deep"
            else:
                try:
                    routing_prompt = f"""
                    Analyze this search query and determine if it needs deep search or shallow search.
                    
                    Query: "{query}"
                    
                    Choose:
                    - "shallow" for simple, specific queries (e.g., "accountant", "Python developer")
                    - "deep" for complex, nuanced queries (e.g., "senior accountant with QuickBooks and tax experience")
                    
                    Return only "shallow" or "deep".
                    """
                    
                    response = self.llm.invoke(routing_prompt)
                    decision = response.content.strip().lower()
                    
                    if decision in ["shallow", "deep"]:
                        return decision
                    else:
                        return "deep"
                        
                except Exception as e:
                    pass
            
            query_words = len(query.split())
            if query_words <= 3:
                return "shallow"
            else:
                return "deep"
        
        chain = RunnableBranch(
            (lambda x: route_search(x) == "shallow", shallow_search),
            (lambda x: route_search(x) == "deep", deep_search),
            deep_search  # Default fallback
        )
        
        return chain
    
    def _llm_rerank(self, results: List[Dict], query: str, jd_context: str, top_k: int) -> List[Dict]:
        """Use LLM to intelligently rerank search results"""
        if not results or not self.llm:
            return results
        
        try:
            candidates_info = []
            for i, result in enumerate(results):
                candidate_info = f"{i+1}. {result.get('name', 'Unknown')} - "
                candidate_info += f"Score: {result.get('combined_rrf_score', result.get('rrf_score', 0)):.4f}"

                metadata = result.get('metadata', {})
                if metadata == {}:
                    # Add this helper method to the SearchRouter class (place it at class scope, e.g., after _llm_rerank)
                    def _extract_candidate_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
                        """Extract structured metadata from candidate text using an LLM + PydanticOutputParser."""
                        try:
                            # Use LangChain's PydanticOutputParser to extract structured metadata
                            from pydantic import BaseModel, Field
                            from typing import List
                            from langchain.output_parsers import PydanticOutputParser

                            class CandidateMetadata(BaseModel):
                                skills: List[str] = Field(default_factory=list)
                                experience: int = 0
                                location: str = "unknown"

                            parser = PydanticOutputParser(pydantic_object=CandidateMetadata)

                            extract_metadata_prompt = f"""
                            Given the candidate text below, extract key metadata including:
                            1. skills (list of strings)
                            2. experience (integer number of years)
                            3. location (city name or "unknown")

                            Provide the response in the JSON format required by the schema below.

                            Schema instructions:
                            {parser.get_format_instructions()}

                            Candidate Text: \"\"\"{result.get('text', '')}\"\"\"
                            """

                            response = self.llm.invoke(extract_metadata_prompt)
                            parsed = parser.parse(response.content)

                            # parser.parse may return a Pydantic model instance or a dict-like object
                            if hasattr(parsed, "dict"):
                                metadata = parsed.dict()
                            else:
                                metadata = dict(parsed)

                            # Normalize and enforce types
                            skills = metadata.get("skills") or []
                            metadata["skills"] = [str(s).strip() for s in skills if str(s).strip()]

                            try:
                                metadata["experience"] = int(metadata.get("experience") or 0)
                            except Exception:
                                metadata["experience"] = 0

                            loc = metadata.get("location") or ""
                            metadata["location"] = str(loc).strip() if str(loc).strip() else "unknown"

                            return metadata

                        except Exception:
                            # On any parsing/LLM error, fall back to empty metadata
                            return {}
                
                                    # Replacement for $SELECTION_PLACEHOLDER$
                
                    metadata = _extract_candidate_metadata(result)
                    result["metadata"] = metadata
                if 'skills' in metadata:
                    candidate_info += f", Skills: {', '.join(metadata['skills'][:3])}"
                if 'experience' in metadata:
                    candidate_info += f", Experience: {metadata['experience']} years"
                
                candidates_info.append(candidate_info)
            
            rerank_prompt = f"""
            You are an expert recruiter. Given a job search query and candidate results, rerank the candidates by their fit for the role.
            
            Search Query: "{query}"
            Job Context: "{jd_context if jd_context else 'General search'}"
            
            Current Results (ranked by algorithm):
            {chr(10).join(candidates_info)}
            
            Instructions:
            1. Consider the search query and job context
            2. Reorder the candidates by their fit for the role
            3. Return only the numbers in the new order (e.g., "3,1,2,4,5")
            4. Focus on semantic fit, not just scores
            
            Reranked order:
            """
            
            response = self.llm.invoke(rerank_prompt)
            reranked_order = response.content.strip()
            
            try:
                numbers = [int(x) for x in re.findall(r'\d+', reranked_order)]
                
                if len(numbers) == len(results):
                    reranked_results = []
                    for num in numbers:
                        if 1 <= num <= len(results):
                            result = results[num - 1].copy()
                            result['llm_reranked'] = True
                            result['llm_rank'] = len(reranked_results) + 1
                            reranked_results.append(result)
                    
                    return reranked_results
                else:
                    pass
                    
            except Exception as e:
                pass
            
        except Exception as e:
            pass
        
        return results
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None, 
               search_mode: str = "auto", jd_context: str = "") -> Dict[str, Any]:
        """Execute search with routing between shallow and deep strategies"""
        
        inputs = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "search_mode": search_mode,
            "jd_context": jd_context
        }
        
        try:
            result = self.search_chain.invoke(inputs)
            return result
            
        except Exception as e:
            # Fallback to simple search
            return {
                "results": [],
                "search_type": "fallback",
                "query": query,
                "top_k": top_k,
                "error": str(e)
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search router statistics"""
        return {
            "llm_available": self.llm is not None,
            "vector_store_ready": self.vector_store.is_ready(),
            "hybrid_indexer_ready": self.hybrid_indexer.get_index_stats()['hybrid_ready'],
            "search_modes": ["auto", "shallow", "deep"]
        }

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from langchain.schema import Document
    from core.vector_store import VectorStore
    from core.hybrid_indexer import HybridIndexer

    # --- set up dependencies ---
    indexer = HybridIndexer()
    sample_docs = [
        Document(
            page_content="Alice Johnson. Python developer, 5 years. Skills: Python, SQL, AWS.",
            metadata={"candidate_id": "c_001", "name": "Alice Johnson", "skills": ["Python", "SQL", "AWS"], "experience": 5, "location": "New York"},
        ),
        Document(
            page_content="Bob Smith. Java engineer with Spring Boot and microservices, 3 years.",
            metadata={"candidate_id": "c_002", "name": "Bob Smith", "skills": ["Java", "Spring"], "experience": 3, "location": "San Francisco"},
        ),
        Document(
            page_content="Carol Lee. Senior data scientist, Python, TensorFlow, ML, 7 years.",
            metadata={"candidate_id": "c_003", "name": "Carol Lee", "skills": ["Python", "TensorFlow", "ML"], "experience": 7, "location": "Austin"},
        ),
    ]
    indexer.index_resumes(sample_docs)
    router = SearchRouter(vector_store=indexer.vector_store, hybrid_indexer=indexer)

    print("=== get_search_stats ===")
    stats = router.get_search_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== search (mode='shallow') ===")
    result = router.search("Python", top_k=2, search_mode="shallow")
    print(f"Search type: {result['search_type']}  |  Results: {len(result['results'])}")
    for r in result['results']:
        print(f"  {r['name']:15s}  score={r['score']:.4f}")

    print("\n=== search (mode='deep') ===")
    result = router.search("senior Python developer with cloud and SQL experience", top_k=2, search_mode="deep")
    print(f"Search type: {result['search_type']}  |  Results: {len(result['results'])}")
    for r in result['results']:
        print(f"  {r['name']:15s}  score={r['score']:.4f}")

    print("\n=== search (mode='auto', short query → shallow) ===")
    result = router.search("Java", top_k=2, search_mode="auto")
    print(f"Search type: {result['search_type']}  |  Results: {len(result['results'])}")

    print("\n=== search (mode='auto', long query → deep) ===")
    result = router.search("data scientist with machine learning and Python skills", top_k=2, search_mode="auto")
    print(f"Search type: {result['search_type']}  |  Results: {len(result['results'])}")
