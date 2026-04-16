"""
Hybrid search system combining BM25 (lexical) and vector (semantic) search.
Provides comprehensive candidate and job matching capabilities.
"""

import sys
sys.path.append(".")
from typing import List, Dict, Any
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from core.vector_store import VectorStore
from utils.utils import get_logger

logger = get_logger(__name__)

class HybridIndexer:
    """Combines BM25 keyword search with vector semantic search for better results"""

    def __init__(self):
        """Initialize both BM25 and vector search components"""
        self.vector_store = VectorStore()
        self.vector_store.initialize()  # Set up Pinecone vector store
        self.bm25_resumes = None        # BM25 index for resumes
        self.resume_texts = []          # Text content for BM25 resume search
        self.resume_metadata = []       # Metadata parallel to resume_texts (same index)

    def index_resumes(self, resumes: List[Document]) -> bool:
        """Index resumes for both keyword and semantic search"""
        if not resumes:
            return False
        try:
            self.resume_texts = []
            self.resume_metadata = []
            for resume in resumes:
                text = resume.page_content.lower().strip()
                self.resume_texts.append(text)
                self.resume_metadata.append(resume.metadata)

            if self.resume_texts:
                tokenized_text = [text.split() for text in self.resume_texts]
                self.bm25_resumes = BM25Okapi(tokenized_text)
                
            if self.vector_store.is_ready():
                self.vector_store.add_resumes(resumes)

            return True
        except Exception as e:
            logger.error(f"Error indexing resumes: {e}")
            return False
            
    
    def search_resumes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search resumes using both BM25 and vector search, then combine results"""
        if not self.bm25_resumes:
            return []
        try:
            query_tokens = query.lower().split()
            bm25_scores = [float(s) for s in self.bm25_resumes.get_scores(query_tokens)]

            vector_results = []
            if self.vector_store.is_ready():
                vector_results = self.vector_store.search_resumes(query, top_k=top_k*2)
                logger.info(f"Vector search returned {len(vector_results)} results for query: {query[:50]}...")
            return self.combine_results(bm25_scores, vector_results, top_k)
        except Exception as e:
            logger.error(f"Error searching resumes: {e}")
            return []

    
    def combine_results(self, bm25_scores: List[float], vector_results: List[Dict],
                        top_k: int) -> List[Dict[str, Any]]:
        """Merge BM25 keyword and vector semantic results using Reciprocal Rank Fusion (RRF).

        BM25 scores are normalized to [0, 1] before ranking.
        Vector (cosine) scores are already in [0, 1].
        RRF combines ranked lists: rrf_score = 1 / (k + rank), k=60 is a standard constant
        that dampens the impact of high ranks and avoids division-by-zero.
        Candidates appearing in both lists get their RRF scores summed.
        """
        RRF_K = 60

        max_bm25 = max(bm25_scores) if bm25_scores else 1
        if max_bm25 == 0:
            max_bm25 = 1

        bm25_ranked = sorted(
            [
                {
                    'candidate_id': self.resume_metadata[i].get('candidate_id', f'c_{i}'),
                    'text': self.resume_texts[i],
                    'name': self.resume_metadata[i].get('name', f'Candidate {i}'),
                    'skills': self.resume_metadata[i].get('skills', []),
                    'location': self.resume_metadata[i].get('location', 'Unknown'),
                    'experience': self.resume_metadata[i].get('experience'),
                    'bm25_score': bm25_scores[i] / max_bm25,  # Normalize BM25 score to [0, 1]
                    'vector_score': 0.0,
                }
                for i in range(len(bm25_scores)) if i < len(self.resume_texts)
            ],
            key=lambda x: x['bm25_score'],
            reverse=True
        )

        vector_ranked = sorted(
            [
                {
                    'candidate_id': item.get('metadata', {}).get('candidate_id', f'v_{i}'),
                    'text': item.get('page_content') or '',
                    'name': item.get('metadata', {}).get('name', f'Candidate {i}'),
                    'skills': item.get('metadata', {}).get('skills', []),
                    'location': item.get('metadata', {}).get('location', 'Unknown'),
                    'experience': item.get('metadata', {}).get('experience'),
                    'bm25_score': 0.0,  
                    'vector_score': float(item.get('score', 0.0))
                }
                for i, item in enumerate(vector_results)
            ],
            key=lambda x: x['vector_score'],
            reverse=True
        )

        merged = {}

        for rank, item in enumerate(bm25_ranked):
            candidate_id = item['candidate_id']
            rrf_score = 1/(RRF_K + rank + 1)
            if candidate_id not in merged:
                merged[candidate_id] = {**item, "combined_score": 0.0}
            merged[candidate_id]["combined_score"] += rrf_score
            merged[candidate_id]['bm25_score'] = item['bm25_score']

        for rank, item in enumerate(vector_ranked):
            candidate_id = item['candidate_id']
            rrf_score = 1/(RRF_K + rank + 1)
            if candidate_id not in merged:
                merged[candidate_id] = {**item, "combined_score": 0.0}
            merged[candidate_id]["combined_score"] += rrf_score
            merged[candidate_id]['vector_score'] = item['vector_score']

            if item['skills']:
                merged[candidate_id]['skills'] = item['skills']
            if item['location'] and item['location'] != 'Unknown':
                merged[candidate_id]['location'] = item['location']
            if item['experience']:
                merged[candidate_id]['experience'] = item['experience']
            if item['name']:
                merged[candidate_id]['name'] = item['name']

        all_sorted = sorted(merged.values(), key=lambda x: x['combined_score'], reverse=True)
        return all_sorted[:top_k]
        
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get status of BM25 and vector search components"""
        return {
            'resumes_ready': bool(self.bm25_resumes),
            'vector_store_ready': self.vector_store.is_ready(),
            'hybrid_ready': bool(self.bm25_resumes) and self.vector_store.is_ready()
        }

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from langchain.schema import Document

    indexer = HybridIndexer()

    print("=== get_index_stats (before indexing) ===")
    print(indexer.get_index_stats())

    sample_docs = [
        Document(
            page_content="Alice Johnson. Python developer with 5 years of experience in Python, SQL, and AWS.",
            metadata={"candidate_id": "c_001", "name": "Alice Johnson", "skills": ["Python", "SQL", "AWS"], "experience": 5, "location": "New York"},
        ),
        Document(
            page_content="Bob Smith. Java engineer specialising in Spring Boot and microservices. 3 years experience.",
            metadata={"candidate_id": "c_002", "name": "Bob Smith", "skills": ["Java", "Spring", "Microservices"], "experience": 3, "location": "San Francisco"},
        ),
        Document(
            page_content="Carol Lee. Data scientist with expertise in Python, TensorFlow and machine learning. 7 years experience.",
            metadata={"candidate_id": "c_003", "name": "Carol Lee", "skills": ["Python", "TensorFlow", "ML"], "experience": 7, "location": "Austin"},
        ),
    ]

    print("\n=== index_resumes ===")
    success = indexer.index_resumes(sample_docs)
    print("Indexing successful:", success)

    print("\n=== get_index_stats (after indexing) ===")
    print(indexer.get_index_stats())

    print("\n=== combine_results (direct call with mock scores) ===")
    mock_bm25_scores = [0.8, 0.3, 0.5]   # one score per doc
    mock_vector_results = [
        {"metadata": {"candidate_id": "c_001", "name": "Alice Johnson", "skills": ["Python", "SQL"], "location": "New York", "experience": 5}, "page_content": "", "score": 0.92},
        {"metadata": {"candidate_id": "c_003", "name": "Carol Lee",    "skills": ["Python", "ML"],  "location": "Austin",    "experience": 7}, "page_content": "", "score": 0.75},
    ]
    combined = indexer.combine_results(mock_bm25_scores, mock_vector_results, top_k=3)
    print(f"Combined {len(combined)} result(s) via RRF:")
    for r in combined:
        print(f"  {r['name']:15s}  bm25={r['bm25_score']:.2f}  vector={r['vector_score']:.2f}  combined={r['combined_score']:.4f}")

    print("\n=== search_resumes ===")
    results = indexer.search_resumes("Python developer with SQL", top_k=2)
    print(f"Search returned {len(results)} result(s):")
    for r in results:
        print(f"  {r['name']:15s}  combined_score={r['combined_score']:.4f}  skills={r['skills']}")
