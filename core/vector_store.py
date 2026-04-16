"""
Pinecone vector database manager for semantic search.
Handles document embedding, indexing, and similarity search.
"""

import sys
sys.path.append(".")
import time
from typing import Dict, Any, List
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec       
from utils.embeddings import get_embeddings
from utils.config import (
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
)
from utils.utils import get_logger, is_quota_error

logger = get_logger(__name__)

class VectorStore:
    def __init__(self):
        self.client = None
        self.index = None
        self.embeddings = None
        self._ready = None

    def initialize(self) -> bool:
        try:
            if not PINECONE_API_KEY:
                logger.warning("PINECONE API Key not set. Vector store disabled.")
                return False
            
            self.client = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone client initialized.")

            if not self.ensure_index():
                logger.error("Failed to ensure Pinecone index exists.")
                return False

            self.embeddings = get_embeddings()
            if not self.embeddings:
                logger.error("Failed to initialize embeddings for Pinecone.")
                return False
            
            self.index = self.client.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone index configured.")
            self._ready = True
            return True
        except Exception as e:
            logger.error(f"Error initializing Pinecone vector store: {e}")
            self._ready = False
            return False
        
    def ensure_index(self) -> bool:
        try:
            if not self.client:
                logger.error("Pinecone client not initialized.")
                return False
            
            existing_indexes = [index.name for index in self.client.list_indexes()]
            if PINECONE_INDEX_NAME in existing_indexes:
                index_info = self.client.describe_index(PINECONE_INDEX_NAME)
                
                if PINECONE_DIMENSION != index_info.dimension:
                    logger.error("Mismatch in index dimensions")
                    self.client.delete_index(PINECONE_INDEX_NAME)
                    logger.info("Deleted existing index due to dimension mismatch.")
                    time.sleep(5)
                else:
                    logger.info("Pinecone index already exists with correct dimensions.")
                    return True
            else:
                logger.info("Pinecone index does not exist. Creating new index.")

            self.client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info("Pinecone index created successfully.")
            time.sleep(5)  # Wait for index to be ready
            return True
        except Exception as e:
            logger.error(f"Error ensuring Pinecone index: {e}")
            return False
        
    def is_ready(self) -> bool:
        return self._ready

    def add_resumes(self, resumes: List[Document]) -> bool:
        if not self.is_ready():
            logger.error("Pinecone vector store not ready.")
            return False
        try:
            vectors = []
            for i, resume in enumerate(resumes):
                # Pinecone rejects None values — strip them from metadata
                metadata = {k: v for k, v in resume.metadata.items() if v is not None}
                page_content = resume.page_content
                embedding = self.embeddings.embed_documents([page_content])[0]

                vectors.append(
                    {
                        "values": embedding,
                        "metadata": metadata,
                        "id": metadata.get("candidate_id", "unknown"),
                    }
                )

            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            if is_quota_error(e):
                logger.error("Pinecone quota exceeded.")
            else:
                logger.error(f"Error adding resumes to Pinecone: {e}")
            return False
        
    def search_resumes(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find similar resumes using semantic search with optional metadata filters"""
        if not self.is_ready():
            logger.error("Pinecone vector store not ready.")
            return []
        
        try:
            query_embedding = self.embeddings.embed_documents([query])[0]

            if filters:
                pinecone_filters = self.convert_filters_to_pinecone(filters)
                result = self.index.query(
                    vector = query_embedding,
                    top_k = top_k,
                    filter = pinecone_filters,
                    include_metadata = True
                )
            else:
                result = self.index.query(
                    vector = query_embedding,
                    top_k = top_k,
                    include_metadata = True
                )
            
            if not result or not hasattr(result, 'matches') or not result.matches:
                #result = None or result = {} or result = {"matches": None}
                logger.info("No matches found in Pinecone search.")
                return []
            
            return [
                {
                    'page_content': item.metadata.get('text', '') if item.metadata else '',
                    'metadata': item.metadata,
                    'score': item.score
                }
                for item in result.matches
            ]
        except Exception as e:
            if is_quota_error(e):
                logger.error("Embedding quota exceeded.")
            else:
                logger.error(f"Error searching resumes in Pinecone: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics like vector count and dimensions"""
        if not self._ready or not self.index:
            return {"status": "not_ready"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "ready",
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "metric": stats.metric
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}
        
    def convert_filters_to_pinecone(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Transform basic filters to Pinecone's query filter format"""
        pinecone_filters = {}

        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                pinecone_filters[key] = {"$eq": value}
            elif isinstance(value, list):
                pinecone_filters[key] = {"$in": value}
            elif isinstance(value, dict):
                if "min" in value and "max" in value:
                    pinecone_filters[key] = {
                        "$gte": value["min"],
                        "$lte": value["max"]
                    }
                elif "min" in value:
                    pinecone_filters[key] = {"$gte": value["min"]}
                elif "max" in value:
                    pinecone_filters[key] = {"$lte": value["max"]}

        return pinecone_filters

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    vs = VectorStore()

    print("=== is_ready (before initialize) ===")
    print("Ready:", vs.is_ready())

    print("\n=== convert_filters_to_pinecone ===")
    sample_filters = {
        "location": "New York",
        "skills": ["Python", "SQL"],
        "experience": {"min": 3, "max": 8},
    }
    pinecone_filters = vs.convert_filters_to_pinecone(sample_filters)
    print("Input filters:", sample_filters)
    print("Pinecone format:", pinecone_filters)

    print("\n=== initialize ===")
    success = vs.initialize()
    print("Initialization successful:", success)

    print("\n=== is_ready (after initialize) ===")
    print("Ready:", vs.is_ready())

    print("\n=== get_stats ===")
    stats = vs.get_stats()
    print("Index stats:", stats)

    if vs.is_ready():
        print("\n=== add_resumes ===")
        from langchain.schema import Document
        sample_docs = [
            Document(
                page_content="Alice Johnson — Python developer with 5 years experience. Skills: Python, SQL, AWS.",
                metadata={"candidate_id": "demo_001", "name": "Alice Johnson", "skills": ["Python", "SQL", "AWS"], "experience": 5, "location": "New York"},
            )
        ]
        added = vs.add_resumes(sample_docs)
        print("Resumes added:", added)

        print("\n=== search_resumes ===")
        results = vs.search_resumes("Python developer with SQL", top_k=3)
        print(f"Search returned {len(results)} result(s).")
        for r in results:
            print(f"  candidate_id={r.get('metadata', {}).get('candidate_id')}  score={r.get('score', 0):.4f}")
