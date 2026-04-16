"""
RAGAS-based evaluation system for measuring search quality.
Evaluates retrieval and answer generation performance using standard metrics.
"""

import sys
sys.path.append(".")
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.search_router import SearchRouter
from utils.utils import get_logger
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, context_precision, faithfulness, answer_correctness
)

logger = get_logger(__name__)

@dataclass
class RAGEvaluationMetrics:
    """Container for RAGAS evaluation metrics with scoring"""
    answer_relevancy: float     # How relevant the answer is to the question
    context_precision: float    # Precision of retrieved context
    faithfulness: float         # Factual consistency with context
    answer_correctness: float   # Overall answer quality
    overall_score: float        # Weighted average of all metrics
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {
            'answer_relevancy': self.answer_relevancy,
            'context_precision': self.context_precision,
            'faithfulness': self.faithfulness,
            'answer_correctness': self.answer_correctness,
            'overall_score': self.overall_score
        }

class RAGEvaluator:
    """RAGAS-powered search quality evaluator with history tracking"""

    def __init__(self, vector_store, hybrid_indexer):
        """Initialize RAGAS metrics and evaluation tracking"""
        self.ragas_metrics = [
            answer_relevancy, context_precision, faithfulness, answer_correctness
        ]
        self.evaluation_history = []  # Track all evaluations
        self.vector_store = vector_store
        self.hybrid_indexer = hybrid_indexer

        # Use Gemini as RAGAS LLM (instead of default OpenAI)
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self.ragas_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        self.ragas_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def evaluate_search_quality(self, query: str, expected_skills: List[str], 
                              search_mode: str = "deep", top_k: int = 5) -> RAGEvaluationMetrics:
        """Run search and evaluate results using RAGAS quality metrics"""
        
        try:
            # Create search router instance
            search_router = SearchRouter(self.vector_store, self.hybrid_indexer)
            search_result = search_router.search(query, top_k, search_mode=search_mode)
            
            if not search_result.get('results'):
                return self.create_default_metrics()
            
            candidates = search_result['results']
            return self.evaluate_with_ragas(query, candidates, expected_skills)
                
        except Exception as e:
            logger.error(f"Search quality evaluation failed: {e}")
            return self.create_default_metrics()
    
    def create_default_metrics(self) -> RAGEvaluationMetrics:
        """Return zero metrics when search fails or returns no results"""
        return RAGEvaluationMetrics(
            answer_relevancy=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            answer_correctness=0.0,
            overall_score=0.0
        )
    
    def evaluate_with_ragas(self, query: str, candidates: List[Dict], 
                            expected_skills: List[str]) -> RAGEvaluationMetrics:
        """Run RAGAS evaluation on search results and calculate final metrics"""
        
        evaluation_data = self.prepare_ragas_data(query, candidates, expected_skills)
        from datasets import Dataset
        hf_data = Dataset.from_pandas(evaluation_data)
        # RAGAS may return a single float OR a list of per-row scores.
        # Normalise to a single float; treat nan as 0.0.
        def _safe(val):
            if isinstance(val, list):
                nums = [float(v) for v in val
                        if not (isinstance(v, float) and np.isnan(v))]
                return np.mean(nums).item() if nums else 0.0
            v = float(val)
            return 0.0 if np.isnan(v) else v

        # Run evaluation with raise_exceptions=False so individual metric
        # failures (e.g. Gemini output parsing) don't abort the whole run.
        # Failed metrics will be nan, which _safe converts to 0.0.
        try:
            results = evaluate(
                hf_data,
                metrics=self.ragas_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=False,
            )
        except Exception as e:
            logger.warning(f"RAGAS evaluate raised an exception, returning defaults: {e}")
            return self.create_default_metrics()

        results_df = results.to_pandas()

        def _get_metric(name):
            if name in results_df.columns:
                vals = results_df[name].dropna().tolist()
                return _safe(vals) if vals else 0.0
            return 0.0

        metrics = RAGEvaluationMetrics(
            answer_relevancy=_get_metric('answer_relevancy'),
            context_precision=_get_metric('context_precision'),
            faithfulness=_get_metric('faithfulness'),
            answer_correctness=_get_metric('answer_correctness'),
            overall_score=0.0
        )
        
        metrics.overall_score = self.calculate_overall_score(metrics)
        self.store_evaluation(query, metrics)
        
        return metrics
    
    def prepare_ragas_data(self, query: str, candidates: List[Dict], 
                           expected_skills: List[str]) -> pd.DataFrame:
        """Convert search results to RAGAS-compatible DataFrame format"""
        
        data = []
        for candidate in candidates:
            metadata = candidate.get('metadata', {})

            # Use actual resume text as context — faithfulness needs rich
            # context to verify answer claims against.  Fall back to a
            # metadata summary only when page_content is missing.
            page_content = candidate.get('page_content') or candidate.get('text', '')
            if page_content.strip():
                context = page_content[:2000]
            else:
                context_parts = []
                if 'skills' in metadata:
                    context_parts.append(f"Skills: {', '.join(metadata['skills'][:5])}")
                if 'experience' in metadata:
                    context_parts.append(f"Experience: {metadata['experience']} years")
                context = " | ".join(context_parts) if context_parts else "No metadata"

            # Build answer as full declarative sentences so RAGAS can
            # extract verifiable statements for faithfulness scoring.
            skills = metadata.get('skills', [])
            name = metadata.get('name', 'Candidate')
            exp = metadata.get('experience')
            location = metadata.get('location')
            sentences = []
            sentences.append(f"{name} is a candidate matching the query.")
            if skills:
                sentences.append(f"{name} is proficient in {', '.join(skills[:5])}.")
            if exp is not None:
                sentences.append(f"{name} has {exp} years of professional experience.")
            if location and location != 'Unknown':
                sentences.append(f"{name} is located in {location}.")
            generated_answer = " ".join(sentences)

            ground_truth = self.create_ground_truth(skills, expected_skills)

            data.append({
                'question': query,
                'contexts': [context],
                'ground_truth': ground_truth,
                'answer': generated_answer
            })
        
        return pd.DataFrame(data)
    
    def create_ground_truth(self, candidate_skills: List[str], expected_skills: List[str]) -> str:
        """Generate ground truth labels based on skill matching percentage"""
        if not expected_skills:
            return "No skills specified"
         
        matched_skills = [skill for skill in expected_skills 
                         if skill.lower() in [s.lower() for s in candidate_skills]]
        match_percentage = len(matched_skills) / len(expected_skills)
        
        if match_percentage >= 0.8:
            return f"Excellent match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.6:
            return f"Good match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.4:
            return f"Moderate match: {len(matched_skills)}/{len(expected_skills)} skills"
        else:
            return f"Poor match: {len(matched_skills)}/{len(expected_skills)} skills"
    
    def calculate_overall_score(self, metrics: RAGEvaluationMetrics) -> float:
        """Compute weighted average of all RAGAS metrics"""
        weights = {
            'answer_relevancy': 0.30,
            'context_precision': 0.30,
            'faithfulness': 0.20,
            'answer_correctness': 0.20
        }
        
        overall_score = (
            np.mean(metrics.answer_relevancy) * weights['answer_relevancy'] +
            np.mean(metrics.context_precision) * weights['context_precision'] +
            np.mean(metrics.faithfulness) * weights['faithfulness'] +
            np.mean(metrics.answer_correctness) * weights['answer_correctness']
        )
        
        return overall_score
    
    def store_evaluation(self, query: str, metrics: RAGEvaluationMetrics):
        """Save evaluation results to history for later analysis"""
        evaluation_record = {
            'query': query,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics.to_dict()
        }
        
        self.evaluation_history.append(evaluation_record)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['answer_relevancy', 'context_precision', 'faithfulness', 
                      'answer_correctness', 'overall_score']:
            values = [record['metrics'][metric] for record in self.evaluation_history]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_metrics': avg_metrics,
            'recent_evaluations': self.evaluation_history[-5:]
        }
    
    def export_evaluations(self, filename: str = "rag_evaluations.csv") -> bool:
        """Export all evaluation history to CSV file for analysis"""
        if not self.evaluation_history:
            return False

        try:
            export_data = []
            for i, record in enumerate(self.evaluation_history):
                row = {
                    'evaluation_id': i + 1,
                    'query': record['query'],
                    'timestamp': record['timestamp'],
                    **record['metrics']
                }
                export_data.append(row)

            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            return True

        except Exception as e:
            logger.error(f"Failed to export evaluations: {e}")
            return False

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from core.vector_store import VectorStore
    from core.hybrid_indexer import HybridIndexer
    from langchain.schema import Document

    indexer = HybridIndexer()
    sample_docs = [
        Document(
            page_content="Alice Johnson. Python developer, 5 years. Skills: Python, SQL, AWS.",
            metadata={"candidate_id": "c_001", "name": "Alice Johnson", "skills": ["Python", "SQL", "AWS"], "experience": 5, "location": "New York"},
        ),
        Document(
            page_content="Bob Smith. Java engineer with Spring Boot, 3 years.",
            metadata={"candidate_id": "c_002", "name": "Bob Smith", "skills": ["Java", "Spring"], "experience": 3, "location": "San Francisco"},
        ),
    ]
    indexer.index_resumes(sample_docs)
    evaluator = RAGEvaluator(vector_store=indexer.vector_store, hybrid_indexer=indexer)

    print("=== create_default_metrics ===")
    defaults = evaluator.create_default_metrics()
    print("Default metrics:", defaults.to_dict())

    print("\n=== create_ground_truth ===")
    cases = [
        (["Python", "SQL", "AWS"], ["Python", "SQL", "AWS", "Docker"]),  # 75% match
        (["Python"], ["Python", "SQL", "AWS"]),                           # 33% match
        ([], ["Python"]),                                                  # no skills
    ]
    for candidate_skills, expected_skills in cases:
        gt = evaluator.create_ground_truth(candidate_skills, expected_skills)
        print(f"  candidate={candidate_skills}  expected={expected_skills}  → '{gt}'")

    print("\n=== calculate_overall_score ===")
    sample_metrics = RAGEvaluationMetrics(
        answer_relevancy=0.85,
        context_precision=0.70,
        faithfulness=0.90,
        answer_correctness=0.75,
        overall_score=0.0,
    )
    score = evaluator.calculate_overall_score(sample_metrics)
    print("Overall score (weighted avg):", round(score, 4))

    print("\n=== store_evaluation & get_evaluation_summary ===")
    evaluator.store_evaluation("Python developer", sample_metrics)
    evaluator.store_evaluation("Java engineer", evaluator.create_default_metrics())
    summary = evaluator.get_evaluation_summary()
    print("Total evaluations:", summary['total_evaluations'])
    print("Average metrics:")
    for k, v in summary['average_metrics'].items():
        print(f"  {k}: {round(v, 4)}")

    print("\n=== export_evaluations ===")
    exported = evaluator.export_evaluations("rag_evaluations_demo.csv")
    print("Exported to CSV:", exported)

    print("\n=== evaluate_search_quality (live RAGAS call) ===")
    try:
        metrics = evaluator.evaluate_search_quality(
            query="Python developer with SQL",
            expected_skills=["Python", "SQL"],
            search_mode="deep",
            top_k=2,
        )
        print("RAGAS metrics:", metrics.to_dict())
    except Exception as e:
        print(f"Live evaluation skipped ({e})")
