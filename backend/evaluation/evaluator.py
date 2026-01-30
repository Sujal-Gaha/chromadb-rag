"""
Simple RAG pipeline evaluator
"""

import time
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util

from app.rag_pipeline import RAGPipeline
from .data import GOLD_DATA

from app.logger import get_logger

log = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""

    question: str
    expected_answer: str
    generated_answer: str
    retrieved_docs: List[str]
    expected_context: List[str]
    similarity_score: float
    response_time: float
    retrieval_count: int
    context_match: bool
    difficulty: str


class SimpleRAGEvaluator:
    """Simple evaluator for RAG pipeline"""

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(embedding1, embedding2).item()

    def _extract_doc_names(self, sources: List[Dict]) -> List[str]:
        """Extract document names from source list"""
        doc_names = []
        for source in sources:
            filename = source.get("filename", "")
            if filename:
                doc_names.append(filename)
        return doc_names

    def _check_context_match(self, retrieved: List[str], expected: List[str]) -> bool:
        """Check if retrieved documents match expected context"""
        retrieved_set = set(retrieved)
        expected_set = set(expected)

        log.debug(f"Retrieved Set: {retrieved_set}")
        log.debug(f"Expected Set: {expected_set}")

        # Return True if at least one expected document is retrieved
        return len(retrieved_set.intersection(expected_set)) > 0

    async def evaluate_single(self, gold_item: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single question"""
        start_time = time.time()

        # Run query through pipeline
        try:
            result = await self.rag_pipeline.query(gold_item["question"])

            # Extract information
            generated_answer = result.get("reply", "")
            retrieved_docs = self._extract_doc_names(result.get("sources", []))
            response_time = float(result.get("elapsed_time", "0").replace("s", ""))
            retrieval_count = result.get("retrieved_documents", 0)

            # Calculate similarity
            similarity = self._calculate_similarity(
                gold_item["expected_answer"], generated_answer
            )

            # Check context match
            context_match = self._check_context_match(
                retrieved_docs, gold_item["expected_context"]
            )

            return EvaluationResult(
                question=gold_item["question"],
                expected_answer=gold_item["expected_answer"],
                generated_answer=generated_answer,
                retrieved_docs=retrieved_docs,
                expected_context=gold_item["expected_context"],
                similarity_score=similarity,
                response_time=response_time,
                retrieval_count=retrieval_count,
                context_match=context_match,
                difficulty=gold_item.get("difficulty", "medium"),
            )

        except Exception as e:
            print(f"Error evaluating question: {gold_item['question']} - {e}")
            # Return failed result
            return EvaluationResult(
                question=gold_item["question"],
                expected_answer=gold_item["expected_answer"],
                generated_answer="",
                retrieved_docs=[],
                expected_context=gold_item["expected_context"],
                similarity_score=0.0,
                response_time=0.0,
                retrieval_count=0,
                context_match=False,
                difficulty=gold_item.get("difficulty", "medium"),
            )

    async def evaluate_all(
        self, gold_data: Optional[list[Dict]] = None
    ) -> pd.DataFrame:
        """Evaluate all gold data questions"""
        if gold_data is None:
            gold_data = GOLD_DATA

        results = []

        print(f"Starting evaluation of {len(gold_data)} questions...")
        print("=" * 80)

        for i, gold_item in enumerate(gold_data):
            print(
                f"Evaluating [{i+1}/{len(gold_data)}]: {gold_item['question'][:60]}..."
            )

            result = await self.evaluate_single(gold_item)
            results.append(result)

            # Print progress
            print(
                f"  Similarity: {result.similarity_score:.3f}, "
                f"Context Match: {result.context_match}, "
                f"Time: {result.response_time:.2f}s"
            )

        print("=" * 80)
        print("Evaluation completed!")

        # Convert to DataFrame
        df = self._results_to_dataframe(results)
        return df

    def _results_to_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []

        for result in results:
            data.append(
                {
                    "question": result.question,
                    "expected_answer": (
                        result.expected_answer[:100] + "..."
                        if len(result.expected_answer) > 100
                        else result.expected_answer
                    ),
                    "generated_answer": (
                        result.generated_answer[:100] + "..."
                        if len(result.generated_answer) > 100
                        else result.generated_answer
                    ),
                    "retrieved_docs": ", ".join(result.retrieved_docs),
                    "expected_docs": ", ".join(result.expected_context),
                    "similarity_score": result.similarity_score,
                    "response_time": result.response_time,
                    "retrieval_count": result.retrieval_count,
                    "context_match": result.context_match,
                    "difficulty": result.difficulty,
                    "context_match_score": 1.0 if result.context_match else 0.0,
                }
            )

        return pd.DataFrame(data)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall evaluation metrics"""

        # Basic metrics
        metrics = {
            "total_questions": len(df),
            "avg_similarity": df["similarity_score"].mean(),
            "avg_response_time": df["response_time"].mean(),
            "avg_retrieval_count": df["retrieval_count"].mean(),
            "context_match_rate": df["context_match_score"].mean(),
            # By difficulty
            "easy_avg_similarity": df[df["difficulty"] == "easy"][
                "similarity_score"
            ].mean(),
            "medium_avg_similarity": df[df["difficulty"] == "medium"][
                "similarity_score"
            ].mean(),
            "hard_avg_similarity": df[df["difficulty"] == "hard"][
                "similarity_score"
            ].mean(),
            "easy_context_match": df[df["difficulty"] == "easy"][
                "context_match_score"
            ].mean(),
            "medium_context_match": df[df["difficulty"] == "medium"][
                "context_match_score"
            ].mean(),
            "hard_context_match": df[df["difficulty"] == "hard"][
                "context_match_score"
            ].mean(),
            # Threshold-based metrics
            "high_similarity_count": len(df[df["similarity_score"] > 0.7]),
            "medium_similarity_count": len(
                df[(df["similarity_score"] >= 0.4) & (df["similarity_score"] <= 0.7)]
            ),
            "low_similarity_count": len(df[df["similarity_score"] < 0.4]),
            "high_similarity_rate": len(df[df["similarity_score"] > 0.7]) / len(df),
            "context_match_count": len(df[df["context_match"]]),
        }

        # Calculate precision for context retrieval
        total_retrieved = df["retrieval_count"].sum()
        if total_retrieved > 0:
            metrics["avg_precision"] = df["context_match_score"].sum() / total_retrieved
        else:
            metrics["avg_precision"] = 0.0

        return metrics


async def run_evaluation():
    """Main function to run evaluation"""
    # Initialize pipeline
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Create evaluator
    evaluator = SimpleRAGEvaluator(pipeline)

    # Run evaluation
    print("=" * 80)
    print("RAG Pipeline Evaluation")
    print("=" * 80)

    results_df = await evaluator.evaluate_all()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(results_df)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Average Semantic Similarity: {metrics['avg_similarity']:.3f}")
    print(f"Context Match Rate: {metrics['context_match_rate']:.2%}")
    print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
    print(f"Average Retrieved Documents: {metrics['avg_retrieval_count']:.1f}")

    print("\nBy Difficulty Level:")
    print(
        f"  Easy:   Similarity: {metrics['easy_avg_similarity']:.3f}, "
        f"Context Match: {metrics['easy_context_match']:.2%}"
    )
    print(
        f"  Medium: Similarity: {metrics['medium_avg_similarity']:.3f}, "
        f"Context Match: {metrics['medium_context_match']:.2%}"
    )
    print(
        f"  Hard:   Similarity: {metrics['hard_avg_similarity']:.3f}, "
        f"Context Match: {metrics['hard_context_match']:.2%}"
    )

    print("\nSimilarity Distribution:")
    print(
        f"  High (>0.7): {metrics['high_similarity_count']} questions "
        f"({metrics['high_similarity_rate']:.1%})"
    )
    print(f"  Medium (0.4-0.7): {metrics['medium_similarity_count']} questions")
    print(f"  Low (<0.4): {metrics['low_similarity_count']} questions")

    return results_df, metrics
