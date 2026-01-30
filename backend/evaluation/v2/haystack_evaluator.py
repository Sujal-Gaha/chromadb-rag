"""
Haystack-based RAG pipeline evaluator
Uses official Haystack evaluation components
"""

import time
import pandas as pd
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from sentence_transformers import SentenceTransformer, util

from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    LLMEvaluator,
)

# Add the parent directory to the path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Now import your app modules
from app.rag_pipeline import RAGPipeline
from .data import GOLD_DATA


@dataclass
class HaystackEvaluationResult:
    """Container for Haystack evaluation results"""

    # Performance metrics
    response_time: float
    retrieval_count: int
    difficulty: str

    question: str
    expected_answer: str
    generated_answer: str
    retrieved_docs: List[str]
    expected_context: List[str]

    # Answer evaluation metrics
    exact_match: bool
    answer_similarity: float

    # Document retrieval metrics
    document_map: Optional[float] = None
    document_mrr: Optional[float] = None
    document_recall: Optional[float] = None


class HaystackRAGEvaluator:
    """RAG evaluator using Haystack's official evaluation components for version 0.42"""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        use_llm_evaluator: bool = False,
    ):
        self.rag_pipeline = rag_pipeline

        # Initialize evaluators for Haystack 0.42
        self.answer_exact_match_eval = AnswerExactMatchEvaluator()

        # Document evaluators
        self.document_map_eval = DocumentMAPEvaluator()
        self.document_mrr_eval = DocumentMRREvaluator()
        self.document_recall_eval = DocumentRecallEvaluator()

        # Sentence transformer for similarity fallback
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

        # LLM evaluator for similarity (optional alternative)
        self.use_llm_evaluator = use_llm_evaluator
        if self.use_llm_evaluator:
            try:
                # In Haystack 0.42, LLMEvaluator might require different initialization
                self.llm_evaluator = LLMEvaluator(
                    instructions="Compare the semantic similarity of these two answers. Return a score from 0.0 to 1.0.",
                    inputs=[("reference", List[str])],
                    outputs=["similarity_score"],
                    examples=[
                        {"inputs": {"responses": "Okay"}, "outputs": {"score": 1}},
                        {"inputs": {"responses": "No"}, "outputs": {"score": 0}},
                    ],
                )
            except Exception as e:
                print(f"LLMEvaluator initialization failed: {e}")
                self.use_llm_evaluator = False

    def _extract_doc_names(self, sources: List[Dict]) -> List[str]:
        """Extract document names from source list"""
        doc_names = []
        for source in sources:
            filename = source.get("filename", "")
            if filename:
                doc_names.append(filename)
        return doc_names

    def _create_haystack_documents(self, sources: List[Dict]) -> List[Any]:
        """Create documents in Haystack format"""
        from haystack import Document

        documents = []
        for source in sources:
            content = source.get("content", "")
            meta = {"filename": source.get("filename", "unknown")}
            documents.append(Document(content=content, meta=meta))
        return documents

    def _create_ground_truth_docs(self, expected_files: List[str]) -> List[Any]:
        """Create ground truth documents"""
        from haystack import Document

        docs = []
        for filename in expected_files:
            # Create a document with relevant flag
            doc = Document(
                content=f"Relevant content from {filename}",
                meta={"filename": filename, "relevant": True},
            )
            docs.append(doc)
        return docs

    def _calculate_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using either LLM evaluator or sentence transformers"""
        try:
            if self.use_llm_evaluator:
                result = self.llm_evaluator.run(
                    reference=reference, candidate=candidate
                )
                # Extract score from result
                if hasattr(result, "get"):
                    return float(result.get("similarity_score", 0.0))
                else:
                    return 0.0
            else:
                # Fallback to cosine similarity using sentence transformers
                ref_embedding = self.similarity_model.encode(
                    reference, convert_to_tensor=True
                )
                cand_embedding = self.similarity_model.encode(
                    candidate, convert_to_tensor=True
                )
                return float(util.cos_sim(ref_embedding, cand_embedding).item())
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0

    async def evaluate_single(
        self, gold_item: Dict[str, Any]
    ) -> HaystackEvaluationResult:
        """Evaluate a single question using Haystack evaluators"""
        start_time = time.time()

        try:
            # Run query through pipeline
            result = await self.rag_pipeline.query(gold_item["question"])

            # Extract information
            generated_answer = result.get("reply", "")
            sources = result.get("sources", [])
            retrieved_docs = self._extract_doc_names(sources)

            # Handle response time
            elapsed_time = result.get("elapsed_time", "0")
            try:
                response_time = float(elapsed_time.replace("s", ""))
            except:
                response_time = 0.0

            retrieval_count = result.get("retrieved_documents", 0)

            # Create document lists for evaluation
            retrieved_document_list = self._create_haystack_documents(sources)
            ground_truth_docs = self._create_ground_truth_docs(
                gold_item["expected_context"]
            )

            # For Haystack 0.42 - Use correct parameter names
            # Evaluate answer exact match
            exact_match_result = self.answer_exact_match_eval.run(
                ground_truth_answers=[gold_item["expected_answer"]],
                predicted_answers=[generated_answer],
            )

            # Calculate answer similarity using our custom method
            similarity_score = self._calculate_similarity(
                gold_item["expected_answer"], generated_answer
            )

            # Evaluate document retrieval metrics for Haystack 0.42
            # Note: Document evaluators in 0.42 expect specific format
            map_result = self.document_map_eval.run(
                ground_truth_documents=[ground_truth_docs],
                retrieved_documents=[retrieved_document_list],
            )

            mrr_result = self.document_mrr_eval.run(
                ground_truth_documents=[ground_truth_docs],
                retrieved_documents=[retrieved_document_list],
            )

            recall_result = self.document_recall_eval.run(
                ground_truth_documents=[ground_truth_docs],
                retrieved_documents=[retrieved_document_list],
            )

            # Extract scores from results
            exact_match_score = (
                exact_match_result.get("score", 0)
                if isinstance(exact_match_result, dict)
                else getattr(exact_match_result, "score", 0)
            )
            map_score = (
                map_result.get("score", 0)
                if isinstance(map_result, dict)
                else getattr(map_result, "score", 0)
            )
            mrr_score = (
                mrr_result.get("score", 0)
                if isinstance(mrr_result, dict)
                else getattr(mrr_result, "score", 0)
            )
            recall_score = (
                recall_result.get("score", 0)
                if isinstance(recall_result, dict)
                else getattr(recall_result, "score", 0)
            )

            return HaystackEvaluationResult(
                question=gold_item["question"],
                expected_answer=gold_item["expected_answer"],
                generated_answer=generated_answer,
                retrieved_docs=retrieved_docs,
                expected_context=gold_item["expected_context"],
                exact_match=bool(exact_match_score == 1.0),
                answer_similarity=similarity_score,
                document_map=float(map_score) if map_score is not None else None,
                document_mrr=float(mrr_score) if mrr_score is not None else None,
                document_recall=(
                    float(recall_score) if recall_score is not None else None
                ),
                response_time=response_time,
                retrieval_count=retrieval_count,
                difficulty=gold_item.get("difficulty", "medium"),
            )

        except Exception as e:
            print(f"Error evaluating question: {gold_item['question']} - {e}")
            import traceback

            traceback.print_exc()

            return HaystackEvaluationResult(
                question=gold_item["question"],
                expected_answer=gold_item["expected_answer"],
                generated_answer=f"ERROR: {str(e)}",
                retrieved_docs=[],
                expected_context=gold_item["expected_context"],
                exact_match=False,
                answer_similarity=0.0,
                response_time=0.0,
                retrieval_count=0,
                difficulty=gold_item.get("difficulty", "medium"),
            )

    async def evaluate_all(
        self, gold_data: Optional[List[Dict]] = None, batch_size: int = 2
    ) -> pd.DataFrame:
        """Evaluate all gold data questions"""
        if gold_data is None:
            gold_data = GOLD_DATA

        results = []

        print(f"Starting Haystack evaluation of {len(gold_data)} questions...")
        print("=" * 80)

        for batch_start in range(0, len(gold_data), batch_size):
            batch_end = min(batch_start + batch_size, len(gold_data))
            batch = gold_data[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(len(gold_data) + batch_size - 1)//batch_size}"
            )

            for i, gold_item in enumerate(batch):
                idx = batch_start + i
                print(
                    f"  [{idx + 1}/{len(gold_data)}]: {gold_item['question'][:60]}..."
                )

                result = await self.evaluate_single(gold_item)
                results.append(result)

                print(
                    f"    Exact: {result.exact_match}, "
                    f"Similarity: {result.answer_similarity:.3f}, "
                    f"Time: {result.response_time:.2f}s"
                )

            if batch_end < len(gold_data):
                await asyncio.sleep(1)

        print("=" * 80)
        print("Haystack evaluation completed!")

        return self._results_to_dataframe(results)

    def _results_to_dataframe(
        self, results: List[HaystackEvaluationResult]
    ) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []

        for result in results:
            row = {
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
                "response_time": result.response_time,
                "retrieval_count": result.retrieval_count,
                "difficulty": result.difficulty,
                "exact_match": float(result.exact_match),
                "answer_similarity": result.answer_similarity,
            }

            if result.document_map is not None:
                row["document_map"] = result.document_map
            if result.document_mrr is not None:
                row["document_mrr"] = result.document_mrr
            if result.document_recall is not None:
                row["document_recall"] = result.document_recall

            data.append(row)

        return pd.DataFrame(data)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall evaluation metrics"""
        metrics = {
            "total_questions": len(df),
            "exact_match_rate": df["exact_match"].mean(),
            "avg_answer_similarity": df["answer_similarity"].mean(),
            "avg_response_time": df["response_time"].mean(),
            "avg_retrieval_count": df["retrieval_count"].mean(),
        }

        if "document_map" in df.columns:
            metrics["mean_average_precision"] = df["document_map"].mean()
        if "document_mrr" in df.columns:
            metrics["mean_reciprocal_rank"] = df["document_mrr"].mean()
        if "document_recall" in df.columns:
            metrics["document_recall"] = df["document_recall"].mean()

        return metrics


# Example usage function
async def run_evaluation():
    """Run evaluation on the RAG pipeline"""
    # Initialize the pipeline
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Create evaluator
    evaluator = HaystackRAGEvaluator(pipeline, use_llm_evaluator=False)

    # Run evaluation
    print("Starting evaluation...")
    results_df = await evaluator.evaluate_all()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(results_df)

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"haystack_evaluation_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")

    return results_df, metrics


if __name__ == "__main__":
    asyncio.run(run_evaluation())
