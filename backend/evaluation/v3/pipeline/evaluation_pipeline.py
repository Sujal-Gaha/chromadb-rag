import time
import asyncio

from datetime import datetime
from typing import Any, Optional

from haystack import Document
from haystack.logging import getLogger

from evaluation.v3.base.evaluator_base import (
    BaseEvaluator,
    BatchResult,
    EvaluationResult,
    QuestionResult,
)
from evaluation.v3.base.metrics_store import MetricsStore

# ✓ ADD THIS IMPORT
from evaluation.v3.data.gold_data import DOCUMENT_CONTENTS

import pandas as pd

log = getLogger(__name__)


class EvaluationPipeline:

    def __init__(self, rag_pipeline, config: Optional[dict[str, Any]] = None):
        self.rag_pipeline = rag_pipeline
        self.config = config or {}

        self.evaluators: dict[str, BaseEvaluator] = {}
        self.metrics_store = MetricsStore()

        self.start_time = None
        self.end_time = None

    def register_evaluator(self, evaluator: BaseEvaluator):
        self.evaluators[evaluator.name] = evaluator
        log.info(f"Registered evaluator: {evaluator}")

    def remove_evaluator(self, evaluator_name: str):
        if evaluator_name in self.evaluators:
            self.evaluators[evaluator_name].cleanup()
            del self.evaluators[evaluator_name]

    # ✓ ADD THIS NEW METHOD
    def _create_document_objects(self, filenames: list[str]) -> list[Document]:
        """
        Create Haystack Document objects with actual content from DOCUMENT_CONTENTS.
        This is used for both ground truth and retrieved documents.
        """
        docs = []

        for filename in filenames:
            if not filename or filename == "unknown":
                continue

            # Clean the filename (remove paths if present)
            clean_filename = filename.split("/")[-1] if "/" in filename else filename

            # Get actual content from DOCUMENT_CONTENTS
            content = DOCUMENT_CONTENTS.get(
                clean_filename, f"Content from {clean_filename}"
            )

            doc = Document(
                content=content,
                meta={
                    "filename": clean_filename,
                    "document_type": "ground_truth",
                },
            )
            docs.append(doc)

        log.debug(
            f"Created {len(docs)} Document objects from {len(filenames)} filenames"
        )
        return docs

    async def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        expected_docs: list[str],
        difficulty: str = "medium",
        question_metadata: Optional[dict[str, Any]] = None,
    ) -> QuestionResult:
        start_time = time.time()

        try:
            rag_result = await self.rag_pipeline.query(question)

            generated_answer = rag_result.get("reply", "")
            sources = rag_result.get("sources", [])
            retrieved_docs = []

            for source in sources:
                filename = source.get("filename", "")
                if filename:
                    retrieved_docs.append(filename)

            response_time = time.time() - start_time

            retrieval_count = rag_result.get("retrieved_documents", 0)

            # ============================================================
            # ✓ CREATE DOCUMENT OBJECTS WITH ACTUAL CONTENT
            # ============================================================

            # Create ground truth Document objects with real content
            expected_doc_objects = self._create_document_objects(expected_docs)

            # Create retrieved Document objects with real content
            retrieved_doc_objects = self._create_document_objects(retrieved_docs)

            log.debug(f"Created {len(expected_doc_objects)} expected doc objects")
            log.debug(f"Created {len(retrieved_doc_objects)} retrieved doc objects")

            # ============================================================

            evaluation_results = []

            for evaluator_name, evaluator in self.evaluators.items():
                try:
                    metadata = {
                        "response_time": response_time,
                        "retrieval_count": retrieval_count,
                        "difficulty": difficulty,
                        "question_metadata": question_metadata or {},
                        "rag_result": rag_result,
                    }

                    # ============================================================
                    # ✓ DETERMINE WHAT TO PASS TO EACH EVALUATOR
                    # ============================================================

                    # Check if evaluator needs Document objects or just filenames
                    # RetrievalEvaluator needs Document objects for proper content comparison
                    if evaluator_name == "RetrievalEvaluator":
                        evaluator_results = await evaluator.evaluate_single(
                            question=question,
                            expected_answer=expected_answer,
                            generated_answer=generated_answer,
                            retrieved_docs=retrieved_doc_objects,  # ✓ Pass Document objects
                            expected_docs=expected_doc_objects,  # ✓ Pass Document objects
                            metadata=metadata,
                        )
                    else:
                        # Other evaluators can use filenames
                        evaluator_results = await evaluator.evaluate_single(
                            question=question,
                            expected_answer=expected_answer,
                            generated_answer=generated_answer,
                            retrieved_docs=retrieved_docs,  # Filenames
                            expected_docs=expected_docs,  # Filenames
                            metadata=metadata,
                        )

                    evaluation_results.extend(evaluator_results)

                except Exception as e:
                    log.error(f"Evaluator {evaluator_name} failed: {e}", exc_info=True)

                    evaluation_results.append(
                        EvaluationResult(
                            evaluator_type=evaluator_name,
                            metric_name="evaluation_error",
                            value=0.0,
                            confidence=0.0,
                            metadata={"error": str(e)},
                        )
                    )

            return QuestionResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                retrieved_docs=retrieved_docs,
                expected_docs=expected_docs,
                response_time=response_time,
                retrieval_count=retrieval_count,
                difficulty=difficulty,
                evaluation_results=evaluation_results,
            )

        except Exception as e:
            log.error(
                f"Failed to evaluate question: {question[:50]}... - {e}", exc_info=True
            )

            error_evaluation_result = EvaluationResult(
                evaluator_type="Pipeline",
                metric_name="pipeline_error",
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
            )

            return QuestionResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=f"ERROR: {str(e)}",
                retrieved_docs=[],
                expected_docs=expected_docs,
                response_time=0.0,
                retrieval_count=0,
                difficulty=difficulty,
                evaluation_results=[error_evaluation_result],
            )

    async def evaluate_batch(
        self,
        gold_data: list[dict[str, Any]],
        batch_size: int = 3,
        progress_callback=None,
    ) -> BatchResult:
        self.start_time = datetime.now()
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        total_questions = len(gold_data)
        questions_evaluated = []

        log.info(f"Starting batch evaluation: {batch_id}")
        log.info(f"Total questions: {total_questions}")
        log.info(f"Batch size: {batch_size}")
        log.info(f"Evaluators: {list(self.evaluators.keys())}")
        log.info("=" * 70)

        for batch_start in range(0, total_questions, batch_size):
            batch_end = min(batch_start + batch_size, total_questions)
            current_batch = gold_data[batch_start:batch_end]

            batch_num = batch_start // batch_size + 1
            total_batches = (total_questions + batch_size - 1) // batch_size

            log.info(f"Processing batch {batch_num} / {total_batches}")

            batch_tasks = []

            for item in current_batch:
                task = self.evaluate_single(
                    question=item["question"],
                    expected_answer=item["expected_answer"],
                    expected_docs=item.get("expected_context", []),
                    difficulty=item.get("difficulty", "medium"),
                    question_metadata=item.get("metadata", {}),
                )
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks)
            questions_evaluated.extend(batch_results)

            for i, result in enumerate(batch_results):
                idx = batch_start + i

                similarity = 0.0
                for eval_result in result.evaluation_results:
                    if eval_result.metric_name == "semantic_similarity":
                        similarity = eval_result.value
                        break

                log.info(f"[{idx + 1}/{total_questions}] {result.question[:50]}...")
                log.info(f"Similarity: {similarity:.3f}")
                log.info(f"Time: {result.response_time:.2f}s")

            if progress_callback:
                progress = (batch_end / total_questions) * 100
                progress_callback(progress, batch_end, total_questions)

            if batch_end < total_questions:
                await asyncio.sleep(0.5)

        aggregated_metrics = self._aggregate_metrics(questions_evaluated)

        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds()

        batch_result = BatchResult(
            batch_id=batch_id,
            timestamp=self.start_time.isoformat(),
            pipeline_version=getattr(self.rag_pipeline, "version", "1.0"),
            questions=questions_evaluated,
            aggregated_metrics=aggregated_metrics,
        )

        self.metrics_store.record_evaluation(batch_result)

        log.info("=" * 70)
        log.info(f"Evaluation completed in {total_duration:.1f} seconds")
        log.info(f"Results saved with batch ID: {batch_id}")

        return batch_result

    def _aggregate_metrics(self, questions: list[QuestionResult]) -> dict[str, Any]:
        aggregated = {
            "total_questions": len(questions),
            "total_duration": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time and self.start_time
                else 0
            ),
        }

        metric_values = {}
        for question in questions:
            for result in question.evaluation_results:
                metric_key = f"{result.evaluator_type}_{result.metric_name}"
                if metric_key not in metric_values:
                    metric_values[metric_key] = []

                metric_values[metric_key].append(result.value)

        for metric_name, values in metric_values.items():
            if values:
                aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregated[f"{metric_name}_min"] = min(values)
                aggregated[f"{metric_name}_max"] = max(values)
                aggregated[f"{metric_name}_count"] = len(values)

        successful = sum(
            1
            for q in questions
            if q.generated_answer and not q.generated_answer.startswith("ERROR")
        )

        aggregated["success_rate"] = successful / len(questions) if questions else 0

        response_times = [q.response_time for q in questions if q.response_time > 0]

        if response_times:
            aggregated["avg_response_time"] = sum(response_times) / len(response_times)

        return aggregated

    def results_to_dataframe(self, batch_result: BatchResult) -> pd.DataFrame:
        rows = []

        for question in batch_result.questions:
            row = {
                "question": question.question,
                "expected_answer": (
                    question.expected_answer[:100] + "..."
                    if len(question.expected_answer) > 100
                    else question.expected_answer
                ),
                "generated_answer": (
                    question.generated_answer[:100] + "..."
                    if len(question.generated_answer) > 100
                    else question.generated_answer
                ),
                "retrieved_docs": ", ".join(question.retrieved_docs),
                "expected_docs": ", ".join(question.expected_docs),
                "response_time": question.response_time,
                "retrieval_count": question.retrieval_count,
                "difficulty": question.difficulty,
            }

            for result in question.evaluation_results:
                metric_key = f"{result.evaluator_type}_{result.metric_name}"
                row[metric_key] = result.value

            rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_report(self, batch_results: BatchResult) -> dict[str, Any]:
        df = self.results_to_dataframe(batch_results)

        summary = {
            "batch_id": batch_results.batch_id,
            "timestamp": batch_results.timestamp,
            "total_questions": len(batch_results.questions),
            "aggregated_metrics": batch_results.aggregated_metrics,
            "difficulty_breakdown": {},
            "top_metrics": {},
        }

        if "difficulty" in df.columns:
            difficulty_counts = df["difficulty"].value_counts().to_dict()
            summary["difficulty_breakdown"] = difficulty_counts

        for key, value in batch_results.aggregated_metrics.items():
            if "_mean" in key and "semantic_similarity" in key:
                summary["top_metrics"]["avg_similarity"] = value
            elif "_mean" in key and "exact_match" in key:
                summary["top_metrics"]["avg_exact_match"] = value
            elif "success_rate" in key:
                summary["top_metrics"]["success_rate"] = value
            elif "avg_response_time" in key:
                summary["top_metrics"]["avg_response_time"] = value

        return summary

    def cleanup(self):
        for evaluator in self.evaluators.values():
            evaluator.cleanup()

        log.info("Cleaned up all evaluators")
