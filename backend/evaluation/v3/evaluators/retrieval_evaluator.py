from typing import Any, Optional, Union
import json

from haystack import Document
from haystack.logging import getLogger
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
)
from evaluation.v3.base.evaluator_base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationType,
)

log = getLogger(__name__)


class RetrievalEvaluator(BaseEvaluator):
    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("RetrievalEvaluator", config or {})
        self.evaluate_type = EvaluationType.RETRIEVAL

        self.map_evaluator = DocumentMAPEvaluator()
        self.mrr_evaluator = DocumentMRREvaluator()
        self.recall_evaluator = DocumentRecallEvaluator()

        self.relevance_threshold = self.config.get("relevance_threshold", 0.7)

        log.info(
            f"RetrievalEvaluator initialized with relevance_threshold: {self.relevance_threshold}"
        )

    async def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_docs: Union[list[Document], list[str]],
        expected_docs: Union[list[Document], list[str]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[EvaluationResult]:

        results = []

        log.info("=" * 80)
        log.info(f"EVALUATING RETRIEVAL FOR QUESTION: {question}")
        log.info(f"{'='*80}")

        if not expected_answer:
            log.warning("No expected answer provided, skipping retrieval evaluation")
            return results

        # ✓ Type narrowing with explicit variable declarations
        ground_truth_docs: list[Document]
        expected_files: list[str]

        if expected_docs and isinstance(expected_docs[0], Document):
            # Already Document objects - use them directly
            ground_truth_docs = expected_docs  # type: ignore
            expected_files = [
                doc.meta.get("filename", "unknown") for doc in ground_truth_docs
            ]
        else:
            # Strings - convert to Document objects (fallback)
            expected_files = expected_docs  # type: ignore
            ground_truth_docs = self._create_ground_truth_docs(expected_files)

        retrieved_haystack_docs: list[Document]
        retrieved_files: list[str]

        if retrieved_docs and isinstance(retrieved_docs[0], Document):
            # Already Document objects - use them directly
            retrieved_haystack_docs = retrieved_docs  # type: ignore
            log.debug(f"Retrieved Haystack Documents: {retrieved_haystack_docs}")
            retrieved_files = [
                doc.meta.get("filename", "unknown") for doc in retrieved_haystack_docs
            ]
        else:
            # Strings - convert to Document objects (fallback)
            retrieved_files = retrieved_docs  # type: ignore
            retrieved_haystack_docs = self._create_retrieved_docs(retrieved_files)

        # Log what we received
        log.info(f"Expected documents (filenames): {expected_files}")
        log.info(f"Retrieved documents (filenames): {retrieved_files}")
        log.info(f"Number of expected docs: {len(ground_truth_docs)}")
        log.info(f"Number of retrieved docs: {len(retrieved_haystack_docs)}")

        if metadata:
            log.debug(f"Question metadata: {json.dumps(metadata, indent=2)}")

        try:
            # Log document details for debugging
            log.debug("\nGROUND TRUTH DOCUMENTS:")
            for i, doc in enumerate(ground_truth_docs):
                log.debug(f"  GT Doc {i}:")
                log.debug(f"    Content: {doc.content[:100] if doc.content else ''}...")
                log.debug(f"    Meta: {doc.meta}")

            log.debug("\nRETRIEVED DOCUMENTS:")
            for i, doc in enumerate(retrieved_haystack_docs):
                log.debug(f"  Ret Doc {i}:")
                log.debug(f"    Content: {doc.content[:100] if doc.content else ''}...")
                log.debug(f"    Meta: {doc.meta}")

            # Evaluate MAP
            log.info("Evaluating Mean Average Precision (MAP)...")
            map_result = self.map_evaluator.run(
                ground_truth_documents=[ground_truth_docs],
                retrieved_documents=[retrieved_haystack_docs],
            )
            map_score = map_result.get("score", 0)
            log.info(f"MAP Score: {map_score:.4f}")

            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="mean_average_precision",
                    value=map_score,
                    confidence=0.9,
                )
            )

            # Evaluate MRR
            log.info("Evaluating Mean Reciprocal Rank (MRR)...")
            mrr_result = self.mrr_evaluator.run(
                ground_truth_documents=[ground_truth_docs],
                retrieved_documents=[retrieved_haystack_docs],
            )
            mrr_score = mrr_result.get("score", 0)
            log.info(f"MRR Score: {mrr_score:.4f}")

            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="mean_reciprocal_rank",
                    value=mrr_score,
                    confidence=0.9,
                )
            )

            # Evaluate Recall
            log.info("Evaluating Document Recall...")
            recall_result = self.recall_evaluator.run(
                ground_truth_documents=[ground_truth_docs],
                retrieved_documents=[retrieved_haystack_docs],
            )
            recall_score = recall_result.get("score", 0)
            log.info(f"Recall Score: {recall_score:.4f}")

            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="document_recall",
                    value=recall_score,
                    confidence=0.9,
                )
            )

            # Calculate context match (simple set intersection)
            log.info("Calculating context match (set intersection)...")
            expected_set = set(expected_files)
            retrieved_set = set(retrieved_files)
            matched_docs = expected_set.intersection(retrieved_set)

            log.info(f"Expected files: {expected_set}")
            log.info(f"Retrieved files: {retrieved_set}")
            log.info(f"Matched files: {matched_docs}")

            if expected_set:
                context_match = len(matched_docs) / len(expected_set)
                log.info(
                    f"Context Match Score: {context_match:.4f} ({len(matched_docs)}/{len(expected_set)})"
                )
            else:
                context_match = 0.0
                log.warning("No expected documents, context_match set to 0.0")

            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="context_match",
                    value=context_match,
                    confidence=1.0,
                    metadata={
                        "matched_docs": list(matched_docs),
                        "expected_docs": list(expected_set),
                        "retrieved_docs": list(retrieved_set),
                    },
                )
            )

            # Summary log
            log.info("=" * 60)
            log.info("RETRIEVAL EVALUATION SUMMARY:")
            log.info("=" * 60)
            log.info(f"Question: {question[:50]}...")
            log.info(f"MAP: {map_score:.4f}")
            log.info(f"MRR: {mrr_score:.4f}")
            log.info(f"Recall: {recall_score:.4f}")
            log.info(f"Context Match: {context_match:.4f}")
            log.info(f"Matched Documents: {list(matched_docs)}")
            log.info("=" * 60)

        except Exception as e:
            log.error(f"Error in retrieval evaluation: {e}", exc_info=True)
            log.error(f"Question: {question}")
            log.error(f"Expected docs: {expected_files}")
            log.error(f"Retrieved docs: {retrieved_files}")

            for metric in [
                "mean_average_precision",
                "mean_reciprocal_rank",
                "document_recall",
                "context_match",
            ]:
                results.append(
                    EvaluationResult(
                        evaluator_type=self.name,
                        metric_name=metric,
                        value=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                    )
                )

        return results

    def _create_ground_truth_docs(self, expected_files: list[str]) -> list[Document]:
        """Create ground truth documents from expected files (fallback for string input)"""
        docs = []

        log.debug(f"Creating ground truth docs from {len(expected_files)} files")

        for filename in expected_files:
            # Extract just the filename without path if present
            clean_filename = filename.split("/")[-1] if "/" in filename else filename

            log.debug(f"  Creating GT doc for: {clean_filename}")

            doc = Document(
                content=f"Relevant content from {clean_filename}",
                meta={
                    "filename": clean_filename,
                    "relevant": True,
                    "document_type": "ground_truth",
                },
            )
            docs.append(doc)

        log.debug(f"Created {len(docs)} ground truth documents")
        return docs

    def _create_retrieved_docs(self, filenames: list[str]) -> list[Document]:
        """Create retrieved documents from filenames (fallback for string input)"""
        docs = []

        log.debug(f"Creating retrieved docs from {len(filenames)} files")

        for filename in filenames:
            if not filename or filename == "unknown":
                log.warning(f"Skipping invalid filename: {filename}")
                continue

            # Extract just the filename without path if present
            clean_filename = filename.split("/")[-1] if "/" in filename else filename

            log.debug(f"  Creating retrieved doc for: {clean_filename}")

            doc = Document(
                content=f"Retrieved content from {clean_filename}",
                meta={"filename": clean_filename, "document_type": "retrieved"},
            )
            docs.append(doc)

        log.debug(f"Created {len(docs)} retrieved documents")
        return docs

    def cleanup(self):
        """Cleanup resources"""
        log.debug("Cleaning up RetrievalEvaluator resources")
        pass
