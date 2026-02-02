import json
import re

from typing import Any, Optional, Union

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
        self.evaluation_type = EvaluationType.RETRIEVAL

        # Haystack's built-in evaluators
        # These expect specific document formats
        self.map_evaluator = DocumentMAPEvaluator()
        self.mrr_evaluator = DocumentMRREvaluator()
        self.recall_evaluator = DocumentRecallEvaluator()

        self.relevance_threshold = self.config.get("relevance_threshold", 0.7)

        log.info(
            f"RetrievalEvaluator initialized with relevance_threshold: {self.relevance_threshold}"
        )

    def _extract_original_filename(self, doc: Document) -> str:
        return (
            doc.meta.get("original_filename")
            or doc.meta.get("filename")
            or "unknown.txt"
        )

    def _convert_title_to_filename(self, title: str) -> str:
        filename = title.lower()
        filename = filename.replace(" ", "-")
        # Remove apostrophes and other special chars
        filename = re.sub(r"[^\w\-\.]", "", filename)

        if not filename.endswith(".txt"):
            filename += ".txt"

        return filename

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
        log.info("=" * 80)

        if not expected_answer:
            log.warning("No expected answer provided, skipping retrieval evaluation")
            return results

        # Normalize expected documents
        # Test data can provide either Document objects or filenames
        ground_truth_docs: list[Document]
        expected_files: list[str]

        if expected_docs and isinstance(expected_docs[0], Document):
            # Already Document objects
            ground_truth_docs = expected_docs  # type: ignore
            expected_files = [
                self._extract_original_filename(doc) for doc in ground_truth_docs
            ]
        else:
            # Just filenames - need to create Document objects
            expected_files = expected_docs  # type: ignore
            ground_truth_docs = self._create_ground_truth_docs(expected_files)

        # Normalize retrieved documents
        retrieved_haystack_docs: list[Document]
        retrieved_files: list[str]

        if retrieved_docs and isinstance(retrieved_docs[0], Document):
            retrieved_haystack_docs = retrieved_docs  # type: ignore

            retrieved_files = [
                self._extract_original_filename(doc) for doc in retrieved_haystack_docs
            ]
        else:
            # Just filenames
            retrieved_files = retrieved_docs  # type: ignore
            retrieved_haystack_docs = self._create_retrieved_docs(retrieved_files)

        log.info(f"Expected documents (filenames): {expected_files}")
        log.info(f"Retrieved documents (filenames): {retrieved_files}")
        log.info(f"Number of expected docs: {len(ground_truth_docs)}")
        log.info(f"Number of retrieved docs: {len(retrieved_haystack_docs)}")

        if metadata:
            log.debug(f"Question metadata: {json.dumps(metadata, indent=2)}")

        try:
            # Debug logging to see what we're evaluating
            log.debug("GROUND TRUTH DOCUMENTS:")
            for i, doc in enumerate(ground_truth_docs):
                log.debug(f"GT Doc {i}:")
                log.debug(f"Content: {doc.content[:100] if doc.content else ''}...")
                log.debug(f"Meta: {doc.meta}")

            log.debug("RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(retrieved_haystack_docs):
                log.debug(f"Ret Doc {i}:")
                log.debug(f"Content: {doc.content[:100] if doc.content else ''}...")
                log.debug(f"Meta: {doc.meta}")

            # ========================================
            # METRIC 1: Mean Average Precision (MAP)
            # ========================================
            # Measures: How well-ranked are relevant documents?
            # - Considers order: Earlier relevant docs = higher score
            # - Range: 0.0 (worst) to 1.0 (perfect)
            # - Formula: Average of precision values at each relevant doc position
            #
            # Example:
            #   Retrieved: [doc1, doc2, doc3, doc4]
            #   Relevant:  [doc1, doc3]
            #   Precision at doc1: 1/1 = 1.0 (1 relevant in top 1)
            #   Precision at doc3: 2/3 = 0.67 (2 relevant in top 3)
            #   MAP = (1.0 + 0.67) / 2 = 0.835

            log.info("Evaluating Mean Average Precision (MAP)...")
            map_result = self.map_evaluator.run(
                ground_truth_documents=[
                    ground_truth_docs
                ],  # Wrapped in list (batch format)
                retrieved_documents=[retrieved_haystack_docs],
            )
            map_score = map_result.get("score", 0)
            log.info(f"MAP Score: {map_score:.4f}")

            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="mean_average_precision",
                    value=map_score,
                    confidence=0.9,  # How confident are we in this metric?
                )
            )

            # ========================================
            # METRIC 2: Mean Reciprocal Rank (MRR)
            # ========================================
            # Measures: Position of first relevant document
            # - Only cares about first relevant result
            # - Range: 0.0 (no relevant docs) to 1.0 (first result is relevant)
            # - Formula: 1 / rank_of_first_relevant_doc
            #
            # Example:
            #   Retrieved: [irrelevant, irrelevant, RELEVANT, ...]
            #   First relevant at position 3
            #   MRR = 1/3 = 0.333

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

            # ========================================
            # METRIC 3: Document Recall
            # ========================================
            # Measures: What % of relevant docs were found?
            # - Ignores order, only cares about presence
            # - Range: 0.0 (found nothing) to 1.0 (found all relevant)
            # - Formula: relevant_found / total_relevant
            #
            # Example:
            #   Expected: [doc1, doc2, doc3]
            #   Retrieved: [doc1, doc5, doc2, doc9]
            #   Recall = 2/3 = 0.67 (found doc1 and doc2, missed doc3)

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

            # ========================================
            # METRIC 4: Context Match (Custom)
            # ========================================
            # Measures: Exact filename matching
            # - Simpler than Haystack metrics
            # - Based on set intersection of filenames
            # - Range: 0.0 to 1.0
            #
            # Why? Haystack evaluators use document content similarity.
            # We want exact file matching for test validation.

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
                    confidence=1.0,  # Exact match - high confidence
                    metadata={
                        "matched_docs": list(matched_docs),
                        "expected_docs": list(expected_set),
                        "retrieved_docs": list(retrieved_set),
                    },
                )
            )

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

            # Return zero scores on error
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
        docs = []

        log.debug(f"Creating ground truth docs from {len(expected_files)} files")

        for filename in expected_files:
            # Extract basename if it's a path
            clean_filename = filename.split("/")[-1] if "/" in filename else filename

            log.debug(f"Creating GT doc for: {clean_filename}")

            doc = Document(
                content=f"Relevant content from {clean_filename}",
                meta={
                    "filename": clean_filename,
                    "original_filename": clean_filename,
                    "relevant": True,  # Mark as relevant for evaluation
                    "document_type": "ground_truth",
                },
            )
            docs.append(doc)

        log.debug(f"Created {len(docs)} ground truth documents")
        return docs

    def _create_retrieved_docs(self, filenames: list[str]) -> list[Document]:
        docs = []

        log.debug(f"Creating retrieved docs from {len(filenames)} files")

        for filename in filenames:
            if not filename or filename == "unknown":
                log.warning(f"Skipping invalid filename: {filename}")
                continue

            clean_filename = filename.split("/")[-1] if "/" in filename else filename

            log.debug(f"Creating retrieved doc for: {clean_filename}")

            doc = Document(
                content=f"Retrieved content from {clean_filename}",
                meta={
                    "filename": clean_filename,
                    "original_filename": clean_filename,
                    "document_type": "retrieved",
                },
            )
            docs.append(doc)

        log.debug(f"Created {len(docs)} retrieved documents")
        return docs

    def cleanup(self):
        log.debug("Cleaning up RetrievalEvaluator resources")
        pass
