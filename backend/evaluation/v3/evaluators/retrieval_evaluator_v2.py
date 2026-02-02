from typing import Any, Optional, Union, List, Set

from haystack import Document
from haystack.logging import getLogger
from evaluation.v3.base.evaluator_base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationType,
)

log = getLogger(__name__)


class RetrievalEvaluatorV2(BaseEvaluator):
    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("RetrievalEvaluator", config or {})
        self.evaluation_type = EvaluationType.RETRIEVAL
        self.relevance_threshold = self.config.get("relevance_threshold", 0.7)
        self.recall_at_ks = [3, 5, 10]
        log.info(
            f"RetrievalEvaluator initialized with relevance_threshold: {self.relevance_threshold}"
        )

    def _extract_original_filename(self, item: Any) -> str:
        """Extract clean filename from Document or string"""
        if isinstance(item, Document):
            return (
                item.meta.get("original_filename")
                or item.meta.get("filename")
                or item.meta.get("file_path")
                or item.meta.get("source")
                or "unknown.txt"
            )
        elif isinstance(item, str):
            return item.split("/")[-1] if "/" in item else item
        return "unknown.txt"

    def _normalize_filenames(
        self, items: Union[List[Document], List[str]]
    ) -> List[str]:
        """Convert whatever input we get into a clean list of filenames"""
        if not items:
            return []
        return [self._extract_original_filename(item) for item in items]

    async def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_docs: Union[List[Document], List[str]],
        expected_docs: Union[List[Document], List[str]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> List[EvaluationResult]:
        results = []

        log.info("=" * 80)
        log.info(f"EVALUATING RETRIEVAL FOR QUESTION: {question}")
        log.info("=" * 80)

        if not expected_docs:
            log.warning(
                "No expected documents provided → skipping retrieval evaluation"
            )
            return results

        expected_files: List[str] = self._normalize_filenames(expected_docs)
        retrieved_files: List[str] = self._normalize_filenames(retrieved_docs)

        expected_set: Set[str] = set(
            f for f in expected_files if f and f != "unknown.txt"
        )
        retrieved_set: Set[str] = set(
            f for f in retrieved_files if f and f != "unknown.txt"
        )

        log.info(f"Expected files: {sorted(expected_set)}")
        log.info(
            f"Retrieved files: {retrieved_files[:15]}{' ...' if len(retrieved_files) > 15 else ''}"
        )
        log.info(
            f"Number expected: {len(expected_set)}   •   Retrieved: {len(retrieved_files)}"
        )

        if not expected_set:
            log.warning("No valid expected filenames after cleaning → skipping metrics")
            return results

        matched_files = expected_set & retrieved_set
        context_match = len(matched_files) / len(expected_set) if expected_set else 0.0

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="context_match",
                value=context_match,
                confidence=1.0,
                metadata={
                    "matched_count": len(matched_files),
                    "expected_count": len(expected_set),
                    "matched_files": sorted(matched_files),
                },
            )
        )

        mrr = self._calculate_mrr(retrieved_files, expected_set)
        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="mean_reciprocal_rank",
                value=mrr,
                confidence=0.95,
                metadata={
                    "first_relevant_rank": self._find_first_relevant_rank(
                        retrieved_files, expected_set
                    )
                },
            )
        )

        for k in self.recall_at_ks:
            recall_k = self._calculate_recall_at_k(retrieved_files, expected_set, k=k)
            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name=f"recall_at_{k}",
                    value=recall_k,
                    confidence=0.9,
                    metadata={"k": k},
                )
            )

        precision_10 = self._calculate_precision_at_k(
            retrieved_files, expected_set, k=10
        )
        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="precision_at_10",
                value=precision_10,
                confidence=0.85,
            )
        )

        log.info("=" * 80)
        log.info("RETRIEVAL EVALUATION SUMMARY")
        log.info("=" * 80)
        log.info(f"Question: {question[:70]}...")
        log.info(
            f"Context Match (any rank) : {context_match:.4f}  ({len(matched_files)}/{len(expected_set)})"
        )
        log.info(f"MRR                       : {mrr:.4f}")
        for k in self.recall_at_ks:
            recall_k = next(
                r.value for r in results if r.metric_name == f"recall_at_{k}"
            )
            log.info(f"Recall@{k:2d}               : {recall_k:.4f}")
        log.info(f"Precision@10              : {precision_10:.4f}")
        if matched_files:
            log.info(f"Matched files             : {sorted(matched_files)}")
        log.info("=" * 60)

        return results

    def _calculate_mrr(self, retrieved: List[str], expected: Set[str]) -> float:
        """Mean Reciprocal Rank — 1 / rank of first relevant document"""
        for rank, fname in enumerate(retrieved, 1):
            if fname in expected:
                return 1.0 / rank
        return 0.0

    def _find_first_relevant_rank(
        self, retrieved: List[str], expected: Set[str]
    ) -> Optional[int]:
        for rank, fname in enumerate(retrieved, 1):
            if fname in expected:
                return rank
        return None

    def _calculate_recall_at_k(
        self, retrieved: List[str], expected: Set[str], k: int
    ) -> float:
        """Recall @ K — fraction of expected files found in top K"""
        if not expected:
            return 1.0
        top_k = set(retrieved[:k])
        found = len(top_k & expected)
        return found / len(expected)

    def _calculate_precision_at_k(
        self, retrieved: List[str], expected: Set[str], k: int
    ) -> float:
        """Precision @ K — fraction of top K that are relevant"""
        if not retrieved:
            return 0.0
        top_k = retrieved[:k]
        relevant_count = sum(1 for f in top_k if f in expected)
        return relevant_count / len(top_k)

    def cleanup(self):
        log.debug("Cleaning up RetrievalEvaluator resources")
        pass
