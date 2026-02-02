import difflib

from typing import Any, Optional, Union


from haystack import Document
from haystack.logging import getLogger
from evaluation.v3.base.evaluator_base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationType,
)

from haystack.components.evaluators import AnswerExactMatchEvaluator

log = getLogger(__name__)


class AnswerEvaluator(BaseEvaluator):
    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("AnswerEvaluator", config or {})
        self.evaluation_type = EvaluationType.ANSWER_QUALITY

        self.exact_match_evaluator = AnswerExactMatchEvaluator()

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

        exact_match_result = self.exact_match_evaluator.run(
            ground_truth_answers=[expected_answer], predicted_answers=[generated_answer]
        )

        exact_match_score = exact_match_result.get("score", 0)

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="exact_match",
                value=exact_match_score,
                confidence=1.0,
            )
        )

        sequence_similarity = self._calculate_sequence_similarity(
            expected_answer, generated_answer
        )

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="sequence_similarity",
                value=sequence_similarity,
                confidence=0.9,
            )
        )

        word_overlap = self._calculate_word_overlap(expected_answer, generated_answer)

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="word_overlap",
                value=word_overlap,
                confidence=0.8,
            )
        )

        expected_len = len(expected_answer.split())
        generated_len = len(generated_answer.split())
        if expected_len > 0:
            length_ratio = min(generated_len / expected_len, 2.0)

        else:
            length_ratio = 1.0 if generated_len == 0 else 2.0

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="answer_length_ratio",
                value=length_ratio,
                confidence=0.8,
            )
        )

        keyword_match = self._calculate_keyword_match(expected_answer, generated_answer)

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="keyword_match",
                value=keyword_match,
                confidence=0.7,
            )
        )

        return results

    def _calculate_sequence_similarity(self, reference: str, candidate: str) -> float:
        try:
            if not reference.strip() or not candidate.strip():
                return 0.0

            matcher = difflib.SequenceMatcher(
                None, reference.lower(), candidate.lower()
            )
            return matcher.ratio()

        except Exception as e:
            log.info(f"Error calculating sequence similarity: {e}")
            return 0.0

    def _calculate_word_overlap(self, reference: str, candidate: str) -> float:
        try:
            if not reference.strip() or not candidate.strip():
                return 0.0

            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())

            if not ref_words and not cand_words:
                return 1.0
            elif not ref_words and not cand_words:
                return 0.0

            intersection = len(ref_words.intersection(cand_words))
            union = len(ref_words.union(cand_words))

            return intersection / union

        except Exception as e:
            log.error(f"Error calculating word overlap: {e}")
            return 0.0

    def _calculate_keyword_match(self, reference: str, candidate: str) -> float:
        try:
            if not reference.strip() or not candidate.strip():
                return 0.0

            ref_lower = reference.lower()
            cand_lower = reference.lower()

            ref_words = [w for w in ref_lower.split() if len(w) > 3]

            if not ref_words:
                return 0.0

            matches = sum(1 for word in ref_words if word in cand_lower)

            return matches / len(ref_words)

        except Exception as e:
            log.error(f"Error calculating keyword match: {e}")
            return 0.0

    def cleanup(self):
        pass
