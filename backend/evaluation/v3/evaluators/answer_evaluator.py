import difflib
import numpy as np

from typing import Any, Optional, Union

from sklearn.metrics.pairwise import cosine_similarity

from haystack import Document
from evaluation.v3.base.evaluator_base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationType,
)
from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    FaithfulnessEvaluator,
)
from haystack_integrations.components.generators.ollama import (
    OllamaChatGenerator,
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

from utils.config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class AnswerEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        self.config = config
        super().__init__("AnswerEvaluator", config=self.config)

        self.evaluation_type = EvaluationType.ANSWER_QUALITY

        self.exact_match_evaluator = AnswerExactMatchEvaluator()

        self.embedder = OllamaTextEmbedder(
            model=self.config.ollama.embedding_model,
            url=self.config.ollama.server_url,
            timeout=self.config.ollama.timeout,
            generation_kwargs={
                "temperature": 0.0,
            },
        )

        judge = OllamaChatGenerator(
            model=self.config.ollama.judge_model,
            url=self.config.ollama.server_url,
            timeout=300,
            generation_kwargs={
                "temperature": 0.0,  # low for consistent judging
                "num_ctx": 8192,
                "format": "json",
            },
        )

        self.faithfulness_evaluator = FaithfulnessEvaluator(chat_generator=judge)

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

        sas_score = 0.0

        if not expected_answer.strip() or not generated_answer.strip():
            log.info(
                "Skipping semantic similarity — one or both answers empty. "
                f"Question: {question[:60]!r}"
                f"Expected answer: {expected_answer}"
                f"Generated answer: {generated_answer}"
            )
            sas_score = 0.0
            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="semantic_similarity",
                    value=0.0,
                    confidence=0.95,
                    metadata={
                        "reason": "one or both answers were empty/whitespace-only"
                    },
                )
            )
        else:
            try:
                expected_result = self.embedder.run(text=expected_answer)
                expected_embedding = expected_result["embedding"]

                generated_result = self.embedder.run(text=generated_answer)
                generated_embedding = generated_result["embedding"]

                expected_array = np.array([expected_embedding])
                generated_array = np.array([generated_embedding])

                sas_score = cosine_similarity(expected_array, generated_array)[0][0]

            except Exception as e:
                log.error(
                    f"Semantic similarity evaluation failed: {e}, defaulting to 0.0"
                )
                sas_score = 0.0

            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="semantic_similarity",
                    value=sas_score,
                    confidence=0.95,
                    metadata={
                        "method": "OllamaEmbedder",
                        "model": self.config.ollama.embedding_model,
                    },
                )
            )

        contexts = [
            doc.content if isinstance(doc, Document) else str(doc)
            for doc in retrieved_docs
        ]

        if not contexts:
            contexts = [""]

        faithfulness_score = 0.0
        try:
            faithfulness_result = self.faithfulness_evaluator.run(
                questions=[question],
                contexts=[contexts],
                predicted_answers=[generated_answer],
            )
            faithfulness_score = faithfulness_result.get("score", 0.0)
        except Exception as e:
            log.warning(f"Faithfulness evaluation failed: {e}, skipping...")
            faithfulness_score = 0.0  # or 0.0

        if faithfulness_score is not None:
            results.append(
                EvaluationResult(
                    evaluator_type=self.name,
                    metric_name="faithfulness",
                    value=faithfulness_score,
                    confidence=0.90,
                    metadata={"judge_model": "llama3.1:8b"},
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
            cand_lower = candidate.lower()

            ref_words = [w for w in ref_lower.split() if len(w) > 3]
            if not ref_words:
                return 0.0

            matches = sum(1 for word in ref_words if word in cand_lower)
            return matches / len(ref_words)

        except Exception as e:
            log.error(f"Error calculating keyword match: {e}")
            return 0.0

    def cleanup(self):
        log.debug("Cleaning up AnswerEvaluator resources")
        pass
