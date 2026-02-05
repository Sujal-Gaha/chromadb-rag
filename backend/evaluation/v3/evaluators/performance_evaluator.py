from typing import Any, Optional, Union

from haystack import Document

from evaluation.v3.base.evaluator_base import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationType,
)

from utils.config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class PerformanceEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        self.config = config
        super().__init__("PerformanceEvaluator", config=self.config)

        self.evaluation_type = EvaluationType.PERFORMANCE

        self.target_response_time = self.config.pipeline.target_response_time

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

        response_time = metadata.get("response_time", 0) if metadata else 0
        retrieval_count = metadata.get("retrieval_count", 0) if metadata else 0

        if self.target_response_time > 0:
            time_score = max(0, 1 - (response_time / self.target_response_time))
        else:
            time_score = 1.0 if response_time < 5 else 0.5

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="response_time_score",
                value=time_score,
                confidence=1.0,
                metadata={"actual_time": response_time},
            )
        )

        optimal_retrieval = len(expected_docs) if expected_docs else 3
        if retrieval_count > 0:
            efficiency = min(optimal_retrieval / retrieval_count, 1.0)
        else:
            efficiency = 0.0

        retrieval_waste_score = 1 - efficiency

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="retrieval_waste",
                value=retrieval_waste_score,
                confidence=0.0,
                metadata={
                    "retrieved_count": retrieval_count,
                    "expected_count": optimal_retrieval,
                },
            )
        )

        has_answer = bool(
            generated_answer and generated_answer.strip()
        ) and not generated_answer.startswith("ERROR")

        results.append(
            EvaluationResult(
                evaluator_type=self.name,
                metric_name="answer_provided",
                value=float(has_answer),
                confidence=1.0,
            )
        )

        return results

    def cleanup(self):
        log.debug("Cleaning up PerformanceEvaluator resources")
        pass
