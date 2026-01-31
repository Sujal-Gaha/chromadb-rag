from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Optional


class EvaluationType(Enum):
    """Types of evaluations supported"""

    ANSWER_QUALITY = "answer_quality"
    RETRIEVAL = "retrieval"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class EvaluationResult:
    """Single evaluation result"""

    evaluator_type: str
    metric_name: str
    value: Any
    confidence: float = 1.0
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class QuestionResult:
    """Results for a single question"""

    question: str
    expected_answer: str
    generated_answer: str
    retrieved_docs: list[str]
    expected_docs: list[str]
    response_time: float
    retrieval_count: int
    difficulty: str
    evaluation_results: list[EvaluationResult]

    def to_dict(self):
        result = asdict(self)
        result["evaluation_results"] = [r.to_dict() for r in self.evaluation_results]
        return result


@dataclass
class BatchResult:
    """Results for a batch evaluation"""

    batch_id: str
    timestamp: str
    pipeline_version: str
    questions: list[QuestionResult]
    aggregated_metrics: dict[str, Any]

    def to_dict(self):
        result = asdict(self)
        result["questions"] = [q.to_dict() for q in self.questions]
        return result


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators"""

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self.evaluation_type = EvaluationType.CUSTOM

    @abstractmethod
    async def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_docs: list[str],
        expected_docs: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[EvaluationResult]:
        """Evaluate a single question"""
        pass

    def get_required_metadata(self) -> list[str]:
        """List metadata fields required by this evaluator"""
        return []

    def cleanup(self):
        """Clean up resources"""
        pass

    def __str__(self):
        return f"{self.name} ({self.evaluation_type.value})"
