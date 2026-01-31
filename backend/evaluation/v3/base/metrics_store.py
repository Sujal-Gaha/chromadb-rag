from datetime import datetime, timezone

from dataclasses import dataclass
from typing import Any

from evaluation.v3.base.evaluator_base import BatchResult

import pandas as pd
import json


@dataclass
class MetricDefinition:
    """Definition of a metric"""

    name: str
    description: str
    range_min: float
    range_max: float
    unit: str
    direction: str  # 'higher_is_better' or 'lower_is_better'
    category: str


class MetricsStore:

    def __init__(self):
        self.metrics_definitions: dict[str, MetricDefinition] = {}
        self.history: list[dict[str, Any]] = []

    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition"""
        self.metrics_definitions[metric_def.name] = metric_def

    def record_evaluation(self, batch_eval: BatchResult):
        """Record a batch evaluation"""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "batch_id": batch_eval.batch_id,
            "pipeline_version": batch_eval.pipeline_version,
            "metrics": batch_eval.aggregated_metrics,
            "question_count": len(batch_eval.questions),
        }
        self.history.append(record)

    def get_history_dataframe(self) -> pd.DataFrame:
        """Get evaluation history as DataFrame"""
        return pd.DataFrame(self.history)

    def get_metric_trend(self, metric_name: str) -> list[dict[str, Any]]:
        """Get historical trend for a specific metric"""
        trend = []
        for record in self.history:
            if metric_name in record["metrics"]:
                trend.append(
                    {
                        "timestamp": record["timestamp"],
                        "value": record["metrics"][metric_name],
                        "batch_id": record["batch_id"],
                    }
                )
        return trend

    def export_to_json(self, filepath: str):
        """Export metrics history to JSON"""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
