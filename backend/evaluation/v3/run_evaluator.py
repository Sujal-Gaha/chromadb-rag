import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(Path(__file__).parent.parent.parent))

from app.rag_pipeline import RAGPipeline
from evaluation.v3.data.gold_data import (
    GOLD_DATA,
)

from evaluation.v3.pipeline.evaluation_pipeline import EvaluationPipeline
from evaluation.v3.evaluators.answer_evaluator import AnswerEvaluator
from evaluation.v3.evaluators.retrieval_evaluator import RetrievalEvaluator
from evaluation.v3.evaluators.retrieval_evaluator_v2 import RetrievalEvaluatorV2
from evaluation.v3.evaluators.performance_evaluator import PerformanceEvaluator
from evaluation.v3.visualization.visualizer import EvaluationVisualizer

from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


async def run_evaluation(
    gold_data=None,
    batch_size=3,
    output_dir="evaluation_results/v3",
    create_visualizations=True,
    save_results=True,
):
    log.debug("=" * 70)
    log.debug("RAG EVALUATION v3")
    log.debug("=" * 70)

    log.debug("Initializing RAG Pipeline...")
    rag_pipeline = RAGPipeline()
    rag_pipeline.initialize()

    log.debug("Setting up Evaluation Pipeline...")
    eval_pipeline = EvaluationPipeline(rag_pipeline)

    log.debug("Registering evaluators...")

    config = get_config()

    answer_evaluator = AnswerEvaluator(config=config)
    eval_pipeline.register_evaluator(answer_evaluator)

    # retrieval_evaluator = RetrievalEvaluator({"relevance_threshold": 0.7})
    # eval_pipeline.register_evaluator(retrieval_evaluator)

    retrieval_evaluator_v2 = RetrievalEvaluatorV2(config=config)
    eval_pipeline.register_evaluator(retrieval_evaluator_v2)

    performance_evaluator = PerformanceEvaluator(config=config)
    eval_pipeline.register_evaluator(performance_evaluator)

    log.info(f"Registered {len(eval_pipeline.evaluators)} evaluators")

    log.info("Running evaluation...")
    gold_data_to_use = gold_data or GOLD_DATA

    log.debug("Initializing filename mapping...")
    for item in gold_data_to_use:
        for doc in item.get("expected_context", []):
            log.info(f"Expected document: {doc}")

    def progress_callback(progress, completed, total):
        log.info(f"Progress: {progress:.1f}% ({completed}/{total} questions)")

    batch_result = await eval_pipeline.evaluate_batch(
        gold_data=gold_data_to_use,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    log.info("Generating summary...")
    summary = eval_pipeline.get_summary_report(batch_result)

    log.info("=" * 70)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 70)

    log.info(f"Batch ID: {summary['batch_id']}")
    log.info(f"Total Questions: {summary['total_questions']}")
    log.info(f"Timestamp: {summary['timestamp']}")

    log.info("Top Metrics:")
    for metric, value in summary["top_metrics"].items():
        if isinstance(value, float):
            log.info(f"{metric}: {value:.4f}")
        else:
            log.info(f"{metric}: {value}")

    if summary["difficulty_breakdown"]:
        log.info("Difficulty Breakdown:")
        for difficulty, count in summary["difficulty_breakdown"].items():
            log.info(f"  {difficulty}: {count}")

    viz_files = {}

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df = eval_pipeline.results_to_dataframe(batch_result)
        csv_path = output_path / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Results saved to: {csv_path}")

        json_path = output_path / f"summary_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(f"Summary saved to: {json_path}")

        if create_visualizations and len(df) > 0:
            log.info("Creating visualizations...")
            visualizer = EvaluationVisualizer(dpi=120)

            viz_files = visualizer.create_comprehensive_report(
                df,
                output_dir=str(output_path / "visualizations"),
                batch_id=batch_result.batch_id,
            )

            log.info(f"Created {len(viz_files)} visualizations")
            for viz_name, viz_path in viz_files.items():
                log.info(f"{viz_name}: {viz_path}")

        batch_json_path = output_path / f"batch_result_{timestamp}.json"
        with open(batch_json_path, "w") as f:
            json.dump(batch_result.to_dict(), f, indent=2, default=str)

    log.info("Cleaning up...")
    eval_pipeline.cleanup()

    log.info("=" * 70)
    log.info("Evaluation completed successfully!")
    log.info("=" * 70)

    return batch_result, summary


async def evaluate_single_question(
    question: str, expected_answer: str, expected_docs: Optional[list[str]] = None
):
    rag_pipeline = RAGPipeline()
    rag_pipeline.initialize()

    config = get_config()

    eval_pipeline = EvaluationPipeline(rag_pipeline)

    eval_pipeline.register_evaluator(AnswerEvaluator(config=config))

    result = await eval_pipeline.evaluate_single(
        question=question,
        expected_answer=expected_answer,
        expected_docs=expected_docs or [],
    )

    log.info(f"Question: {question}")
    log.info(f"Expected: {expected_answer}")
    log.info(f"Generated: {result.generated_answer}")
    log.info(f"Response time: {result.response_time:.2f}s")

    log.info("Evaluation Results:")
    for eval_result in result.evaluation_results:
        log.info(
            f"  {eval_result.evaluator_type}.{eval_result.metric_name}: {eval_result.value:.3f}"
        )

    eval_pipeline.cleanup()

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation v3")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of questions to process in parallel",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/v3",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization generation"
    )
    parser.add_argument(
        "--single", action="store_true", help="Run single question test"
    )

    args = parser.parse_args()

    if args.single:
        test_question = (
            "What happened in the school bus accident when John was a teenager?"
        )
        test_answer = "The school bus skidded on wet pavement and collided with a fallen tree. John's best friend Tommy suffered a broken arm and the bus driver was hospitalized. John emerged unscathed but philosophically unaffected."
        test_docs = ["the-absurd-adolescence-of-john-doe.txt"]

        asyncio.run(evaluate_single_question(test_question, test_answer, test_docs))
    else:
        asyncio.run(
            run_evaluation(
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                create_visualizations=not args.no_viz,
                save_results=True,
            )
        )


if __name__ == "__main__":
    main()
