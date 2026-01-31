import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(Path(__file__).parent.parent.parent))

# Import your existing modules
from app.rag_pipeline import RAGPipeline
from evaluation.v3.data.gold_data import (
    GOLD_DATA,
)

# Import v3 modules
from evaluation.v3.pipeline.evaluation_pipeline import EvaluationPipeline
from evaluation.v3.evaluators.answer_evaluator import AnswerEvaluator
from evaluation.v3.evaluators.retrieval_evaluator import RetrievalEvaluator
from evaluation.v3.evaluators.performance_evaluator import PerformanceEvaluator
from evaluation.v3.visualization.visualizer import EvaluationVisualizer


async def run_evaluation(
    gold_data=None,
    batch_size=3,
    output_dir="evaluation_results/v3",
    create_visualizations=True,
    save_results=True,
):
    """
    Run a complete evaluation using the v3 architecture.

    Args:
        gold_data: List of gold data items (uses GOLD_DATA by default)
        batch_size: Number of questions to process in parallel
        output_dir: Directory to save results and visualizations
        create_visualizations: Whether to generate charts
        save_results: Whether to save results to files
    """
    print("=" * 70)
    print("🚀 RAG EVALUATION v3")
    print("=" * 70)

    # 1. Initialize RAG Pipeline
    print("🔧 Initializing RAG Pipeline...")
    rag_pipeline = RAGPipeline()
    rag_pipeline.initialize()

    # 2. Create Evaluation Pipeline
    print("🔧 Setting up Evaluation Pipeline...")
    eval_pipeline = EvaluationPipeline(rag_pipeline)

    # 3. Register Evaluators
    print("🔧 Registering evaluators...")

    # Answer quality evaluator
    answer_evaluator = AnswerEvaluator({"similarity_model": "all-MiniLM-L6-v2"})
    eval_pipeline.register_evaluator(answer_evaluator)

    # Retrieval evaluator
    retrieval_evaluator = RetrievalEvaluator({"relevance_threshold": 0.7})
    eval_pipeline.register_evaluator(retrieval_evaluator)

    # Performance evaluator
    performance_evaluator = PerformanceEvaluator({"target_response_time": 2.0})
    eval_pipeline.register_evaluator(performance_evaluator)

    print(f"✅ Registered {len(eval_pipeline.evaluators)} evaluators")

    # 4. Run Evaluation
    print("\n📊 Running evaluation...")
    gold_data_to_use = gold_data or GOLD_DATA

    def progress_callback(progress, completed, total):
        print(f"   Progress: {progress:.1f}% ({completed}/{total} questions)")

    batch_result = await eval_pipeline.evaluate_batch(
        gold_data=gold_data_to_use,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    # 5. Generate Summary
    print("\n📈 Generating summary...")
    summary = eval_pipeline.get_summary_report(batch_result)

    print("\n" + "=" * 70)
    print("📋 EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Batch ID: {summary['batch_id']}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Timestamp: {summary['timestamp']}")

    print("\n📊 Top Metrics:")
    for metric, value in summary["top_metrics"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    if summary["difficulty_breakdown"]:
        print("\n🎯 Difficulty Breakdown:")
        for difficulty, count in summary["difficulty_breakdown"].items():
            print(f"  {difficulty}: {count}")

    # 6. Save Results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results to CSV
        df = eval_pipeline.results_to_dataframe(batch_result)
        csv_path = output_path / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")

        # Save summary to JSON
        json_path = output_path / f"summary_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Summary saved to: {json_path}")

        # 7. Create Visualizations
        if create_visualizations and len(df) > 0:
            print("\n🎨 Creating visualizations...")
            visualizer = EvaluationVisualizer(dpi=120)

            viz_files = visualizer.create_comprehensive_report(
                df,
                output_dir=str(output_path / "visualizations"),
                batch_id=batch_result.batch_id,
            )

            print(f"✅ Created {len(viz_files)} visualizations")
            for viz_name, viz_path in viz_files.items():
                print(f"   📊 {viz_name}: {viz_path}")

        # Save full batch result
        batch_json_path = output_path / f"batch_result_{timestamp}.json"
        with open(batch_json_path, "w") as f:
            json.dump(batch_result.to_dict(), f, indent=2, default=str)

    print("\nCleaning up...")
    eval_pipeline.cleanup()

    print("\n" + "=" * 70)
    print("Evaluation completed successfully!")
    print("=" * 70)

    return batch_result, summary


async def evaluate_single_question(
    question: str, expected_answer: str, expected_docs: Optional[list[str]] = None
):
    """
    Evaluate a single question (useful for testing)
    """
    rag_pipeline = RAGPipeline()
    rag_pipeline.initialize()

    eval_pipeline = EvaluationPipeline(rag_pipeline)

    eval_pipeline.register_evaluator(AnswerEvaluator())

    result = await eval_pipeline.evaluate_single(
        question=question,
        expected_answer=expected_answer,
        expected_docs=expected_docs or [],
    )

    print(f"\nQuestion: {question}")
    print(f"Expected: {expected_answer}")
    print(f"Generated: {result.generated_answer}")
    print(f"Response time: {result.response_time:.2f}s")

    print("\nEvaluation Results:")
    for eval_result in result.evaluation_results:
        print(
            f"  {eval_result.evaluator_type}.{eval_result.metric_name}: {eval_result.value:.3f}"
        )

    eval_pipeline.cleanup()

    return result


def main():
    """Command-line interface"""
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
