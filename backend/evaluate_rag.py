#!/usr/bin/env python3
"""
Simple script to run RAG pipeline evaluation
"""

import asyncio
from evaluation.evaluator import run_evaluation
from evaluation.visualize import create_evaluation_report


async def main():
    print("Starting RAG Pipeline Evaluation...")
    print("=" * 80)

    # Run evaluation
    results_df, metrics = await run_evaluation()

    # Create visualizations
    create_evaluation_report(results_df, metrics, "rag_evaluation")

    # Save results to CSV
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to 'evaluation_results.csv'")

    # Print top performing questions
    print("\n" + "=" * 80)
    print("TOP 5 PERFORMING QUESTIONS")
    print("=" * 80)
    top_5 = results_df.nlargest(5, "similarity_score")
    for idx, row in top_5.iterrows():
        print(f"\nSimilarity: {row['similarity_score']:.3f}")
        print(f"Question: {row['question']}")
        print(f"Context Match: {row['context_match']}")


if __name__ == "__main__":
    asyncio.run(main())
