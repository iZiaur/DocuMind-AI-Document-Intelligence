"""
CLI Entrypoint for DocuMind AI Stage 5 Evaluation Benchmark.
Usage:
    python -m backend.evaluation.run_eval
    python -m backend.evaluation.run_eval --k 3 --output eval_results.json
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.evaluation.evaluator import run_evaluation, print_evaluation_summary


def main():
    parser = argparse.ArgumentParser(
        description="DocuMind AI Stage 5 Reproducible Evaluation Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON file (defaults to backend/evaluation/dataset.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top passages to retrieve (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Path to save machine-readable evaluation results JSON (default: eval_results.json)",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable optional model-judged grading if OPENAI_API_KEY is configured",
    )

    args = parser.parse_args()

    print("[DocuMind AI Evaluation] Initializing benchmark run...")
    report = run_evaluation(
        dataset_path=args.dataset,
        k=args.k,
        output_path=args.output,
        use_llm_judge=args.use_llm_judge,
    )

    print_evaluation_summary(report)
    print(f"[DocuMind AI Evaluation] Complete results written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
