"""
DocuMind AI Stage 5 Evaluation Package.
"""
from backend.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    evaluate_key_facts,
    evaluate_citations,
    evaluate_abstention,
)
from backend.evaluation.evaluator import run_evaluation, load_dataset
