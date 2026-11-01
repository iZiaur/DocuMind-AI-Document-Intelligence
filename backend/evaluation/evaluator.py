"""
DocuMind AI - Stage 5 End-to-End Evaluation Engine.
Loads evaluation dataset, ingests sample benchmark documents into an isolated database,
evaluates retrieval and generation metrics, and generates machine-readable reports.
Benchmark document PDFs are stored in an isolated temporary directory and automatically cleaned up.
"""

import os
import json
import uuid
import datetime
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.models import Base, User, Document
from backend.auth import hash_password
from backend.storage import save_pdf_securely
from backend.ingestion import process_document_pipeline
from backend.main import retrieve_authorized_passages
from backend.rag import generate_rag_answer, LLM_MODEL
from backend.embeddings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, OPENAI_API_KEY
from backend.test_main import create_pdf_with_text
from backend.evaluation.sample_documents import SAMPLE_DOCUMENTS
from backend.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    evaluate_key_facts,
    evaluate_citations,
    evaluate_abstention,
    evaluate_abstention_details,
)


def load_dataset(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads evaluation test cases from JSON file."""
    if not dataset_path:
        base_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(base_dir, "dataset.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_evaluation_environment() -> Tuple[Session, User]:
    """Sets up an isolated in-memory SQLite database and test user for evaluation."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = Session(bind=engine)

    # Create synthetic evaluation evaluator user
    user = User(
        email="evaluator_bot@documind.ai",
        username="evaluator_bot",
        hashed_password=hash_password("EvalSecure123!"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return db, user


def ingest_sample_documents(
    db: Session,
    user: User,
    storage_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Synthesizes and ingests all sample benchmark PDFs into the evaluation database.
    If storage_dir is provided, writes to that temporary directory, isolating benchmark files
    from the production storage directory.
    Returns: Mapping of document_title -> document_id.
    """
    doc_id_map: Dict[str, str] = {}

    for title, doc_spec in SAMPLE_DOCUMENTS.items():
        pdf_bytes = create_pdf_with_text(doc_spec["pages"])
        doc_id, disk_path, page_count, file_hash = save_pdf_securely(
            file_bytes=pdf_bytes,
            original_filename=doc_spec["filename"],
            target_dir=storage_dir,
        )

        doc = Document(
            id=doc_id,
            owner_id=user.id,
            title=title,
            filename=doc_spec["filename"],
            file_hash=file_hash,
            storage_path=disk_path,
            file_size_bytes=len(pdf_bytes),
            page_count=page_count,
            content_type="application/pdf",
            status="queued",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        process_document_pipeline(doc.id, db)
        db.refresh(doc)
        doc_id_map[title] = doc.id

    return doc_id_map


def run_evaluation(
    dataset_path: Optional[str] = None,
    k: int = 3,
    output_path: Optional[str] = None,
    db: Optional[Session] = None,
    user: Optional[User] = None,
    use_llm_judge: bool = False,
    storage_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the reproducible evaluation benchmark.
    Guarantees isolation:
    - Uses an in-memory SQLite database if not supplied.
    - Uses an isolated TemporaryDirectory for PDF files if not supplied, cleaning up on exit.
    - Resilient to individual case failures (logs error safely and continues with remaining cases).
    Returns: Complete results dictionary with aggregate and per-case metrics.
    """
    dataset = load_dataset(dataset_path)
    cases = dataset.get("cases", [])

    temp_dir_obj = None
    if storage_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="documind_eval_")
        storage_dir = temp_dir_obj.name

    try:
        if db is None or user is None:
            db, user = setup_evaluation_environment()
            ingest_sample_documents(db, user, storage_dir=storage_dir)

        case_results: List[Dict[str, Any]] = []

        # Accumulators for aggregate metrics
        recalls: List[float] = []
        precisions: List[float] = []
        reciprocal_ranks: List[float] = []
        key_fact_scores: List[float] = []
        answerable_citation_precisions: List[float] = []
        answerable_citation_recalls: List[float] = []

        # Abstention breakdown accumulators
        valid_answerable_count = 0
        valid_unanswerable_count = 0
        answerable_false_refusals = 0
        unanswerable_correct_abstentions = 0
        unexpected_citations_count = 0
        overall_correct_count = 0
        failed_cases_count = 0

        for case in cases:
            cid = case.get("id", "unknown")
            q = case.get("question", "")
            exp_evidence = case.get("expected_evidence", [])
            is_answerable = case.get("is_answerable", True)
            exp_abstention = case.get("expected_abstention", False)
            key_facts = case.get("key_facts", [])

            try:
                # 1. Retrieval Phase
                passages = retrieve_authorized_passages(
                    query_text=q,
                    user_id=user.id,
                    top_k=k,
                    document_id=None,
                    db=db,
                )

                # 2. Generation Phase
                response = generate_rag_answer(
                    question=q,
                    passages=passages,
                )

                # 3. Metric Calculations
                recall = compute_recall_at_k(passages, exp_evidence, k=k)
                precision = compute_precision_at_k(passages, exp_evidence, k=k)
                rr = compute_reciprocal_rank(passages, exp_evidence, k=k)

                fact_score = evaluate_key_facts(response.answer, key_facts)
                cit_prec, cit_rec = evaluate_citations(response.citations, exp_evidence)

                abst_details = evaluate_abstention_details(
                    expected_abstention=exp_abstention,
                    actual_insufficient_evidence=response.insufficient_evidence,
                    actual_citations_count=len(response.citations),
                )

                # Record metrics strictly by answerability
                if is_answerable:
                    valid_answerable_count += 1
                    if recall is not None:
                        recalls.append(recall)
                    if precision is not None:
                        precisions.append(precision)
                    if rr is not None:
                        reciprocal_ranks.append(rr)
                    key_fact_scores.append(fact_score)

                    # Option B: Averaged across all valid answerable cases.
                    # Cases emitting 0 citations receive cit_prec=0.0 (penalized), including them in the denominator.
                    # (Distinguished from conditional citation precision, which would exclude 0-citation cases).
                    if cit_prec is not None:
                        answerable_citation_precisions.append(cit_prec)
                    if cit_rec is not None:
                        answerable_citation_recalls.append(cit_rec)

                    if abst_details["is_false_refusal"]:
                        answerable_false_refusals += 1
                else:
                    valid_unanswerable_count += 1
                    if len(response.citations) > 0:
                        unexpected_citations_count += 1
                    if abst_details["is_correct_abstention"]:
                        unanswerable_correct_abstentions += 1

                if abst_details["is_overall_correct"]:
                    overall_correct_count += 1

                case_results.append({
                    "id": cid,
                    "status": "success",
                    "category": case.get("category", "unknown"),
                    "question": q,
                    "is_answerable": is_answerable,
                    "expected_abstention": exp_abstention,
                    "actual_insufficient_evidence": response.insufficient_evidence,
                    "retrieved_count": len(passages),
                    "citations_count": len(response.citations),
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "reciprocal_rank": rr,
                    "key_fact_score": fact_score,
                    "citation_precision": cit_prec,
                    "citation_recall": cit_rec,
                    "abstention_correct": abst_details["is_overall_correct"],
                    "is_false_refusal": abst_details["is_false_refusal"],
                    "is_correct_abstention": abst_details["is_correct_abstention"],
                    "answer_preview": response.answer[:120] + ("..." if len(response.answer) > 120 else ""),
                })

            except Exception as err:
                failed_cases_count += 1
                safe_err_msg = f"{type(err).__name__}: An error occurred while evaluating case {cid}"
                case_results.append({
                    "id": cid,
                    "status": "failed",
                    "error": safe_err_msg,
                    "category": case.get("category", "unknown"),
                    "question": q,
                    "is_answerable": is_answerable,
                    "expected_abstention": exp_abstention,
                    "actual_insufficient_evidence": None,
                    "retrieved_count": 0,
                    "citations_count": 0,
                    "recall_at_k": None,
                    "precision_at_k": None,
                    "reciprocal_rank": None,
                    "key_fact_score": 0.0,
                    "citation_precision": None,
                    "citation_recall": None,
                    "abstention_correct": False,
                    "is_false_refusal": False,
                    "is_correct_abstention": False,
                    "answer_preview": "[EVALUATION FAILED]",
                })

        # Calculate exact aggregate metrics over valid, successfully evaluated cases
        successful_cases_count = len(cases) - failed_cases_count

        mean_recall = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        mean_precision = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
        mean_mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0
        mean_key_facts = round(sum(key_fact_scores) / len(key_fact_scores), 4) if key_fact_scores else 0.0

        mean_cit_prec = (
            round(sum(answerable_citation_precisions) / len(answerable_citation_precisions), 4)
            if answerable_citation_precisions else 0.0
        )
        mean_cit_rec = (
            round(sum(answerable_citation_recalls) / len(answerable_citation_recalls), 4)
            if answerable_citation_recalls else 0.0
        )

        true_abstention_rate = (
            round(unanswerable_correct_abstentions / valid_unanswerable_count, 4)
            if valid_unanswerable_count > 0 else 0.0
        )
        false_refusal_rate = (
            round(answerable_false_refusals / valid_answerable_count, 4)
            if valid_answerable_count > 0 else 0.0
        )
        unexpected_citation_rate = (
            round(unexpected_citations_count / valid_unanswerable_count, 4)
            if valid_unanswerable_count > 0 else 0.0
        )
        overall_correct_handling_rate = (
            round(overall_correct_count / successful_cases_count, 4)
            if successful_cases_count > 0 else 0.0
        )

        report: Dict[str, Any] = {
            "dataset_version": dataset.get("version", "1.0.0"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "evaluation_config": {
                "top_k": k,
                "model_used": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "live_llm_active": bool(OPENAI_API_KEY),
                "llm_judge_active": use_llm_judge and bool(OPENAI_API_KEY),
                "isolated_temp_storage": True,
            },
            "aggregate_metrics": {
                "mean_recall_at_k": mean_recall,
                "mean_precision_at_k": mean_precision,
                "mean_reciprocal_rank_mrr": mean_mrr,
                "answer_key_fact_score": mean_key_facts,
                "mean_citation_precision": mean_cit_prec,
                "mean_citation_recall": mean_cit_rec,
                "citation_precision_denominator_mode": "all_valid_answerable_cases",
                "true_abstention_rate": true_abstention_rate,
                "false_refusal_rate": false_refusal_rate,
                "unexpected_citation_rate": unexpected_citation_rate,
                "overall_correct_handling_rate": overall_correct_handling_rate,
                "abstention_accuracy": overall_correct_handling_rate,  # backward compatibility
                "total_evaluated_cases": len(cases),
                "successful_cases_count": successful_cases_count,
                "failed_cases_count": failed_cases_count,
                "answerable_cases_count": valid_answerable_count,
                "unanswerable_cases_count": valid_unanswerable_count,
            },
            "case_results": case_results,
        }

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def print_evaluation_summary(report: Dict[str, Any]) -> None:
    """Formats and prints an ASCII summary table of evaluation results to stdout."""
    cfg = report["evaluation_config"]
    agg = report["aggregate_metrics"]

    print("\n" + "=" * 76)
    print("           DOCUMIND AI — STAGE 5 EVALUATION BENCHMARK REPORT")
    print("=" * 76)
    print(f" Dataset Version : {report['dataset_version']}")
    print(f" Timestamp       : {report['timestamp']}")
    print(f" Retrieval k     : {cfg['top_k']}")
    print(f" Embeddings      : {cfg['embedding_model']} ({cfg['embedding_dimension']}-dim)")
    print(f" Generator Model : {cfg['model_used']} (Live OpenAI: {cfg['live_llm_active']})")
    print(f" Storage Mode    : Isolated TempDirectory (Cleaned up: True)")
    print("-" * 76)
    print(" RETRIEVAL METRICS (Evaluated on Answerable Queries):")
    print(f"   * Mean Recall@{cfg['top_k']}      : {agg['mean_recall_at_k'] * 100:.1f}%  (Target: > 80%)")
    print(f"   * Mean Precision@{cfg['top_k']}   : {agg['mean_precision_at_k'] * 100:.1f}%")
    print(f"   * MRR (Mean Reciprocal Rank) : {agg['mean_reciprocal_rank_mrr']:.4f}")
    print("-" * 76)
    print(" GENERATION & CITATION METRICS (All Valid Answerable Cases):")
    print(f"   * Answer Key-Fact Pass Rate : {agg['answer_key_fact_score'] * 100:.1f}%")
    print(f"   * Citation Precision        : {agg['mean_citation_precision'] * 100:.1f}%  (Denominator: {agg['answerable_cases_count']} valid answerable cases; 0-citation cases penalized with 0.0)")
    print(f"   * Citation Recall           : {agg['mean_citation_recall'] * 100:.1f}%  (Denominator: {agg['answerable_cases_count']} valid answerable cases)")
    print("-" * 76)
    print(" ABSTENTION & SAFETY METRICS (Explicitly Split):")
    print(f"   * True Abstention Rate      : {agg['true_abstention_rate'] * 100:.1f}%  ({agg['unanswerable_cases_count'] - int(round(agg['unexpected_citation_rate'] * agg['unanswerable_cases_count']))}/{agg['unanswerable_cases_count']} unanswerable correctly refused)")
    print(f"   * False Refusal Rate        : {agg['false_refusal_rate'] * 100:.1f}%  ({int(round(agg['false_refusal_rate'] * agg['answerable_cases_count']))}/{agg['answerable_cases_count']} answerable queries wrongly refused)")
    print(f"   * Unexpected Citation Rate  : {agg['unexpected_citation_rate'] * 100:.1f}%  (unanswerable queries with citations)")
    print(f"   * Overall Correct Handling  : {agg['overall_correct_handling_rate'] * 100:.1f}%  ({int(round(agg['overall_correct_handling_rate'] * agg['successful_cases_count']))}/{agg['successful_cases_count']} valid cases correctly handled)")
    print("-" * 76)
    print(f" Cases Summary: {agg['total_evaluated_cases']} total ({agg['answerable_cases_count']} answerable, {agg['unanswerable_cases_count']} unanswerable, {agg['failed_cases_count']} failed)")
    print("=" * 76 + "\n")
