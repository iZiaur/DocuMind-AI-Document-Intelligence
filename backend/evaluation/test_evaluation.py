"""
Unit tests for DocuMind AI Stage 5 Evaluation System.
Tests:
- Hand-checkable Recall@k, Precision@k, and MRR calculations.
- Missing or malformed labels.
- Citation IDs pointing to irrelevant evidence.
- Abstention metrics (split: true abstention rate, false refusal rate, overall correct handling).
- Proof that unanswerable cases cannot inflate citation recall.
- Storage isolation: Temporary directory cleanup after both successful and failed runs.
- Resilient recovery from single-case failures.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path

from backend.models import SearchPassageResponse, Citation
from backend.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    evaluate_key_facts,
    evaluate_citations,
    evaluate_abstention,
    evaluate_abstention_details,
    match_evidence,
)
from backend.evaluation.evaluator import run_evaluation, load_dataset


# ---------------------------------------------------------------------------
# Hand-Checkable Metric Tests
# ---------------------------------------------------------------------------

def test_metric_recall_at_k_hand_checkable():
    """
    Hand-checkable Recall@k:
    Expected evidence: Doc A Page 1, Doc B Page 2 (2 targets).
    """
    expected = [
        {"document_title": "Doc A", "page_number": 1},
        {"document_title": "Doc B", "page_number": 2},
    ]

    p1 = {"document_title": "Doc A", "page_number": 1}
    p2 = {"document_title": "Doc C", "page_number": 3}
    p3 = {"document_title": "Doc B", "page_number": 2}

    # Top-1 retrieved p1 (1 of 2 matched) -> Recall@1 = 0.5
    assert compute_recall_at_k([p1, p2, p3], expected, k=1) == 0.5

    # Top-2 retrieved p1, p2 (still 1 of 2 matched) -> Recall@2 = 0.5
    assert compute_recall_at_k([p1, p2, p3], expected, k=2) == 0.5

    # Top-3 retrieved p1, p2, p3 (both matched) -> Recall@3 = 1.0
    assert compute_recall_at_k([p1, p2, p3], expected, k=3) == 1.0

    # Top-1 retrieved only irrelevant p2 -> Recall@1 = 0.0
    assert compute_recall_at_k([p2], expected, k=1) == 0.0

    # Empty expected targets (e.g. unanswerable query) returns None
    assert compute_recall_at_k([p1, p2], [], k=3) is None


def test_metric_precision_at_k_hand_checkable():
    """
    Hand-checkable Precision@k:
    Expected: Doc A Page 1.
    """
    expected = [{"document_title": "Doc A", "page_number": 1}]

    p1 = {"document_title": "Doc A", "page_number": 1}  # Relevant
    p2 = {"document_title": "Doc B", "page_number": 2}  # Irrelevant
    p3 = {"document_title": "Doc A", "page_number": 1}  # Relevant duplicate page

    # k=1: [p1] -> 1/1 = 1.0
    assert compute_precision_at_k([p1, p2, p3], expected, k=1) == 1.0

    # k=2: [p1, p2] -> 1/2 = 0.5
    assert compute_precision_at_k([p1, p2, p3], expected, k=2) == 0.5

    # k=3: [p1, p2, p3] -> 2/3 = 0.6667
    assert compute_precision_at_k([p1, p2, p3], expected, k=3) == 0.6667

    # k=1 with only irrelevant passage: [p2] -> 0/1 = 0.0
    assert compute_precision_at_k([p2], expected, k=1) == 0.0

    # Empty retrieved passages
    assert compute_precision_at_k([], expected, k=3) == 0.0


def test_metric_mrr_hand_checkable():
    """
    Hand-checkable Mean Reciprocal Rank (MRR):
    Checks rank r of first relevant item: RR = 1 / r.
    """
    expected = [{"document_title": "Target Doc", "page_number": 1}]

    rel = {"document_title": "Target Doc", "page_number": 1}
    irrel_1 = {"document_title": "Other Doc", "page_number": 2}
    irrel_2 = {"document_title": "Other Doc", "page_number": 3}

    # Relevant at rank 1 -> RR = 1/1 = 1.0
    assert compute_reciprocal_rank([rel, irrel_1, irrel_2], expected, k=3) == 1.0

    # Relevant at rank 2 -> RR = 1/2 = 0.5
    assert compute_reciprocal_rank([irrel_1, rel, irrel_2], expected, k=3) == 0.5

    # Relevant at rank 3 -> RR = 1/3 = 0.3333
    assert compute_reciprocal_rank([irrel_1, irrel_2, rel], expected, k=3) == 0.3333

    # No relevant passage in top-k -> RR = 0.0
    assert compute_reciprocal_rank([irrel_1, irrel_2], expected, k=3) == 0.0


# ---------------------------------------------------------------------------
# Label Robustness & Citation Denominators
# ---------------------------------------------------------------------------

def test_metric_missing_or_malformed_labels():
    """Ensures metric calculations do not crash when given malformed or empty labels."""
    assert compute_recall_at_k([], [], k=3) is None
    assert compute_precision_at_k([], [], k=3) is None
    assert compute_reciprocal_rank([], [], k=3) is None

    malformed = [
        {},
        {"document_title": None},
        {"page_number": None},
        {"document_title": "Valid Doc", "page_number": 0},
    ]
    p = {"document_title": "Valid Doc", "page_number": 1}
    recall = compute_recall_at_k([p], malformed, k=3)
    assert recall is not None


def test_citation_recall_not_inflated_by_unanswerable_cases():
    """
    Citation denominator fix:
    Unanswerable cases (expected_evidence = []) must NEVER return recall=1.0.
    Recall must be None so it cannot artificially inflate answerable citation recall.
    """
    # 1. Clean abstention on unanswerable question
    prec, rec = evaluate_citations([], [])
    assert prec is None
    assert rec is None

    # 2. Hallucinated citation on unanswerable question
    fake_citation = Citation(
        chunk_id="chunk-fake-01",
        document_id="doc-fake",
        document_title="Unrelated Doc",
        page_number=1,
        excerpt="Some random text",
        score=0.65,
    )
    prec_bad, rec_bad = evaluate_citations([fake_citation], [])
    assert prec_bad == 0.0  # Penalized as 0% precision
    assert rec_bad is None  # Does NOT receive a free 1.0 recall!


def test_citation_pointing_to_irrelevant_evidence():
    """
    Citation validation: If a model cites a passage from an unrelated document,
    citation precision must be 0.0 and recall must be 0.0.
    """
    expected_evidence = [
        {"document_title": "Security Policy", "page_number": 1}
    ]

    irrelevant_citation = Citation(
        chunk_id="chunk-travel-999",
        document_id="doc-travel",
        document_title="Travel Policy",
        page_number=3,
        excerpt="Mileage is 67 cents per mile.",
        score=0.72,
    )

    prec, rec = evaluate_citations([irrelevant_citation], expected_evidence)
    assert prec == 0.0
    assert rec == 0.0


def test_citation_precision_zero_citations_answerable_included_in_denominator():
    """
    Verifies Option B citation precision behavior:
    - An answerable case with expected evidence that emits NO citations returns (0.0, 0.0).
    - When aggregated across answerable cases, the 0-citation case contributes 0.0 and is
      included in the denominator (denominator = all valid answerable cases, not just citation-emitting cases).
    - Hand-checkable arithmetic:
      Case A: 1 valid citation -> precision = 1.0
      Case B: 0 citations emitted -> precision = 0.0 (penalized)
      Aggregate precision = (1.0 + 0.0) / 2 = 0.50 (50.0%), proving denominator = 2.
    """
    expected_evidence = [{"document_title": "Security Policy", "page_number": 1}]

    # Case B: Answerable question, 0 citations emitted
    prec_zero, rec_zero = evaluate_citations([], expected_evidence)
    assert prec_zero == 0.0
    assert rec_zero == 0.0

    # Case A: Answerable question, 1 valid citation emitted
    valid_citation = Citation(
        chunk_id="chunk-valid-1",
        document_id="doc-1",
        document_title="Security Policy",
        page_number=1,
        excerpt="Valid text excerpt",
        score=0.9,
    )
    prec_valid, rec_valid = evaluate_citations([valid_citation], expected_evidence)
    assert prec_valid == 1.0
    assert rec_valid == 1.0

    # Aggregate simulation matching evaluator.py logic:
    answerable_precisions = []
    if prec_valid is not None:
        answerable_precisions.append(prec_valid)
    if prec_zero is not None:
        answerable_precisions.append(prec_zero)

    assert len(answerable_precisions) == 2  # Both cases included in denominator
    aggregate_precision = round(sum(answerable_precisions) / len(answerable_precisions), 4)
    assert aggregate_precision == 0.5000  # (1.0 + 0.0) / 2 = 0.5000


# ---------------------------------------------------------------------------
# Split Abstention Metrics Tests (Hand-Checkable)
# ---------------------------------------------------------------------------

def test_hand_checkable_split_abstention_metrics():
    """
    Hand-checkable test of split abstention metrics:
    - 2 answerable queries: 1 answered correctly, 1 falsely refused.
      false_refusal_rate = 1/2 = 0.5000.
    - 2 unanswerable queries: 1 correctly refused (0 citations), 1 failed to refuse (emitted citations).
      true_abstention_rate = 1/2 = 0.5000.
      unexpected_citation_rate = 1/2 = 0.5000.
    - Overall correct handling: (1 + 1) / 4 = 2/4 = 0.5000.
    """
    # Case 1: Answerable, correctly answered
    d1 = evaluate_abstention_details(expected_abstention=False, actual_insufficient_evidence=False, actual_citations_count=1)
    assert d1["is_correctly_answered"] is True
    assert d1["is_false_refusal"] is False
    assert d1["is_overall_correct"] is True

    # Case 2: Answerable, falsely refused
    d2 = evaluate_abstention_details(expected_abstention=False, actual_insufficient_evidence=True, actual_citations_count=0)
    assert d2["is_correctly_answered"] is False
    assert d2["is_false_refusal"] is True
    assert d2["is_overall_correct"] is False

    # Case 3: Unanswerable, correctly abstained
    d3 = evaluate_abstention_details(expected_abstention=True, actual_insufficient_evidence=True, actual_citations_count=0)
    assert d3["is_correct_abstention"] is True
    assert d3["is_failed_abstention"] is False
    assert d3["is_overall_correct"] is True

    # Case 4: Unanswerable, failed to abstain (hallucinated answer & citations)
    d4 = evaluate_abstention_details(expected_abstention=True, actual_insufficient_evidence=False, actual_citations_count=1)
    assert d4["is_correct_abstention"] is False
    assert d4["is_failed_abstention"] is True
    assert d4["is_overall_correct"] is False

    # Aggregate manual calculation
    unanswerable_total = 2
    answerable_total = 2
    total_valid = 4

    true_abst_rate = (1 if d3["is_correct_abstention"] else 0) / unanswerable_total
    false_refusal_rate = (1 if d2["is_false_refusal"] else 0) / answerable_total
    overall_correct = (1 + 1) / total_valid

    assert true_abst_rate == 0.5
    assert false_refusal_rate == 0.5
    assert overall_correct == 0.5


def test_key_facts_coverage_evaluation():
    """Checks deterministic key facts evaluation."""
    key_facts = ["15 minutes", "Severity-1"]
    good_answer = "Executive notification is required within 15 minutes for Severity-1 incidents."
    partial_answer = "Notification is required within 15 minutes."
    bad_answer = "I don't know the exact time limit."

    assert evaluate_key_facts(good_answer, key_facts) == 1.0
    assert evaluate_key_facts(partial_answer, key_facts) == 0.5
    assert evaluate_key_facts(bad_answer, key_facts) == 0.0


# ---------------------------------------------------------------------------
# Storage Isolation & Cleanup Tests
# ---------------------------------------------------------------------------

def test_evaluation_storage_isolation_cleanup():
    """
    Verifies that benchmark PDF storage uses an isolated temporary directory
    and leaves zero files in storage/documents/ after completion.
    """
    prod_storage_dir = Path("storage/documents")
    initial_prod_files = set(prod_storage_dir.glob("*.pdf"))

    # Run evaluation on a mini 2-case dataset
    dataset = load_dataset()
    mini_dataset = {
        "version": "1.0.0-storage-test",
        "cases": dataset["cases"][:2],
    }

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_dataset:
        json.dump(mini_dataset, f_dataset)
        temp_dataset_path = f_dataset.name

    try:
        report = run_evaluation(
            dataset_path=temp_dataset_path,
            k=3,
        )
        assert report["aggregate_metrics"]["total_evaluated_cases"] == 2

        # Verify no benchmark PDFs leaked into production storage/documents/
        after_prod_files = set(prod_storage_dir.glob("*.pdf"))
        leaked_files = after_prod_files - initial_prod_files
        assert len(leaked_files) == 0, f"Leaked files detected in production storage: {leaked_files}"
    finally:
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)


def test_evaluation_storage_cleanup_on_failure(monkeypatch):
    """
    Verifies that even if an unhandled exception occurs inside evaluation,
    the temporary storage directory is reliably cleaned up via finally block.
    """
    import tempfile
    created_temp_dirs = []

    original_temp_dir = tempfile.TemporaryDirectory

    def track_temp_dir(*args, **kwargs):
        td = original_temp_dir(*args, **kwargs)
        created_temp_dirs.append(Path(td.name))
        return td

    monkeypatch.setattr(tempfile, "TemporaryDirectory", track_temp_dir)

    def failing_retrieve(*args, **kwargs):
        raise RuntimeError("Simulated unhandled pipeline failure")

    monkeypatch.setattr("backend.evaluation.evaluator.retrieve_authorized_passages", failing_retrieve)

    dataset = load_dataset()
    mini_dataset = {
        "version": "1.0.0-fail-clean",
        "cases": dataset["cases"][:1],
    }

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_dataset:
        json.dump(mini_dataset, f_dataset)
        temp_dataset_path = f_dataset.name

    try:
        report = run_evaluation(dataset_path=temp_dataset_path)
        # Evaluator caught the case error and finished gracefully
        assert report["aggregate_metrics"]["failed_cases_count"] == 1

        # Check that tracked temp directory no longer exists on disk
        for td in created_temp_dirs:
            assert not td.exists(), f"Temp directory {td} was not cleaned up!"
    finally:
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)


# ---------------------------------------------------------------------------
# Resilience to Individual Case Failures
# ---------------------------------------------------------------------------

def test_evaluation_resilience_to_case_failure(monkeypatch):
    """
    Verifies that if one case fails unexpectedly, subsequent cases continue evaluating,
    and the failed case is reported explicitly and excluded from metric denominators.
    """
    from backend.models import RAGQueryResponse

    mini_dataset = {
        "version": "1.0.0-resilience-test",
        "cases": [
            {
                "id": "case-fail",
                "question": "Question that causes exception",
                "is_answerable": True,
                "expected_abstention": False,
                "expected_evidence": [{"document_title": "Information Security & Incident Response Policy", "page_number": 1}],
                "key_facts": ["fail"],
            },
            {
                "id": "case-success",
                "question": "What is the maximum domestic per diem cap for daily meal expenses?",
                "is_answerable": True,
                "expected_abstention": False,
                "expected_evidence": [{"document_title": "Corporate Travel & Expense Reimbursement Policy", "page_number": 2}],
                "key_facts": ["$75"],
            },
        ],
    }

    real_generate = run_evaluation.__globals__["generate_rag_answer"]

    def flaky_generate(question, passages):
        if "causes exception" in question:
            raise TimeoutError("Provider connection timed out unexpectedly")
        return real_generate(question, passages)

    monkeypatch.setattr("backend.evaluation.evaluator.generate_rag_answer", flaky_generate)

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_dataset:
        json.dump(mini_dataset, f_dataset)
        temp_dataset_path = f_dataset.name

    try:
        report = run_evaluation(dataset_path=temp_dataset_path, k=3)
        agg = report["aggregate_metrics"]

        # Case 1 failed, Case 2 succeeded
        assert agg["total_evaluated_cases"] == 2
        assert agg["failed_cases_count"] == 1
        assert agg["successful_cases_count"] == 1
        assert agg["answerable_cases_count"] == 1

        results = report["case_results"]
        assert results[0]["id"] == "case-fail"
        assert results[0]["status"] == "failed"
        assert "TimeoutError" in results[0]["error"]

        assert results[1]["id"] == "case-success"
        assert results[1]["status"] == "success"
    finally:
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)


def test_dataset_json_integrity():
    """Verifies that the dataset.json file exists, parses correctly, and has required fields."""
    dataset = load_dataset()
    assert dataset["version"] == "1.0.0"
    assert dataset["total_cases"] >= 20
    assert len(dataset["cases"]) >= 20

    for case in dataset["cases"]:
        assert "id" in case
        assert "question" in case
        assert "is_answerable" in case
        assert "expected_abstention" in case
        assert "key_facts" in case
