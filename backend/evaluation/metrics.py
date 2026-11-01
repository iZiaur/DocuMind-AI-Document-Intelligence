"""
Evaluation metrics for DocuMind AI Stage 5.
Implements:
1. Retrieval metrics: Recall@k, Precision@k, Mean Reciprocal Rank (MRR).
2. Answer & citation metrics: Key fact coverage, citation precision/recall, abstention accuracy.
"""

from typing import List, Dict, Any, Optional, Tuple, Set


def match_evidence(
    doc_title: str,
    page_num: int,
    expected_evidence: List[Dict[str, Any]],
    chunk_id: Optional[str] = None,
) -> bool:
    """
    Determines if a document title and page number match any expected evidence item.
    Relevance is page-level by default to avoid false penalties from chunking boundaries.
    If chunk_id is explicitly provided in an expected evidence item, chunk-level match is checked.
    """
    for exp in expected_evidence:
        exp_title = exp.get("document_title", "").strip().lower()
        exp_page = exp.get("page_number")
        exp_chunk = exp.get("chunk_id")

        if exp_chunk and chunk_id:
            if exp_chunk == chunk_id:
                return True

        if exp_title and exp_title == doc_title.strip().lower():
            if exp_page is None or exp_page == page_num:
                return True

    return False


def compute_recall_at_k(
    retrieved_passages: List[Any],
    expected_evidence: List[Dict[str, Any]],
    k: int = 3,
) -> Optional[float]:
    """
    Recall@k:
    Proportion of expected evidence targets successfully retrieved in the top-k passages.
    Returns None if expected_evidence is empty (e.g. unanswerable / abstention queries).
    """
    if not expected_evidence:
        return None

    if k <= 0:
        return 0.0

    # Distinct expected targets by (title, page)
    expected_targets: Set[Tuple[str, int]] = {
        (e.get("document_title", "").strip().lower(), e.get("page_number", 0))
        for e in expected_evidence
        if e.get("document_title")
    }

    if not expected_targets:
        return None

    top_passages = retrieved_passages[:k]
    matched_targets: Set[Tuple[str, int]] = set()

    for p in top_passages:
        doc_title = getattr(p, "document_title", "") if hasattr(p, "document_title") else p.get("document_title", "")
        page_num = getattr(p, "page_number", 0) if hasattr(p, "page_number") else p.get("page_number", 0)
        norm_title = doc_title.strip().lower()

        for exp_title, exp_page in expected_targets:
            if norm_title == exp_title and (exp_page == 0 or exp_page == page_num):
                matched_targets.add((exp_title, exp_page))

    return round(len(matched_targets) / len(expected_targets), 4)


def compute_precision_at_k(
    retrieved_passages: List[Any],
    expected_evidence: List[Dict[str, Any]],
    k: int = 3,
) -> Optional[float]:
    """
    Precision@k:
    Proportion of top-k retrieved passages that contain relevant expected evidence.
    Returns None if expected_evidence is empty (unanswerable / abstention queries).
    """
    if not expected_evidence:
        return None

    if k <= 0:
        return 0.0

    top_passages = retrieved_passages[:k]
    if not top_passages:
        return 0.0

    relevant_count = 0
    for p in top_passages:
        doc_title = getattr(p, "document_title", "") if hasattr(p, "document_title") else p.get("document_title", "")
        page_num = getattr(p, "page_number", 0) if hasattr(p, "page_number") else p.get("page_number", 0)
        chunk_id = getattr(p, "chunk_id", None) if hasattr(p, "chunk_id") else p.get("chunk_id", None)

        if match_evidence(doc_title, page_num, expected_evidence, chunk_id=chunk_id):
            relevant_count += 1

    return round(relevant_count / len(top_passages), 4)


def compute_reciprocal_rank(
    retrieved_passages: List[Any],
    expected_evidence: List[Dict[str, Any]],
    k: int = 3,
) -> Optional[float]:
    """
    Reciprocal Rank (RR):
    1 / (rank of first relevant passage retrieved in top-k).
    Returns 0.0 if no relevant passage appears in top-k.
    Returns None if expected_evidence is empty (unanswerable queries).
    """
    if not expected_evidence:
        return None

    if k <= 0:
        return 0.0

    top_passages = retrieved_passages[:k]
    for rank, p in enumerate(top_passages, start=1):
        doc_title = getattr(p, "document_title", "") if hasattr(p, "document_title") else p.get("document_title", "")
        page_num = getattr(p, "page_number", 0) if hasattr(p, "page_number") else p.get("page_number", 0)
        chunk_id = getattr(p, "chunk_id", None) if hasattr(p, "chunk_id") else p.get("chunk_id", None)

        if match_evidence(doc_title, page_num, expected_evidence, chunk_id=chunk_id):
            return round(1.0 / rank, 4)

    return 0.0


def evaluate_key_facts(
    answer: str,
    key_facts: List[str],
) -> float:
    """
    Evaluates what fraction of required key fact phrases are present in the answer.
    Case-insensitive substring matching.
    """
    if not key_facts:
        return 1.0

    if not answer:
        return 0.0

    lower_ans = answer.lower()
    matched = 0

    for fact in key_facts:
        if fact.lower() in lower_ans:
            matched += 1

    return round(matched / len(key_facts), 4)


def evaluate_citations(
    citations: List[Any],
    expected_evidence: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Evaluates citation precision and recall:
    - Citation Precision: Fraction of generated citations that match expected evidence.
      Averaged across ALL valid answerable cases (Option B). When an answerable question
      emits 0 citations, it receives precision = 0.0, penalizing missing evidence attribution
      and ensuring the case is included in the aggregate denominator.
      (Distinguished from conditional citation precision, which would exclude 0-citation cases).
    - Citation Recall: Fraction of expected evidence targets referenced by generated citations.
      Evaluated across all valid answerable cases with expected evidence.

    Unanswerable questions (where expected_evidence is empty) have no evidence to recall.
    They return recall=None to prevent artificially inflating answerable citation recall.
    If unexpected/hallucinated citations are generated on an unanswerable question,
    precision is 0.0 (0 of N citations are valid expected evidence).
    If no citations are generated, precision is None (not applicable).
    """
    # Case: Unanswerable question where no citations are expected
    if not expected_evidence:
        if citations:
            return 0.0, None  # Hallucinated citations on an unanswerable question
        return None, None    # Clean abstention; citation metrics not applicable

    expected_targets: Set[Tuple[str, int]] = {
        (e.get("document_title", "").strip().lower(), e.get("page_number", 0))
        for e in expected_evidence
        if e.get("document_title")
    }

    if not expected_targets:
        return None, None

    if not citations:
        return 0.0, 0.0

    # 1. Precision: How many citations point to expected evidence?
    valid_citations_count = 0
    covered_targets: Set[Tuple[str, int]] = set()

    for c in citations:
        doc_title = getattr(c, "document_title", "") if hasattr(c, "document_title") else c.get("document_title", "")
        page_num = getattr(c, "page_number", 0) if hasattr(c, "page_number") else c.get("page_number", 0)
        chunk_id = getattr(c, "chunk_id", None) if hasattr(c, "chunk_id") else c.get("chunk_id", None)

        if match_evidence(doc_title, page_num, expected_evidence, chunk_id=chunk_id):
            valid_citations_count += 1
            norm_title = doc_title.strip().lower()
            for exp_title, exp_page in expected_targets:
                if norm_title == exp_title and (exp_page == 0 or exp_page == page_num):
                    covered_targets.add((exp_title, exp_page))

    precision = round(valid_citations_count / len(citations), 4)
    recall = round(len(covered_targets) / len(expected_targets), 4)

    return precision, recall


def evaluate_abstention(
    expected_abstention: bool,
    actual_insufficient_evidence: bool,
    actual_citations_count: int,
) -> bool:
    """
    Backward-compatible overall correctness check for a single case:
    - If expected_abstention=True: True iff actual_insufficient_evidence=True and citations=0.
    - If expected_abstention=False: True iff actual_insufficient_evidence=False (not falsely refused).
    """
    if expected_abstention:
        return actual_insufficient_evidence is True and actual_citations_count == 0
    else:
        return actual_insufficient_evidence is False


def evaluate_abstention_details(
    expected_abstention: bool,
    actual_insufficient_evidence: bool,
    actual_citations_count: int,
) -> Dict[str, bool]:
    """
    Computes detailed case-level abstention flags:
    - is_correct_abstention: True if expected_abstention=True and correctly abstained with 0 citations.
    - is_failed_abstention: True if expected_abstention=True but failed to abstain or emitted citations.
    - is_false_refusal: True if expected_abstention=False but model falsely refused (insufficient_evidence=True).
    - is_correctly_answered: True if expected_abstention=False and model did not falsely refuse.
    - is_overall_correct: True if either correct abstention or correctly answered without false refusal.
    """
    if expected_abstention:
        correct_abst = (actual_insufficient_evidence is True and actual_citations_count == 0)
        return {
            "is_correct_abstention": correct_abst,
            "is_failed_abstention": not correct_abst,
            "is_false_refusal": False,
            "is_correctly_answered": False,
            "is_overall_correct": correct_abst,
        }
    else:
        false_refusal = (actual_insufficient_evidence is True)
        return {
            "is_correct_abstention": False,
            "is_failed_abstention": False,
            "is_false_refusal": false_refusal,
            "is_correctly_answered": not false_refusal,
            "is_overall_correct": not false_refusal,
        }
