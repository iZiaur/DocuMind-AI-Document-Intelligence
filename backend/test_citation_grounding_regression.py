import pytest
from backend.models import SearchPassageResponse
from backend.rag import (
    validate_and_extract_citations,
    format_context_prompt,
    score_passage_against_answer,
    extract_sentence_excerpt,
)


@pytest.fixture
def remote_work_passages():
    """Sample passages mirroring an enterprise Remote Work Policy document."""
    p_meta = SearchPassageResponse(
        chunk_id="chunk-meta-001",
        document_id="doc-rwp-2026",
        document_title="Enterprise Remote Work Policy",
        page_number=1,
        chunk_index=0,
        text=(
            "Title: Enterprise Remote Work Policy. Document ID: POL-2024-03. "
            "Effective Date: January 15, 2024. Applicability: All regular and probationary employees."
        ),
        score=0.88,
    )
    p_new_hires = SearchPassageResponse(
        chunk_id="chunk-onboarding-002",
        document_id="doc-rwp-2026",
        document_title="Enterprise Remote Work Policy",
        page_number=2,
        chunk_index=1,
        text=(
            "Section 3.1 Onboarding Requirements: New employees are required to work from the office "
            "at least four days per week during their initial 90-day probationary period to facilitate mentoring."
        ),
        score=0.84,
    )
    p_remote_rules = SearchPassageResponse(
        chunk_id="chunk-remote-rules-003",
        document_id="doc-rwp-2026",
        document_title="Enterprise Remote Work Policy",
        page_number=2,
        chunk_index=2,
        text=(
            "Section 3.2 General Hybrid Operations: Regular full-time employees may work remotely up to "
            "three days per week with manager approval. Working remotely on designated core collaboration "
            "days requires VP sign-off."
        ),
        score=0.81,
    )
    return [p_meta, p_new_hires, p_remote_rules]


def test_remote_work_policy_grounding_selects_true_evidence_not_metadata_or_distractor(remote_work_passages):
    """
    Regression Test for Observed Issue:
    Question: 'How many days per week can employees work remotely?'
    Answer: 'Employees may work remotely up to three days per week with manager approval.'

    Verifies:
    1. The true supporting passage (chunk-remote-rules-003) is selected.
    2. The document metadata chunk (chunk-meta-001) is excluded.
    3. The 4-day office probation distractor (chunk-onboarding-002) is excluded.
    4. The excerpt explicitly contains the supporting statement.
    """
    answer = "Employees may work remotely up to three days per week with manager approval."
    citations, clean_ans = validate_and_extract_citations(answer, remote_work_passages)

    assert len(citations) == 1, f"Expected exactly 1 supporting citation, got {len(citations)}"
    primary = citations[0]
    assert primary.chunk_id == "chunk-remote-rules-003"
    assert primary.page_number == 2
    assert "three days per week with manager approval" in primary.excerpt
    assert clean_ans == answer

    # Explicitly ensure neither metadata nor probation distractor were attached
    cited_ids = {c.chunk_id for c in citations}
    assert "chunk-meta-001" not in cited_ids
    assert "chunk-onboarding-002" not in cited_ids


def test_multi_part_answer_includes_multiple_supporting_citations_only_when_useful(remote_work_passages):
    """
    Verifies that multiple citations are returned ONLY when multiple distinct claims
    in the answer are supported by different passages.
    """
    answer = (
        "Regular employees may work remotely up to three days per week with manager approval. "
        "However, new employees must work from the office at least four days per week during onboarding."
    )
    citations, clean_ans = validate_and_extract_citations(answer, remote_work_passages)

    assert len(citations) == 2, f"Expected 2 citations for dual-claim answer, got {len(citations)}"
    cited_ids = {c.chunk_id for c in citations}
    assert "chunk-remote-rules-003" in cited_ids
    assert "chunk-onboarding-002" in cited_ids
    assert "chunk-meta-001" not in cited_ids


def test_single_claim_does_not_attach_spurious_second_citation(remote_work_passages):
    """
    Verifies that an answer making only a single supported claim is not padded
    with an irrelevant second citation simply because a second chunk was retrieved.
    """
    answer = "Regular employees may work remotely up to three days per week with manager approval."
    citations, _ = validate_and_extract_citations(answer, remote_work_passages)

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-remote-rules-003"


def test_explicit_passage_index_citations_mapping(remote_work_passages):
    """
    Verifies that explicit citations in formats [3], [source: 3], or [passage 3]
    map correctly to the 3rd retrieved passage and clean the bracket tags.
    """
    answer_with_tag = "Employees may work remotely up to three days per week with manager approval [3]."
    citations, clean_ans = validate_and_extract_citations(answer_with_tag, remote_work_passages)

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-remote-rules-003"
    assert clean_ans == "Employees may work remotely up to three days per week with manager approval."


def test_explicit_uuid_citations_mapping(remote_work_passages):
    """
    Verifies that explicit citations in format [source: <chunk_id>] map correctly
    and clean the bracket tags.
    """
    answer_with_tag = (
        "Employees may work remotely up to three days per week with manager approval "
        "[source: chunk-remote-rules-003]."
    )
    citations, clean_ans = validate_and_extract_citations(answer_with_tag, remote_work_passages)

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-remote-rules-003"
    assert clean_ans == "Employees may work remotely up to three days per week with manager approval."


def test_hallucinated_explicit_citation_discarded(remote_work_passages):
    """
    Verifies that an explicit citation token referencing an unretrieved chunk ID
    is rejected and grounded evidence is chosen instead.
    """
    answer_with_fake = (
        "Employees may work remotely up to three days per week with manager approval "
        "[source: chunk-hallucinated-999]."
    )
    citations, clean_ans = validate_and_extract_citations(answer_with_fake, remote_work_passages)

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-remote-rules-003"
    assert all(c.chunk_id != "chunk-hallucinated-999" for c in citations)


def test_refusal_and_unrelated_answers_return_empty_citations(remote_work_passages):
    """
    Verifies that refusal answers and completely unrelated answers return zero citations.
    """
    refusal = "I do not have sufficient information in the provided documents to answer this question."
    cits_refusal, _ = validate_and_extract_citations(refusal, remote_work_passages)
    assert len(cits_refusal) == 0

    unrelated = "Chocolate chip cookies require flour, sugar, butter, and vanilla extract."
    cits_unrelated, _ = validate_and_extract_citations(unrelated, remote_work_passages)
    assert len(cits_unrelated) == 0


def test_centered_excerpt_preserves_supporting_sentence_in_long_chunk():
    """
    Verifies that extract_sentence_excerpt centers around the supporting sentence
    even when preceded by a long preamble, preventing truncation of the actual evidence.
    """
    long_preamble = (
        "Section 7.4 Historical Preamble and Operational Tenets. The company was founded with an "
        "emphasis on employee well-being, trust, and flexibility across all regional operating branches. "
        "All procedures must conform to standard organizational workflows and local labor compliance. "
    )
    supporting_statement = "Employees may work remotely up to three days per week with manager approval."
    following_text = " Please consult your direct team lead for scheduling adjustments."
    full_text = long_preamble + supporting_statement + following_text

    excerpt = extract_sentence_excerpt(full_text, supporting_statement, max_len=220)
    assert "three days per week with manager approval" in excerpt
    assert len(excerpt) <= 230


def test_format_context_prompt_includes_index_and_attributes(remote_work_passages):
    """
    Verifies format_context_prompt outputs index, id, document, and page attributes.
    """
    prompt = format_context_prompt(remote_work_passages)
    assert '<passage id="chunk-meta-001" index="1" document="Enterprise Remote Work Policy" page="1">' in prompt
    assert '<passage id="chunk-onboarding-002" index="2" document="Enterprise Remote Work Policy" page="2">' in prompt
    assert '<passage id="chunk-remote-rules-003" index="3" document="Enterprise Remote Work Policy" page="2">' in prompt


def test_deployed_issue_reproduction_clean_answer_and_valid_citation(remote_work_passages):
    """
    Direct test for the observed issue:
    Input answer: 'Employees may work remotely up to three days per week with manager approval [source: 1, 3].'
    Verifies:
    1. '[source: 1, 3]' marker is completely stripped from clean_answer.
    2. clean_answer equals 'Employees may work remotely up to three days per week with manager approval.'
    3. Exactly 1 citation (chunk-remote-rules-003) is retained because chunk-meta-001 (passage 1) is metadata.
    4. Excerpt contains the exact supporting sentence.
    """
    raw_answer = "Employees may work remotely up to three days per week with manager approval [source: 1, 3]."
    citations, clean_ans = validate_and_extract_citations(raw_answer, remote_work_passages)

    assert clean_ans == "Employees may work remotely up to three days per week with manager approval."
    assert "[source:" not in clean_ans
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-remote-rules-003"
    assert "three days per week with manager approval" in citations[0].excerpt


def test_remove_citation_marker_various_formats(remote_work_passages):
    """
    Verifies stripping of all supported internal citation formats:
    - [source: 1]
    - [source: 1, 3]
    - [passage 1]
    - [passages: 1, 2]
    - [source: <UUID>]
    - [<UUID>]
    - [1]
    - [1, 3]
    """
    cases = [
        (
            "Employees may work remotely up to three days per week [source: 3].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [source: 1, 3].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [passage 3].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [passages: 1, 3].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [source: chunk-remote-rules-003].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [chunk-remote-rules-003].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [3].",
            "Employees may work remotely up to three days per week.",
        ),
        (
            "Employees may work remotely up to three days per week [1, 3].",
            "Employees may work remotely up to three days per week.",
        ),
    ]

    for raw, expected in cases:
        citations, clean_ans = validate_and_extract_citations(raw, remote_work_passages)
        assert clean_ans == expected, f"Failed on {raw}: got {clean_ans!r}"
        assert len(citations) >= 1
        assert citations[0].chunk_id == "chunk-remote-rules-003"


def test_preserve_ordinary_bracketed_text(remote_work_passages):
    """
    Verifies that legitimate, non-citation bracketed content in generated answers is preserved:
    - [Note: manager approval required]
    - [e.g., domestic travel]
    - [SOC 2 Type II]
    - [2024]
    """
    answer_with_note = (
        "Employees may work remotely up to three days per week [Note: manager approval required] "
        "[source: chunk-remote-rules-003]."
    )
    citations, clean_ans = validate_and_extract_citations(answer_with_note, remote_work_passages)
    assert "[Note: manager approval required]" in clean_ans
    assert "[source:" not in clean_ans
    assert len(citations) == 1

    answer_with_eg = (
        "Employees may work remotely up to three days per week [e.g., Tuesday through Thursday] [3]."
    )
    citations, clean_ans = validate_and_extract_citations(answer_with_eg, remote_work_passages)
    assert "[e.g., Tuesday through Thursday]" in clean_ans
    assert "[3]" not in clean_ans

    answer_with_soc = "The system is compliant with [SOC 2 Type II] standards [source: chunk-remote-rules-003]."
    citations, clean_ans = validate_and_extract_citations(answer_with_soc, remote_work_passages)
    assert "[SOC 2 Type II]" in clean_ans
    assert "[source:" not in clean_ans

    answer_with_year = "The policy was instituted in [2024] [source: chunk-remote-rules-003]."
    citations, clean_ans = validate_and_extract_citations(answer_with_year, remote_work_passages)
    assert "[2024]" in clean_ans
    assert "[source:" not in clean_ans
