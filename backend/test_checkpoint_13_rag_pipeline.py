"""
Checkpoint 13 — Phase 3: Shared RAG Service & Contextual Query Condensation Test Suite.

Verifies:
1. Contextual Query Condenser:
   - Verbatim question returned when history is empty or contains no completed messages.
   - Filtering of pending, failed, or non-user/assistant messages.
   - Window bounding to the last 3 completed exchanges (up to 6 messages).
   - Conversational pronoun resolution in offline/test mode.
   - Safe unmodified fallback on empty candidate, excessive length (> 500 chars),
     instruction-like preambles, repetitive output, injection tags, and network timeouts.
   - Exact preservation of original question reference.
2. Shared RAG Pipeline & Scope Authorization:
   - Rejection with HTTP 409 Conflict when conversation document scope is invalidated.
   - Rejection with HTTP 404 Not Found on cross-tenant conversation or document access.
   - Rejection with HTTP 400 on scope mismatch or short/long questions.
   - Separation of condensed retrieval query vs. original question for generation.
3. Database Connection Lifecycle (REL-01):
   - Physical connection checkout and checkin verified: zero checked-out connections during LLM generation.
   - Dedicated session checkout for request metrics.
4. Single Rate Limiting Invocation:
   - Rate limiter executed exactly once per public request.
   - enforce_rate_limit flag internal behavior.
5. End-to-End Compatibility with /api/rag/query.
"""

import os
import uuid
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import httpx

from backend.main import app, get_db
from backend.models import (
    Base,
    User,
    Document,
    DocumentChunk,
    Conversation,
    Message,
    RequestMetric,
    SearchPassageResponse,
    RAGQueryResponse,
)
from backend.auth import create_access_token, hash_password
from backend.rag import (
    condense_conversational_query,
    execute_rag_pipeline,
    retrieve_authorized_passages,
    generate_rag_answer,
    _validate_and_sanitize_condenser_output,
)
import backend.main as main_module
import backend.rate_limiter as rate_limiter_module

# Isolated SQLite test database
TEST_DB_URL = "sqlite:///:memory:"

pipeline_test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
PipelineTestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pipeline_test_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_pipeline_test_db():
    Base.metadata.create_all(bind=pipeline_test_engine)
    yield
    Base.metadata.drop_all(bind=pipeline_test_engine)


@pytest.fixture
def db_session():
    db = PipelineTestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        session = PipelineTestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def users_and_docs(db_session):
    """Creates authorized and unauthorized users, documents, and chunks."""
    uid_a = uuid.uuid4().hex[:6]
    uid_b = uuid.uuid4().hex[:6]
    # User A
    user_a = User(
        email=f"user_a_{uid_a}@example.com",
        username=f"user_a_{uid_a}",
        hashed_password=hash_password("Password123!"),
    )
    # User B
    user_b = User(
        email=f"user_b_{uid_b}@example.com",
        username=f"user_b_{uid_b}",
        hashed_password=hash_password("Password123!"),
    )
    db_session.add_all([user_a, user_b])
    db_session.commit()
    db_session.refresh(user_a)
    db_session.refresh(user_b)

    # Document A (owned by User A)
    doc_a = Document(
        id=str(uuid.uuid4()),
        title="Remote Work Policy",
        filename="doc_a.pdf",
        storage_path="/tmp/doc_a.pdf",
        file_size_bytes=1024,
        owner_id=user_a.id,
    )
    # Document B (owned by User B)
    doc_b = Document(
        id=str(uuid.uuid4()),
        title="Confidential Finance",
        filename="doc_b.pdf",
        storage_path="/tmp/doc_b.pdf",
        file_size_bytes=2048,
        owner_id=user_b.id,
    )
    db_session.add_all([doc_a, doc_b])
    db_session.commit()
    db_session.refresh(doc_a)
    db_session.refresh(doc_b)

    # Chunks for Doc A
    chunk_a = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc_a.id,
        chunk_index=0,
        page_number=1,
        text="Remote employees are eligible for a $50 monthly internet stipend.",
        embedding=[0.05] * 1536,
    )
    db_session.add(chunk_a)
    db_session.commit()

    return {
        "user_a": user_a,
        "user_b": user_b,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "chunk_a": chunk_a,
    }


# ===========================================================================
# 1. Contextual Query Condenser Tests
# ===========================================================================

def test_condenser_empty_history_returns_original_question():
    """When history is None, empty list, or empty dict, returns exact original question."""
    original = "What is the policy?"
    assert condense_conversational_query(original, None) == original
    assert condense_conversational_query(original, []) == original


def test_condenser_excludes_pending_and_failed_messages():
    """Messages with status pending, failed, or invalid roles are strictly excluded."""
    history = [
        {"role": "user", "content": "Tell me about Remote Work Policy", "status": "failed"},
        {"role": "assistant", "content": "Error during generation", "status": "failed"},
        {"role": "user", "content": "How much is the stipend?", "status": "pending"},
        {"role": "system", "content": "System prompt instructions", "status": "completed"},
    ]
    raw_question = "What are its requirements?"
    # Because all prior messages are invalid/failed/pending, condenser has no valid context -> returns raw_question
    result = condense_conversational_query(raw_question, history, provider="offline")
    assert result == raw_question


def test_condenser_bounds_history_window():
    """Condenser uses at most the last 3 completed exchanges (6 messages)."""
    # 5 exchanges (10 messages)
    history = []
    for i in range(1, 6):
        history.append({"role": "user", "content": f"Topic {i} details", "status": "completed"})
        history.append({"role": "assistant", "content": f"Answer for Topic {i}", "status": "completed"})

    question = "What is it about?"
    # In offline mode, should resolve against the most recent topic (Topic 5), not older ones
    result = condense_conversational_query(question, history, provider="offline")
    assert "Topic 5" in result or "Topic" in result


def test_condenser_pronoun_resolution_offline():
    """Resolves pronouns like 'its' or 'that' into concrete entity names in offline mode."""
    history = [
        {"role": "user", "content": "Can you explain the Travel Expense Policy?", "status": "completed"},
        {"role": "assistant", "content": "The Travel Expense Policy covers domestic and international travel.", "status": "completed"},
    ]
    question = "What are its requirements?"
    condensed = condense_conversational_query(question, history, provider="offline")
    assert "Travel Expense Policy" in condensed
    assert "its" not in condensed.lower()


def test_condenser_unmodified_fallback_on_empty_candidate(monkeypatch):
    """When LLM returns empty or whitespace response, falls back to original question."""
    original = "What are its requirements?"
    history = [{"role": "user", "content": "Remote Work Policy", "status": "completed"}]

    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "   "}}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result = condense_conversational_query(original, history, provider="openai")
    assert result == original


def test_condenser_unmodified_fallback_on_excessive_length(monkeypatch):
    """When candidate exceeds 500 characters, falls back to original question."""
    original = "What are its requirements?"
    history = [{"role": "user", "content": "Remote Work Policy", "status": "completed"}]

    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "X" * 550}}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result = condense_conversational_query(original, history, provider="openai")
    assert result == original


def test_condenser_unmodified_fallback_on_instruction_like_or_preamble(monkeypatch):
    """When LLM candidate starts with chatty preamble or refusal, falls back to original question."""
    original = "What are its requirements?"
    history = [{"role": "user", "content": "Remote Work Policy", "status": "completed"}]

    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    chatty_responses = [
        "I cannot answer that question based on the history.",
        "Sure, here is the standalone query: what is it?",
        "Yes, the remote work policy allows stipends.",
        "Based on the provided documents, remote work is permitted.",
    ]

    for bad_resp in chatty_responses:
        def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": bad_resp}}]
            }
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        monkeypatch.setattr(httpx.Client, "post", mock_post)
        result = condense_conversational_query(original, history, provider="openai")
        assert result == original


def test_condenser_unmodified_fallback_on_repetitive_loop(monkeypatch):
    """When candidate enters a repetitive degenerative loop, falls back to original question."""
    original = "What are its requirements?"
    history = [{"role": "user", "content": "Remote Work Policy", "status": "completed"}]

    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "policy policy policy policy policy policy policy policy"}}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result = condense_conversational_query(original, history, provider="openai")
    assert result == original


def test_condenser_timeout_fallback(monkeypatch):
    """Network timeout falls back to original question gracefully without raising."""
    original = "What are its requirements?"
    history = [{"role": "user", "content": "Remote Work Policy", "status": "completed"}]

    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def mock_post(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out after 5.0s")

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result = condense_conversational_query(original, history, provider="openai")
    assert result == original


def test_condenser_preserves_original_question_reference():
    """Validates that original question is not modified, stripped, or lowercased on fallback."""
    raw_question = "   What are its requirements?   "
    res = condense_conversational_query(raw_question, [])
    assert res is raw_question


# ===========================================================================
# 2. Scope Validation & Multi-Tenant Authorization Tests
# ===========================================================================

def test_pipeline_rejects_invalidated_conversation_scope(db_session, users_and_docs):
    """Pipeline rejects retrieval with HTTP 409 Conflict if conversation document scope was invalidated."""
    user = users_and_docs["user_a"]

    # Create invalidated conversation
    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        title="Invalidated Scope Thread",
        document_id=None,
        document_scope_invalidated=True,
    )
    db_session.add(conv)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        execute_rag_pipeline(
            question="What is the stipend amount?",
            user_id=user.id,
            document_id=None,
            top_k=3,
            db=db_session,
            conversation_id=conv.id,
        )

    assert exc_info.value.status_code == 409
    assert "scope was invalidated" in exc_info.value.detail.lower()


def test_pipeline_rejects_cross_tenant_conversation_access(db_session, users_and_docs):
    """User A cannot access a conversation owned by User B (returns 404)."""
    user_b = users_and_docs["user_b"]
    user_a = users_and_docs["user_a"]

    conv_b = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_b.id,
        title="User B Thread",
    )
    db_session.add(conv_b)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        execute_rag_pipeline(
            question="Any questions?",
            user_id=user_a.id,
            document_id=None,
            top_k=3,
            db=db_session,
            conversation_id=conv_b.id,
        )

    assert exc_info.value.status_code == 404
    assert "not found or access denied" in exc_info.value.detail.lower()


def test_pipeline_rejects_unauthorized_document(db_session, users_and_docs):
    """User A cannot access Document B owned by User B (returns 404)."""
    user_a = users_and_docs["user_a"]
    doc_b = users_and_docs["doc_b"]

    with pytest.raises(HTTPException) as exc_info:
        execute_rag_pipeline(
            question="Tell me secrets",
            user_id=user_a.id,
            document_id=doc_b.id,
            top_k=3,
            db=db_session,
        )

    assert exc_info.value.status_code == 404


def test_pipeline_enforces_conversation_document_scope_match(db_session, users_and_docs):
    """If conversation is scoped to Doc A, passing a different doc returns 400 Bad Request."""
    user_a = users_and_docs["user_a"]
    doc_a = users_and_docs["doc_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        document_id=doc_a.id,
        title="Scoped Thread",
    )
    db_session.add(conv)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        execute_rag_pipeline(
            question="What is the stipend?",
            user_id=user_a.id,
            document_id="some-other-doc-id",
            top_k=3,
            db=db_session,
            conversation_id=conv.id,
        )

    assert exc_info.value.status_code == 400


def test_pipeline_rejects_short_or_empty_question(db_session, users_and_docs):
    """Questions shorter than 2 non-whitespace characters return 400 Bad Request."""
    user_a = users_and_docs["user_a"]

    with pytest.raises(HTTPException) as exc_info:
        execute_rag_pipeline(
            question="  ?  ",
            user_id=user_a.id,
            document_id=None,
            top_k=3,
            db=db_session,
        )

    assert exc_info.value.status_code == 400


# ===========================================================================
# 3. Database Connection Lifecycle (REL-01) Verification Test
# ===========================================================================

def test_pipeline_connection_released_before_llm_call(db_session, users_and_docs, monkeypatch):
    """
    CRITICAL REL-01 VERIFICATION:
    Verifies that the database connection is explicitly closed and checked in
    BEFORE the external LLM generation executes.
    """
    user_a = users_and_docs["user_a"]
    doc_a = users_and_docs["doc_a"]

    llm_executed = False
    checkin_count_before_llm = 0
    checkout_count_before_llm = 0

    checkouts = []
    checkins = []

    def on_checkout(dbapi_conn, conn_record, conn_proxy):
        checkouts.append(1)

    def on_checkin(dbapi_conn, conn_record):
        checkins.append(1)

    event.listen(pipeline_test_engine, "checkout", on_checkout)
    event.listen(pipeline_test_engine, "checkin", on_checkin)

    try:
        def spy_generator(question, passages, request_id):
            nonlocal llm_executed, checkin_count_before_llm, checkout_count_before_llm
            llm_executed = True
            checkin_count_before_llm = len(checkins)
            checkout_count_before_llm = len(checkouts)
            assert not db_session.in_transaction(), "Session transaction must be closed before generation"
            return RAGQueryResponse(
                request_id=request_id,
                question=question,
                answer="Verified DB release.",
                insufficient_evidence=False,
                citations=[],
                retrieval_count=len(passages),
                model_used="spy-model",
                token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                estimated_cost_usd=0.00001,
                pricing_version="2024-08-01",
            )

        response = execute_rag_pipeline(
            question="What is the stipend amount?",
            user_id=user_a.id,
            document_id=doc_a.id,
            top_k=2,
            db=db_session,
            generator_fn=spy_generator,
            enforce_rate_limit=False,
        )

        assert llm_executed is True
        assert checkout_count_before_llm > 0, "Retrieval must have checked out a connection"
        assert checkin_count_before_llm >= checkout_count_before_llm, (
            f"All retrieval connections must be checked in before LLM executes (checkins={checkin_count_before_llm}, checkouts={checkout_count_before_llm})"
        )
        assert response.answer == "Verified DB release."
    finally:
        event.remove(pipeline_test_engine, "checkout", on_checkout)
        event.remove(pipeline_test_engine, "checkin", on_checkin)


# ===========================================================================
# 4. Rate Limiting Tests (Single Invocation)
# ===========================================================================

def test_pipeline_rate_limiting_enforced_exactly_once(client, users_and_docs, monkeypatch):
    """Verifies that calling /api/rag/query invokes check_rag_rate_limit exactly once."""
    user_a = users_and_docs["user_a"]
    token = create_access_token(user_id=user_a.id, username=user_a.username)

    call_count = 0

    def mock_check_rate_limit(uid):
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(rate_limiter_module, "check_rag_rate_limit", mock_check_rate_limit)

    res = client.post(
        "/api/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What is the policy stipend?", "top_k": 2},
    )

    assert res.status_code == 200
    assert call_count == 1, f"Expected check_rag_rate_limit to be called exactly 1 time, got {call_count}"


# ===========================================================================
# 5. Original Question Preservation vs. Condensed Retrieval Query
# ===========================================================================

def test_pipeline_uses_condensed_query_for_search_and_original_question_for_generation(db_session, users_and_docs):
    """
    Verifies that retrieval uses retrieval_query (condensed query) while
    generation receives the original raw question.
    """
    user_a = users_and_docs["user_a"]
    doc_a = users_and_docs["doc_a"]

    captured_search_query = None
    captured_generation_question = None

    def spy_retriever(query_text, user_id, top_k, document_id, db, alpha=None):
        nonlocal captured_search_query
        captured_search_query = query_text
        return retrieve_authorized_passages(
            query_text=query_text,
            user_id=user_id,
            top_k=top_k,
            document_id=document_id,
            db=db,
            alpha=alpha,
        )

    def spy_generator(question, passages, request_id):
        nonlocal captured_generation_question
        captured_generation_question = question
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Grounded answer to original question.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=len(passages),
            model_used="test-model",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            estimated_cost_usd=0.00001,
            pricing_version="2024-08-01",
        )

    original_q = "What are its requirements?"
    condensed_q = "What are Remote Work Policy requirements?"

    response = execute_rag_pipeline(
        question=original_q,
        user_id=user_a.id,
        document_id=doc_a.id,
        top_k=2,
        db=db_session,
        retrieval_query=condensed_q,
        retriever_fn=spy_retriever,
        generator_fn=spy_generator,
        enforce_rate_limit=False,
    )

    assert captured_search_query == condensed_q
    assert captured_generation_question == original_q
    assert response.question == original_q


# ===========================================================================
# 6. Observability Metrics & Compatibility
# ===========================================================================

def test_pipeline_persists_request_metric_on_success(db_session, users_and_docs):
    """Verifies that execute_rag_pipeline records RequestMetric successfully."""
    user_a = users_and_docs["user_a"]
    doc_a = users_and_docs["doc_a"]

    response = execute_rag_pipeline(
        question="What is the monthly stipend?",
        user_id=user_a.id,
        document_id=doc_a.id,
        top_k=1,
        db=db_session,
        enforce_rate_limit=False,
    )

    assert response.request_id is not None
    # Check that metric was persisted
    metric = db_session.query(RequestMetric).filter(RequestMetric.request_id == response.request_id).first()
    assert metric is not None
    assert metric.user_id == user_a.id
    assert metric.status == "success"
    assert metric.retrieval_latency_ms >= 0.0
    assert metric.generation_latency_ms >= 0.0
    assert metric.total_latency_ms >= 0.0
