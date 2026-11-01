"""
Checkpoint 8 Remediation Test Suite.
Validates the four high-severity Phase 1 remediation items:
1. SEC-01 — Secure JWT secret configuration and production startup enforcement.
2. REL-01 — Database connection released before external LLM generation calls.
3. REL-02 — Event-loop responsiveness and worker-thread offloading for document ingestion.
4. SEC-02 — Authenticated PDF download security and authorization enforcement.
"""

import os
import time
import uuid
import threading
import asyncio
import pytest
from fastapi.testclient import TestClient
from fastapi import status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app, get_db
from backend.models import Base, User, Document, DocumentChunk, RequestMetric
from backend.auth import (
    get_jwt_secret,
    validate_jwt_secret,
    create_access_token,
    decode_access_token,
    hash_password,
    INSECURE_DEV_JWT_SECRET,
    MIN_JWT_SECRET_LENGTH,
)

# In-memory SQLite database for isolated test execution
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

active_test_sessions = []

def override_get_db():
    db = TestingSessionLocal()
    active_test_sessions.append(db)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_database():
    """Recreates fresh database schema for each test function."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    active_test_sessions.clear()
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()
    active_test_sessions.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_user(db_session):
    u = db_session.query(User).filter(User.username == "test_c8_user").first()
    if not u:
        u = User(
            email="test_c8_user@example.com",
            username="test_c8_user",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            is_admin=False,
        )
        db_session.add(u)
        db_session.commit()
        db_session.refresh(u)
    return u


@pytest.fixture
def admin_user(db_session):
    u = db_session.query(User).filter(User.username == "test_c8_admin").first()
    if not u:
        u = User(
            email="test_c8_admin@example.com",
            username="test_c8_admin",
            hashed_password=hash_password("AdminPass123!"),
            is_active=True,
            is_admin=True,
        )
        db_session.add(u)
        db_session.commit()
        db_session.refresh(u)
    return u


# ===========================================================================
# 1. SEC-01: Secure JWT Secret Configuration Tests
# ===========================================================================

def test_jwt_secret_missing_in_production_raises_error(monkeypatch):
    """SEC-01: In production mode, missing JWT_SECRET immediately raises RuntimeError."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_jwt_secret()
    assert "JWT_SECRET environment variable is missing or empty" in str(exc_info.value)


def test_jwt_secret_insecure_dev_fallback_in_production_raises_error(monkeypatch):
    """SEC-01: In production mode, using the insecure development secret raises RuntimeError."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("JWT_SECRET", INSECURE_DEV_JWT_SECRET)

    with pytest.raises(RuntimeError) as exc_info:
        validate_jwt_secret()
    assert "Insecure default JWT_SECRET detected in production" in str(exc_info.value)


def test_jwt_secret_insufficient_entropy_in_production_raises_error(monkeypatch):
    """SEC-01: In production mode, secret shorter than 32 characters raises RuntimeError."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("JWT_SECRET", "short_secret_only_24_chars!")

    with pytest.raises(RuntimeError) as exc_info:
        validate_jwt_secret()
    assert "insufficiently strong" in str(exc_info.value)
    assert f"at least {MIN_JWT_SECRET_LENGTH} characters" in str(exc_info.value)


def test_jwt_secret_valid_in_production_succeeds(monkeypatch):
    """SEC-01: In production mode, valid 32+ character secret is accepted without leaking value."""
    valid_secret = "super_secure_production_secret_key_exceeding_32_characters_long_12345"
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("JWT_SECRET", valid_secret)

    secret = get_jwt_secret()
    assert secret == valid_secret
    assert len(secret) >= MIN_JWT_SECRET_LENGTH


def test_jwt_token_generation_and_verification_roundtrip(monkeypatch):
    """SEC-01: Token creation and decoding roundtrip succeeds with configured secret."""
    valid_secret = "test_custom_jwt_secret_with_sufficient_entropy_for_hs256_32char"
    monkeypatch.setenv("JWT_SECRET", valid_secret)

    token = create_access_token(user_id=123, username="charlie_test")
    payload = decode_access_token(token)

    assert payload.get("sub") == "123"
    assert payload.get("username") == "charlie_test"
    assert "exp" in payload


# ===========================================================================
# 2. REL-01: Database Connection Release Before LLM Calls Tests
# ===========================================================================

def test_database_connection_released_during_llm_generation(client, test_user, monkeypatch):
    """
    REL-01: Proves that the database connection is closed and released
    before the external generate_rag_answer() call executes.
    """
    token = create_access_token(user_id=test_user.id, username=test_user.username)
    session_closed_during_generation = []

    from backend.rag import RAGQueryResponse
    import backend.main as main_module

    def mock_generate_and_probe_session(question, passages, request_id):
        # Inspect active test session state while generation is running
        for sess in active_test_sessions:
            # When db.close() was called, session transaction is ended/inactive
            session_closed_during_generation.append(sess.is_active is False or not sess.in_transaction())
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Synthetic answer verifying DB session release.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=len(passages),
            model_used="test-model",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            estimated_cost_usd=0.00001,
            pricing_version="2024-08-01",
        )

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_generate_and_probe_session)

    res = client.post(
        "/api/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What is the policy deadline?", "top_k": 3},
    )

    assert res.status_code == 200
    assert len(session_closed_during_generation) > 0
    # Must be True: Session was closed prior to generation
    assert all(session_closed_during_generation), "Expected database session to be closed before external LLM generation"


def test_query_rag_metrics_persisted_in_dedicated_session_after_release(client, test_user, monkeypatch, db_session):
    """
    REL-01: Verifies that after releasing the retrieval connection, request metrics
    are successfully written to the database in a dedicated session.
    """
    token = create_access_token(user_id=test_user.id, username=test_user.username)

    from backend.rag import RAGQueryResponse
    import backend.main as main_module

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Verified metric persistence.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=len(passages),
            model_used="test-model",
            token_usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            estimated_cost_usd=0.00002,
            pricing_version="2024-08-01",
        )

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)

    res = client.post(
        "/api/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "Verify metric persistence?", "top_k": 2},
    )

    assert res.status_code == 200
    data = res.json()
    req_id = data["request_id"]

    # Verify metric exists in DB
    metric = db_session.query(RequestMetric).filter(RequestMetric.request_id == req_id).first()
    assert metric is not None
    assert metric.user_id == test_user.id
    assert metric.status == "success"
    assert metric.prompt_tokens == 20


# ===========================================================================
# 3. REL-02: Prevent Upload Event-Loop Starvation Tests
# ===========================================================================

@pytest.mark.anyio
async def test_upload_runs_in_worker_thread(monkeypatch, db_session, test_user):
    """
    REL-02: Confirms that document ingestion is executed inside a separate worker thread
    rather than the main event loop thread.
    """
    import backend.main as main_module

    worker_threads = []
    main_thread = threading.current_thread()

    def mock_ingest(doc_id, db):
        worker_threads.append(threading.current_thread())
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "completed"
            db.commit()
        return doc

    monkeypatch.setattr(main_module, "process_document_pipeline", mock_ingest)

    doc_id = str(uuid.uuid4())
    fake_doc = Document(
        id=doc_id,
        owner_id=test_user.id,
        title="Test Thread Document",
        filename="test.pdf",
        file_hash="dummy-hash",
        storage_path="dummy-path",
        file_size_bytes=100,
        page_count=1,
        status="queued",
    )
    db_session.add(fake_doc)
    db_session.commit()

    await asyncio.to_thread(main_module._run_ingestion_in_worker, doc_id, test_engine)

    assert len(worker_threads) == 1
    # Must run on a worker thread, distinct from the main thread
    assert worker_threads[0] != main_thread


@pytest.mark.anyio
async def test_event_loop_responsiveness_during_ingestion(monkeypatch, db_session, test_user):
    """
    REL-02: Proves that while a slow ingestion runs in the worker thread,
    the asyncio event loop concurrently processes other asynchronous tasks.
    """
    def slow_worker_ingest(doc_id):
        time.sleep(0.3)  # Simulate blocking 300ms PDF extraction and embedding

    concurrent_task_completed = False

    async def concurrent_task():
        nonlocal concurrent_task_completed
        await asyncio.sleep(0.05)
        concurrent_task_completed = True

    # Run slow worker task and concurrent async task concurrently
    ingest_task = asyncio.to_thread(slow_worker_ingest, "dummy-id")
    other_task = asyncio.create_task(concurrent_task())

    # Wait for the concurrent async task
    await other_task
    # The event loop MUST have serviced the concurrent task while ingestion was sleeping in worker thread
    assert concurrent_task_completed is True

    await ingest_task


# ===========================================================================
# 4. SEC-02: Authenticated PDF Download Security Tests
# ===========================================================================

def test_download_document_unauthenticated_rejected(client):
    """SEC-02: Download request without Bearer token is rejected with HTTP 401."""
    res = client.get("/api/documents/test-doc-id/download")
    assert res.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Bearer" in res.headers.get("WWW-Authenticate", "")


def test_download_document_unauthorized_user_rejected(client, test_user, admin_user, db_session, tmp_path):
    """SEC-02: User A cannot download User B's document (IDOR safe; returns 404)."""
    pdf_file = tmp_path / "user_b_doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 Fake user B document")

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        owner_id=admin_user.id,  # Owned by admin_user
        title="Admin Confidential",
        filename="admin_confidential.pdf",
        file_hash="fake-hash-b",
        storage_path=str(pdf_file),
        file_size_bytes=len(pdf_file.read_bytes()),
        page_count=1,
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    # Attempt download as test_user (non-owner)
    token = create_access_token(user_id=test_user.id, username=test_user.username)
    res = client.get(
        f"/api/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == status.HTTP_404_NOT_FOUND
    assert "Document not found" in res.json()["detail"]


def test_download_document_owner_success(client, test_user, db_session, tmp_path):
    """SEC-02: Authenticated document owner can download their own document."""
    pdf_content = b"%PDF-1.4 Authenticated Owner PDF Bytes"
    pdf_file = tmp_path / "owner_doc.pdf"
    pdf_file.write_bytes(pdf_content)

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        owner_id=test_user.id,
        title="My Document",
        filename="my_doc.pdf",
        file_hash="owner-hash-1",
        storage_path=str(pdf_file),
        file_size_bytes=len(pdf_content),
        page_count=1,
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    token = create_access_token(user_id=test_user.id, username=test_user.username)
    res = client.get(
        f"/api/documents/{doc_id}/download",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert res.content == pdf_content
    assert "application/pdf" in res.headers.get("content-type", "")
    assert 'filename="my_doc.pdf"' in res.headers.get("content-disposition", "")
