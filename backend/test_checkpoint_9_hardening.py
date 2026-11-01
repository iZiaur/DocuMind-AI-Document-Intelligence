"""
Checkpoint 9 Security Hardening Test Suite.

Validates the three medium-severity security findings:
1. SEC-03 — Memory-safe file upload streaming, 10MB limit enforcement, structural PDF validation,
            atomic rename, and temporary file cleanup.
2. SEC-04 — In-memory sliding-window rate limiting on /api/auth/login and /api/rag/query,
            anti-spoofing client IP resolution, authenticated user quota isolation, and Retry-After headers.
3. SEC-05 — Accidental offline/mock provider fallback prevention in production, fail-fast startup checks,
            and hermetic test runner guard preservation.
"""

import os
import io
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status, HTTPException, Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app, get_db, validate_provider_configuration
from backend.models import Base, User, Document, DocumentChunk
from backend.auth import create_access_token, hash_password
from backend.storage import (
    stream_and_validate_upload,
    save_pdf_securely,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    STREAM_CHUNK_SIZE,
)
from backend.rate_limiter import (
    SlidingWindowRateLimiter,
    resolve_client_ip,
    check_login_rate_limit,
    check_rag_rate_limit,
    rate_limiter,
)
from backend.embeddings import (
    get_embedding_provider,
    validate_embedding_provider_configuration,
    is_production_mode as is_emb_production_mode,
)
from backend.rag import (
    get_generation_provider,
    validate_generation_provider_configuration,
    is_production_mode as is_rag_production_mode,
)

# In-memory SQLite database for isolated test execution
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_database():
    """Recreates fresh database schema and resets rate limiter for each test."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    rate_limiter.reset()
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()
    rate_limiter.reset()


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
    u = db_session.query(User).filter(User.username == "c9_user1").first()
    if not u:
        u = User(
            email="c9_user1@example.com",
            username="c9_user1",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            is_admin=False,
        )
        db_session.add(u)
        db_session.commit()
        db_session.refresh(u)
    return u


@pytest.fixture
def test_user_2(db_session):
    u = db_session.query(User).filter(User.username == "c9_user2").first()
    if not u:
        u = User(
            email="c9_user2@example.com",
            username="c9_user2",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            is_admin=False,
        )
        db_session.add(u)
        db_session.commit()
        db_session.refresh(u)
    return u


def _create_minimal_pdf_bytes(text: str = "DocuMind Test Document Content") -> bytes:
    """Constructs a strictly valid, minimal PDF file in memory."""
    content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
    stream_len = len(content.encode("latin-1"))
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(stream_len).encode("latin-1") + b" >>\nstream\n"
        + content.encode("latin-1") + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000340 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n424\n%%EOF\n"
    )
    return pdf


# ===========================================================================
# 1. SEC-03: Memory-Safe Upload Streaming & Size Enforcement Tests
# ===========================================================================

def test_upload_valid_pdf_within_limit_succeeds(client, test_user):
    """Verifies that a valid PDF under 10MB streams, validates, and uploads successfully."""
    token = create_access_token(user_id=test_user.id, username=test_user.username)
    pdf_data = _create_minimal_pdf_bytes("DocuMind Ingestion Test")

    files = {"file": ("valid_doc.pdf", pdf_data, "application/pdf")}
    data = {"title": "Valid Streamed Document"}
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/api/documents/upload", files=files, data=data, headers=headers)
    assert response.status_code == status.HTTP_201_CREATED
    res_data = response.json()
    assert res_data["title"] == "Valid Streamed Document"
    assert res_data["filename"] == "valid_doc.pdf"
    assert res_data["file_size_bytes"] == len(pdf_data)


def test_upload_exceeding_content_length_rejected_before_streaming(client, test_user):
    """Verifies that uploads advertising Content-Length > 10MB are rejected with 413 immediately."""
    token = create_access_token(user_id=test_user.id, username=test_user.username)
    pdf_data = _create_minimal_pdf_bytes("Small Content")

    # Claim an oversized Content-Length that exceeds configured max limit
    oversized_length = (MAX_UPLOAD_SIZE_MB + 5) * 1024 * 1024
    files = {"file": ("oversized.pdf", pdf_data, "application/pdf")}
    data = {"title": "Oversized Document"}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Length": str(oversized_length),
    }

    response = client.post("/api/documents/upload", files=files, data=data, headers=headers)
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


def test_upload_streaming_detects_overflow_and_cleans_up(tmp_path):
    """
    Verifies that when streaming a file that exceeds the maximum limit mid-stream (e.g. lied about Content-Length),
    it raises HTTP 413 and guarantees the temporary file is deleted.
    """
    import asyncio

    async def _run():
        chunk_part1 = b"%PDF-1.4\n" + (b"X" * (MAX_UPLOAD_SIZE_BYTES // 2))
        chunk_overflow = b"Y" * ((MAX_UPLOAD_SIZE_BYTES // 2) + (2 * 1024 * 1024))  # Exceeds limit

        mock_upload = MagicMock()
        mock_upload.filename = "large_stream.pdf"
        mock_upload.headers = {}  # No Content-Length header

        read_calls = [chunk_part1, chunk_overflow, b""]
        async def mock_read(_size=STREAM_CHUNK_SIZE):
            if read_calls:
                return read_calls.pop(0)
            return b""

        mock_upload.read = mock_read

        target_dir = str(tmp_path / "storage")

        with pytest.raises(HTTPException) as exc_info:
            await stream_and_validate_upload(mock_upload, target_dir=target_dir)

        assert exc_info.value.status_code == 413
        assert "exceeds" in exc_info.value.detail.lower()

        # Verify target directory has zero leftover .tmp files
        leftover_files = [f for f in os.listdir(target_dir) if f.endswith(".tmp")]
        assert len(leftover_files) == 0

    asyncio.run(_run())


def test_upload_non_pdf_magic_bytes_rejected(client, test_user):
    """Verifies that a file without %PDF- magic bytes is rejected with 400 Bad Request."""
    token = create_access_token(user_id=test_user.id, username=test_user.username)
    invalid_data = b"This is a plain text file pretending to be a PDF."

    files = {"file": ("fake.pdf", invalid_data, "application/pdf")}
    data = {"title": "Fake PDF"}
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/api/documents/upload", files=files, data=data, headers=headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "File header does not match PDF signature" in response.json()["detail"]


def test_upload_corrupted_pdf_structure_rejected(client, test_user):
    """Verifies that a file starting with %PDF- but having corrupted body is rejected."""
    token = create_access_token(user_id=test_user.id, username=test_user.username)
    corrupted_data = b"%PDF-1.4\nSome corrupted binary data that cannot be parsed by pypdf %%EOF"

    files = {"file": ("corrupted.pdf", corrupted_data, "application/pdf")}
    data = {"title": "Corrupted PDF"}
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/api/documents/upload", files=files, data=data, headers=headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Malformed or corrupted PDF structure" in response.json()["detail"]


# ===========================================================================
# 2. SEC-04: Rate Limiting & Anti-Spoofing Tests
# ===========================================================================

def test_login_rate_limiting_enforcement(client, test_user, monkeypatch):
    """
    Verifies IP-based rate limiting on /api/auth/login.
    Allows up to limit, then rejects subsequent requests with HTTP 429 and Retry-After header.
    """
    monkeypatch.setenv("RATE_LIMIT_LOGIN_PER_MINUTE", "3")

    payload = {
        "username_or_email": test_user.username,
        "password": "Password123!",
    }

    # First 3 attempts succeed
    for i in range(3):
        res = client.post("/api/auth/login", json=payload)
        assert res.status_code == status.HTTP_200_OK, f"Attempt {i+1} failed"

    # 4th attempt should be blocked by rate limiter
    blocked_res = client.post("/api/auth/login", json=payload)
    assert blocked_res.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Retry-After" in blocked_res.headers
    retry_after = int(blocked_res.headers["Retry-After"])
    assert 1 <= retry_after <= 60
    assert "Rate limit exceeded" in blocked_res.json()["detail"]


def test_login_rate_limiting_anti_spoofing(client, test_user, monkeypatch):
    """
    Verifies anti-spoofing defense for client IP resolution:
    Untrusted direct clients cannot bypass rate limiting by rotating X-Forwarded-For headers.
    """
    monkeypatch.setenv("RATE_LIMIT_LOGIN_PER_MINUTE", "2")
    # Untrusted proxy host: 198.51.100.5
    monkeypatch.setenv("TRUSTED_PROXIES", "10.0.0.1")  # testclient does not match 10.0.0.1

    payload = {
        "username_or_email": test_user.username,
        "password": "Password123!",
    }

    # Attempt 1 with spoofed header
    res1 = client.post(
        "/api/auth/login",
        json=payload,
        headers={"X-Forwarded-For": "1.1.1.1"},
    )
    assert res1.status_code == status.HTTP_200_OK

    # Attempt 2 with another spoofed header
    res2 = client.post(
        "/api/auth/login",
        json=payload,
        headers={"X-Forwarded-For": "2.2.2.2"},
    )
    assert res2.status_code == status.HTTP_200_OK

    # Attempt 3 with a third spoofed header: Should be BLOCKED because direct peer IP is identical!
    res3 = client.post(
        "/api/auth/login",
        json=payload,
        headers={"X-Forwarded-For": "3.3.3.3"},
    )
    assert res3.status_code == status.HTTP_429_TOO_MANY_REQUESTS


def test_resolve_client_ip_trusted_proxy_handling():
    """Unit test for resolve_client_ip confirming trusted vs untrusted proxy rules."""
    # Untrusted request
    mock_req_untrusted = MagicMock(spec=Request)
    mock_req_untrusted.client.host = "203.0.113.10"
    mock_req_untrusted.headers = {"X-Forwarded-For": "192.0.2.1"}

    resolved = resolve_client_ip(mock_req_untrusted)
    assert resolved == "203.0.113.10"  # Ignores X-Forwarded-For from untrusted host

    # Trusted request (127.0.0.1 is in default TRUSTED_PROXIES)
    mock_req_trusted = MagicMock(spec=Request)
    mock_req_trusted.client.host = "127.0.0.1"
    mock_req_trusted.headers = {"X-Forwarded-For": "198.51.100.42, 10.0.0.1"}

    resolved_trusted = resolve_client_ip(mock_req_trusted)
    assert resolved_trusted == "198.51.100.42"  # Takes first IP from trusted proxy


def test_rag_rate_limiting_user_quota_isolation(client, test_user, test_user_2, monkeypatch):
    """
    Verifies user-based quota isolation on /api/rag/query:
    When User 1 exhausts their rate limit quota, User 2 remains unaffected.
    """
    monkeypatch.setenv("RATE_LIMIT_RAG_PER_MINUTE", "2")

    token1 = create_access_token(user_id=test_user.id, username=test_user.username)
    token2 = create_access_token(user_id=test_user_2.id, username=test_user_2.username)

    rag_payload = {"question": "What is the security architecture?"}

    # User 1 makes 2 requests
    res1 = client.post("/api/rag/query", json=rag_payload, headers={"Authorization": f"Bearer {token1}"})
    assert res1.status_code == status.HTTP_200_OK

    res2 = client.post("/api/rag/query", json=rag_payload, headers={"Authorization": f"Bearer {token1}"})
    assert res2.status_code == status.HTTP_200_OK

    # User 1 3rd request is blocked with 429
    res3 = client.post("/api/rag/query", json=rag_payload, headers={"Authorization": f"Bearer {token1}"})
    assert res3.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Retry-After" in res3.headers

    # User 2 makes a request: Should SUCCEED because quotas are isolated by user_id
    res_user2 = client.post("/api/rag/query", json=rag_payload, headers={"Authorization": f"Bearer {token2}"})
    assert res_user2.status_code == status.HTTP_200_OK


def test_sliding_window_rate_limiter_memory_bounding():
    """Verifies that the rate limiter evicts expired keys and respects max_keys bound."""
    limiter = SlidingWindowRateLimiter(max_keys=5)

    # Insert 5 keys
    for i in range(5):
        limiter.check_rate_limit(f"key_{i}", limit=10, window_seconds=60.0)

    assert len(limiter._cache) == 5

    # Insert a 6th key; should prune or evict oldest to enforce max_keys
    limiter.check_rate_limit("key_5", limit=10, window_seconds=60.0)
    assert len(limiter._cache) <= 5


# ===========================================================================
# 3. SEC-05: Production Provider Enforcement & Anti-Fallback Tests
# ===========================================================================

def test_production_mode_detection(monkeypatch):
    """Verifies is_production_mode accurately distinguishes production vs development."""
    # Under pytest with DOCUMIND_ENV=production
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    assert is_emb_production_mode() is True
    assert is_rag_production_mode() is True

    # Under pytest with DOCUMIND_ENV=development
    monkeypatch.setenv("DOCUMIND_ENV", "development")
    assert is_emb_production_mode() is False
    assert is_rag_production_mode() is False


def test_production_missing_embedding_provider_raises_error(monkeypatch):
    """Verifies that running in production without explicit EMBEDDING_PROVIDER fails fast."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_embedding_provider()

    assert "EMBEDDING_PROVIDER must be explicitly configured in production" in str(exc_info.value)


def test_production_offline_embedding_provider_prohibited(monkeypatch):
    """Verifies that setting EMBEDDING_PROVIDER='offline' in production raises RuntimeError."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    with pytest.raises(RuntimeError) as exc_info:
        get_embedding_provider()

    assert "EMBEDDING_PROVIDER cannot be set to 'offline' in production mode" in str(exc_info.value)


def test_production_embedding_provider_missing_api_key_raises_error(monkeypatch):
    """Verifies that selecting 'gemini' or 'openai' in production without an API key fails fast."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")

    # Gemini missing key
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError) as exc_info:
        get_embedding_provider()
    assert "GEMINI_API_KEY is missing or empty" in str(exc_info.value)

    # OpenAI missing key
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError) as exc_info:
        get_embedding_provider()
    assert "OPENAI_API_KEY is missing or empty" in str(exc_info.value)


def test_production_missing_generation_provider_raises_error(monkeypatch):
    """Verifies that running in production without explicit GENERATION_PROVIDER fails fast."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_generation_provider()

    assert "GENERATION_PROVIDER must be explicitly configured in production" in str(exc_info.value)


def test_production_offline_generation_provider_prohibited(monkeypatch):
    """Verifies that setting GENERATION_PROVIDER='offline' in production raises RuntimeError."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")

    with pytest.raises(RuntimeError) as exc_info:
        get_generation_provider()

    assert "GENERATION_PROVIDER cannot be set to 'offline' in production mode" in str(exc_info.value)


def test_production_generation_provider_missing_api_key_raises_error(monkeypatch):
    """Verifies that selecting 'gemini' or 'openai' in production without an API key fails fast."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")

    # Gemini missing key
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError) as exc_info:
        get_generation_provider()
    assert "GEMINI_API_KEY is missing or empty" in str(exc_info.value)

    # OpenAI missing key
    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError) as exc_info:
        get_generation_provider()
    assert "OPENAI_API_KEY is missing or empty" in str(exc_info.value)


def test_production_valid_configuration_succeeds(monkeypatch):
    """Verifies that complete, valid configurations pass startup validation in production."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-production-gemini-key")
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")

    assert get_embedding_provider() == "gemini"
    assert get_generation_provider() == "gemini"

    # Composite startup validation should run without raising any error
    validate_provider_configuration()


def test_development_mode_permits_explicit_offline_providers(monkeypatch):
    """Verifies that development environments are permitted to run with offline mock providers."""
    monkeypatch.setenv("DOCUMIND_ENV", "development")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")

    assert get_embedding_provider() == "offline"
    assert get_generation_provider() == "offline"

    # Startup validation should run without error in development
    validate_provider_configuration()


def test_pytest_runner_hermetic_offline_guard_preserved(monkeypatch):
    """
    Verifies that during automated test runs without explicit DOCUMIND_ENV=production,
    providers safely default to 'offline' if unset, preventing accidental live external API calls.
    """
    monkeypatch.delenv("DOCUMIND_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # PYTEST_CURRENT_TEST is naturally present in pytest runs
    assert get_embedding_provider() == "offline"
    assert get_generation_provider() == "offline"
