"""
Unit and integration tests for DocuMind AI Stage 6 Observability.
Covers:
1. Unique Request ID generation.
2. Request lifecycle latency recording (retrieval, generation, total).
3. Missing token usage handling (strictly None / unknown, never zero).
4. Configurable, versioned estimated API cost calculation.
5. Provider failure capture and error categorization.
6. Sensitive data redaction (no secrets, paths, or document content).
7. Dashboard authorization (admin vs regular user).
8. Structured logging format.
"""

import pytest
import math
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from backend.models import Base, User, Document, DocumentChunk, RequestMetric
from backend.main import app, get_db, migrate_schema, bootstrap_admin_user, init_db
from backend.bootstrap import provision_admin_account
from backend.auth import hash_password, create_access_token
from backend.observability import (
    PRICING_CONFIG_VERSION,
    MODEL_PRICING_CATALOG,
    calculate_estimated_cost,
    categorize_error,
    log_request_event,
    calculate_p95_latency,
    get_monitoring_dashboard_summary,
    record_request_metric,
)
from sqlalchemy.pool import StaticPool
from backend.embeddings import get_embedding

# Isolated in-memory SQLite database for observability test suite
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
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
    """Recreates fresh database schema for each test function."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def db_session():
    """Provides a database session bound to the in-memory test database."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(db_session):
    """Creates a regular non-admin user and returns authorization headers."""
    user = User(
        email="dev@documind.ai",
        username="devuser",
        hashed_password=hash_password("DevPassword123!"),
        is_active=True,
        is_admin=False,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token(user_id=user.id, username=user.username)
    return {"Authorization": f"Bearer {token}"}, user


@pytest.fixture
def admin_headers(db_session):
    """Creates an administrator user and returns authorization headers."""
    admin = User(
        email="admin@documind.ai",
        username="admin",
        hashed_password=hash_password("AdminSecurePass123!"),
        is_active=True,
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)

    token = create_access_token(user_id=admin.id, username=admin.username)
    return {"Authorization": f"Bearer {token}"}, admin


# ---------------------------------------------------------------------------
# Test 1 & 2: Request ID Generation and Latency Lifecycle Tracking
# ---------------------------------------------------------------------------

def test_request_lifecycle_and_latency_tracking(client, auth_headers, db_session):
    """
    Verifies that every RAG query generates a unique request ID and records
    retrieval, generation, and total latency in the response and database.
    """
    headers, user = auth_headers

    # Ingest a sample document and chunk
    doc = Document(
        id="obs-doc-01",
        owner_id=user.id,
        title="Engineering Guide",
        filename="guide.pdf",
        storage_path="mock/path.pdf",
        file_size_bytes=1024,
        page_count=2,
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    chunk = DocumentChunk(
        id="obs-chunk-01",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="The production deployment freeze starts on Friday at 5 PM.",
        embedding=get_embedding("The production deployment freeze starts on Friday at 5 PM."),
    )
    db_session.add(chunk)
    db_session.commit()

    payload = {
        "question": "When does the production deployment freeze start?",
        "top_k": 3,
    }
    response = client.post("/api/rag/query", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # 1. Unique request ID exists
    assert "request_id" in data
    assert len(data["request_id"]) >= 8

    # 2. Latency metrics attached to response
    assert data["retrieval_latency_ms"] is not None
    assert data["retrieval_latency_ms"] >= 0.0
    assert data["generation_latency_ms"] is not None
    assert data["generation_latency_ms"] >= 0.0
    assert data["total_latency_ms"] is not None
    assert data["total_latency_ms"] >= data["retrieval_latency_ms"]

    # 3. Persisted into RequestMetric database row
    metric = db_session.query(RequestMetric).filter_by(request_id=data["request_id"]).first()
    assert metric is not None
    assert metric.status == "success"
    assert metric.user_id == user.id
    assert metric.retrieval_latency_ms == data["retrieval_latency_ms"]
    assert metric.total_latency_ms == data["total_latency_ms"]


# ---------------------------------------------------------------------------
# Test 3: Missing Token Usage (Stores Unknown / None, Never Zero)
# ---------------------------------------------------------------------------

def test_missing_token_usage_stores_unknown_not_zero(client, auth_headers, db_session):
    """
    Verifies that when token usage is unavailable (e.g. offline mode or missing usage),
    tokens are stored as None (unknown) and never silently replaced with zero.
    """
    headers, _ = auth_headers

    payload = {
        "question": "What is the policy on expense reimbursement?",
        "top_k": 2,
    }
    response = client.post("/api/rag/query", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # In offline mode, token_usage is None
    assert data["token_usage"] is None
    assert data["estimated_cost_usd"] is None

    # Check database record
    metric = db_session.query(RequestMetric).filter_by(request_id=data["request_id"]).first()
    assert metric is not None
    assert metric.prompt_tokens is None  # NOT 0!
    assert metric.completion_tokens is None  # NOT 0!
    assert metric.total_tokens is None  # NOT 0!
    assert metric.estimated_cost_usd is None  # NOT 0.0!


# ---------------------------------------------------------------------------
# Test 4: Estimated API Cost Calculation & Versioning
# ---------------------------------------------------------------------------

def test_estimated_cost_calculation_and_versioning():
    """
    Verifies exact hand-checkable cost calculation and pricing configuration versioning.
    Formula: prompt_tokens * (0.150 / 1M) + completion_tokens * (0.600 / 1M).
    Example: 1,000 prompt tokens + 500 completion tokens = 0.00015 + 0.00030 = 0.00045 USD.
    """
    cost, version = calculate_estimated_cost(
        model_name="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert version == PRICING_CONFIG_VERSION
    assert cost == pytest.approx(0.00045, rel=1e-5)

    # Missing tokens returns (None, version) - no fabrication!
    cost_missing, ver_missing = calculate_estimated_cost("gpt-4o-mini", None, 500)
    assert cost_missing is None
    assert ver_missing == PRICING_CONFIG_VERSION

    cost_missing_both, _ = calculate_estimated_cost("gpt-4o-mini", None, None)
    assert cost_missing_both is None

    # Embedding cost
    cost_emb, _ = calculate_estimated_cost("text-embedding-3-small", 5000, 0)
    # 5,000 * 0.020 / 1M = 0.00010 USD
    assert cost_emb == pytest.approx(0.00010, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 5 & 6: Failure Tracking and Sensitive Data Redaction
# ---------------------------------------------------------------------------

def test_provider_failure_categorization_and_redaction():
    """
    Verifies that provider exceptions are mapped to clean categories and
    user-friendly messages without exposing secrets or paths.
    """
    # 1. Timeout
    timeout_err = httpx.ReadTimeout("Connection timed out to https://api.openai.com/v1/chat")
    cat, msg = categorize_error(timeout_err)
    assert cat == "provider_timeout"
    assert "https://" not in msg
    assert "timed out" in msg

    # 2. Invalid API Key
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat")
    resp_401 = httpx.Response(401, request=mock_request)
    http_401_err = httpx.HTTPStatusError("401 Unauthorized", request=mock_request, response=resp_401)
    cat_401, msg_401 = categorize_error(http_401_err)
    assert cat_401 == "invalid_api_key"
    assert "API key" in msg_401

    # 3. Rate Limit
    resp_429 = httpx.Response(429, request=mock_request)
    http_429_err = httpx.HTTPStatusError("429 Too Many Requests", request=mock_request, response=resp_429)
    cat_429, msg_429 = categorize_error(http_429_err)
    assert cat_429 == "rate_limit_exceeded"

    # 4. Sensitive Path Redaction in HTTPException
    http_path_err = HTTPException(status_code=400, detail="Cannot find file at C:\\Users\\Secrets\\passwords.txt")
    cat_path, msg_path = categorize_error(http_path_err)
    assert "C:" not in msg_path
    assert "passwords" not in msg_path


def test_metric_persistence_excludes_sensitive_document_text(db_session):
    """
    Verifies that RequestMetric schema does not have columns for raw document text,
    query text, or prompt templates.
    """
    metric = record_request_metric(
        db=db_session,
        request_id="req-clean-01",
        user_id=1,
        document_count=2,
        retrieval_latency_ms=15.2,
        generation_latency_ms=85.0,
        total_latency_ms=100.2,
        status="success",
        prompt_tokens=120,
        completion_tokens=40,
        total_tokens=160,
        estimated_cost_usd=0.000042,
    )
    assert metric is not None

    # Verify no sensitive columns exist on RequestMetric
    cols = {c.name for c in RequestMetric.__table__.columns}
    assert "text" not in cols
    assert "prompt" not in cols
    assert "question" not in cols
    assert "document_text" not in cols
    assert "api_key" not in cols


# ---------------------------------------------------------------------------
# Test 7: Dashboard Authorization (Admin vs Regular User)
# ---------------------------------------------------------------------------

def test_dashboard_authorization(client, auth_headers, admin_headers):
    """
    Verifies that the monitoring dashboard endpoint is strictly admin-only:
    - Regular user receives 403 Forbidden.
    - Admin user receives 200 OK with monitoring data.
    """
    user_hdr, _ = auth_headers
    admin_hdr, _ = admin_headers

    # 1. Non-admin user denied
    res_user = client.get("/api/monitoring/dashboard", headers=user_hdr)
    assert res_user.status_code == 403
    assert "Admin privileges required" in res_user.json()["detail"]

    # 2. Admin user allowed
    res_admin = client.get("/api/monitoring/dashboard", headers=admin_hdr)
    assert res_admin.status_code == 200
    data = res_admin.json()
    assert "total_requests" in data
    assert "avg_total_latency_ms" in data
    assert "p95_total_latency_ms" in data
    assert "error_rate" in data
    assert "token_usage" in data
    assert "total_estimated_cost_usd" in data
    assert "pricing_version" in data


# ---------------------------------------------------------------------------
# Test 8: Monitoring Dashboard Aggregates & P95 Latency
# ---------------------------------------------------------------------------

def test_monitoring_dashboard_aggregates_and_p95(db_session):
    """
    Verifies dashboard aggregation calculations:
    - total requests, successful vs failed
    - error rate
    - average and P95 latency
    - total estimated cost
    """
    # Create 5 simulated request records
    latencies = [100.0, 150.0, 200.0, 300.0, 800.0]  # P95 should pick 800.0
    for i, lat in enumerate(latencies):
        status = "failed" if i == 4 else "success"
        err_cat = "provider_timeout" if status == "failed" else None
        record_request_metric(
            db=db_session,
            request_id=f"req-agg-{i}",
            user_id=1,
            document_count=2,
            retrieval_latency_ms=20.0,
            generation_latency_ms=lat - 20.0,
            total_latency_ms=lat,
            status=status,
            error_category=err_cat,
            prompt_tokens=100 if status == "success" else None,
            completion_tokens=50 if status == "success" else None,
            total_tokens=150 if status == "success" else None,
            estimated_cost_usd=0.000045 if status == "success" else None,
        )

    summary = get_monitoring_dashboard_summary(db_session)

    assert summary["total_requests"] == 5
    assert summary["successful_requests"] == 4
    assert summary["failed_requests"] == 1
    assert summary["error_rate"] == 0.20  # 1/5 = 20%
    assert summary["avg_total_latency_ms"] == round(sum(latencies) / 5, 2)
    assert summary["p95_total_latency_ms"] == 800.0

    # Token usage: 4 successful requests with 100 prompt tokens = 400 total
    assert summary["token_usage"]["total_prompt_tokens"] == 400
    assert summary["token_usage"]["total_completion_tokens"] == 200
    assert summary["token_usage"]["requests_with_unknown_tokens"] == 1  # The failed one
    assert summary["total_estimated_cost_usd"] == pytest.approx(4 * 0.000045, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 9: Structured Logging Output
# ---------------------------------------------------------------------------

def test_structured_logging_format(caplog):
    """
    Verifies that structured log messages emit key lifecycle fields
    without leaking private document contents.
    """
    import logging
    with caplog.at_level(logging.INFO, logger="documind.observability"):
        log_request_event(
            request_id="req-test-log-01",
            event="rag_query_completed",
            duration_ms=145.23,
            status="success",
            user_id=42,
            prompt_tokens=150,
            completion_tokens=30,
            estimated_cost_usd=0.0000405,
        )

    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "[OBSERVABILITY]" in msg
    assert "request_id=req-test-log-01" in msg
    assert "status=success" in msg
    assert "duration_ms=145.23" in msg
    assert "user_id=42" in msg
    assert "prompt_tokens=150" in msg
    assert "completion_tokens=30" in msg
    assert "estimated_cost_usd=0.0000405" in msg


# ---------------------------------------------------------------------------
# Test 10: Admin Self-Promotion Prevention & Explicit Provisioning
# ---------------------------------------------------------------------------
# Test 10: Admin Bootstrap Security & Anti-Squatting Regressions
# ---------------------------------------------------------------------------

def test_bootstrap_username_public_registration_rejected(client, monkeypatch):
    """
    Regression Test 1: Public registration of configured bootstrap username or
    reserved administrative aliases ('admin', 'administrator', 'root') is strictly
    rejected with HTTP 400.
    """
    monkeypatch.setenv("DOCUMIND_ADMIN_BOOTSTRAP_USERNAME", "custom_sysadmin")

    # 1. Reserved aliases (case- and whitespace-insensitive)
    for reserved_name in ["admin", "ADMIN", "  administrator  ", "root"]:
        res = client.post("/api/auth/register", json={
            "username": reserved_name,
            "email": f"test_{reserved_name.strip().lower()}@example.com",
            "password": "Password123!",
        })
        assert res.status_code == 400
        assert "reserved" in res.json()["detail"].lower()

    # 2. Configured environment bootstrap username
    res_custom = client.post("/api/auth/register", json={
        "username": "custom_sysadmin",
        "email": "custom@example.com",
        "password": "Password123!",
    })
    assert res_custom.status_code == 400
    assert "reserved" in res_custom.json()["detail"].lower()


def test_existing_account_with_unrelated_password_never_promoted_automatically(db_session):
    """
    Regression Test 2: An existing unprivileged account is never automatically
    promoted by bootstrap without explicit operator force promotion.
    """
    existing_user = User(
        username="legacy_operator",
        email="operator@example.com",
        hashed_password=hash_password("OldUnrelatedPass123!"),
        is_active=True,
        is_admin=False,
    )
    db_session.add(existing_user)
    db_session.commit()

    # Attempt automatic provisioning (force_promote=False)
    success, msg = provision_admin_account(
        db=db_session,
        username="legacy_operator",
        secret="BootstrapSecret999!",
        force_promote=False,
    )
    assert success is False
    assert "conflict" in msg.lower()

    # Verify user remains non-admin in database
    db_session.refresh(existing_user)
    assert existing_user.is_admin is False


def test_legitimate_bootstrap_provisioning_succeeds(client, db_session):
    """
    Regression Test 3: Legitimate explicit bootstrap provisioning creates the
    intended administrator, allowing access to /api/monitoring/dashboard.
    """
    # 1. Provision new admin
    success, msg = provision_admin_account(
        db=db_session,
        username="legit_admin",
        secret="SuperSecretAdminPass123!",
        email="legit_admin@documind.ai",
    )
    assert success is True
    assert "successfully created" in msg.lower()

    # 2. Verify admin in database
    admin_user = db_session.query(User).filter(User.username == "legit_admin").first()
    assert admin_user is not None
    assert admin_user.is_admin is True

    # 3. Log in with provisioned credentials
    login_res = client.post("/api/auth/login", json={
        "username_or_email": "legit_admin",
        "password": "SuperSecretAdminPass123!",
    })
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]

    # 4. Access dashboard -> HTTP 200 OK
    dash_res = client.get("/api/monitoring/dashboard", headers={"Authorization": f"Bearer {token}"})
    assert dash_res.status_code == 200
    assert "total_requests" in dash_res.json()


def test_normal_authenticated_user_denied_dashboard_access(client, auth_headers):
    """
    Regression Test 4: A normal authenticated user with is_admin=False receives
    HTTP 403 Forbidden from /api/monitoring/dashboard.
    """
    headers, user = auth_headers
    assert user.is_admin is False

    dash_res = client.get("/api/monitoring/dashboard", headers=headers)
    assert dash_res.status_code == 403
    assert "Admin privileges required" in dash_res.json()["detail"]


def test_missing_bootstrap_secret_refuses_elevation(db_session):
    """
    Regression Test 5: Missing or empty bootstrap secret never elevates an account.
    """
    user = User(
        username="some_user",
        email="some@example.com",
        hashed_password=hash_password("Pass123!"),
        is_active=True,
        is_admin=False,
    )
    db_session.add(user)
    db_session.commit()

    # None secret
    success_none, _ = provision_admin_account(db_session, username="some_user", secret=None)
    assert success_none is False

    # Empty string secret
    success_empty, _ = provision_admin_account(db_session, username="some_user", secret="   ")
    assert success_empty is False

    db_session.refresh(user)
    assert user.is_admin is False


def test_original_exploit_sequence_fails_to_grant_admin(client, db_session, monkeypatch):
    """
    Regression Test 6: The original exploit sequence:
    1. Register an unprivileged account before bootstrap is configured.
    2. Later configure DOCUMIND_ADMIN_BOOTSTRAP_USERNAME and DOCUMIND_ADMIN_BOOTSTRAP_SECRET.
    3. Re-initialize the application (init_db).
    4. Verify the account is NOT elevated and cannot access the admin dashboard.
    """
    # Step 1: Pre-register an unprivileged account
    reg_res = client.post("/api/auth/register", json={
        "username": "attacker_user",
        "email": "attacker@evil.com",
        "password": "AttackerChosenPass123!",
    })
    assert reg_res.status_code == 201
    attacker_token = reg_res.json()["access_token"]
    assert reg_res.json()["user"]["is_admin"] is False

    # Step 2: Operator later sets bootstrap env pointing to this username
    monkeypatch.setenv("DOCUMIND_ADMIN_BOOTSTRAP_USERNAME", "attacker_user")
    monkeypatch.setenv("DOCUMIND_ADMIN_BOOTSTRAP_SECRET", "OperatorSecretKey456!")

    # Step 3: Re-initialize the database (as happens during application reboot / lifespan)
    from backend.test_observability import test_engine
    init_db(target_engine=test_engine)

    # Step 4: Verify the account in the database was NOT promoted
    user = db_session.query(User).filter(User.username == "attacker_user").first()
    assert user is not None
    assert user.is_admin is False

    # Step 5: Attacker tries to access dashboard with their token -> HTTP 403 Forbidden!
    dash_res = client.get("/api/monitoring/dashboard", headers={"Authorization": f"Bearer {attacker_token}"})
    assert dash_res.status_code == 403
    assert "Admin privileges required" in dash_res.json()["detail"]


# ---------------------------------------------------------------------------
# Test 11: Schema Migration Compatibility for Legacy Databases
# ---------------------------------------------------------------------------

def test_schema_migration_legacy_and_current():
    """
    Issue 2: Verifies safe, idempotent schema migration:
    - Successfully upgrades a legacy table lacking is_admin column.
    - Preserves existing rows and assigns default False.
    - Running migration a second time is completely idempotent.
    """
    from sqlalchemy import text, create_engine
    from sqlalchemy.pool import StaticPool

    legacy_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Simulate legacy Stage 1-5 database: create users table WITHOUT is_admin
    with legacy_engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email VARCHAR(255) NOT NULL UNIQUE,
                username VARCHAR(50) NOT NULL UNIQUE,
                hashed_password VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            INSERT INTO users (email, username, hashed_password, is_active)
            VALUES ('legacy_user@documind.ai', 'legacyuser', 'hashed_pass_placeholder', 1);
        """))
        conn.commit()

    # Verify is_admin is not in legacy table
    with legacy_engine.connect() as conn:
        cols_before = [r[1] for r in conn.execute(text("PRAGMA table_info(users);")).fetchall()]
        assert "is_admin" not in cols_before

    # Run idempotent schema migration
    migrate_schema(legacy_engine)

    # Verify is_admin column was safely added
    with legacy_engine.connect() as conn:
        cols_after = [r[1] for r in conn.execute(text("PRAGMA table_info(users);")).fetchall()]
        assert "is_admin" in cols_after

        # Verify existing row preserved and has is_admin=0 (False)
        row = conn.execute(text("SELECT id, username, is_admin FROM users WHERE username = 'legacyuser';")).fetchone()
        assert row is not None
        assert row[1] == "legacyuser"
        assert row[2] == 0 or row[2] is False

    # Run migration again to confirm idempotency (no OperationalError)
    migrate_schema(legacy_engine)


# ---------------------------------------------------------------------------
# Test 12: Unknown vs. Measured Zero Token & Cost Aggregates
# ---------------------------------------------------------------------------

def test_unknown_and_partial_token_and_cost_aggregates(db_session):
    """
    Issue 3: Verifies that:
    1. All unknown values expose total_tokens=None and total_estimated_cost_usd=None (not 0).
    2. Measured zero values expose total_tokens=0 and total_estimated_cost_usd=0.0 (distinguishable from None).
    3. Partial mixtures mark is_partial=True and cost_is_partial=True with accurate counts.
    """
    # 1. Test All Unknown
    record_request_metric(
        db=db_session,
        request_id="req-unk-01",
        user_id=1,
        document_count=2,
        retrieval_latency_ms=10.0,
        generation_latency_ms=20.0,
        total_latency_ms=30.0,
        status="success",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        estimated_cost_usd=None,
    )
    summary_unk = get_monitoring_dashboard_summary(db_session)
    assert summary_unk["token_usage"]["total_tokens"] is None
    assert summary_unk["total_estimated_cost_usd"] is None
    assert summary_unk["token_usage"]["unknown_token_requests"] == 1
    assert summary_unk["token_usage"]["known_token_requests"] == 0
    assert summary_unk["token_usage"]["is_partial"] is False
    assert summary_unk["cost_is_partial"] is False

    # 2. Test Partial Mixture (Add one known metered request)
    record_request_metric(
        db=db_session,
        request_id="req-known-02",
        user_id=1,
        document_count=2,
        retrieval_latency_ms=10.0,
        generation_latency_ms=20.0,
        total_latency_ms=30.0,
        status="success",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.000045,
    )
    summary_partial = get_monitoring_dashboard_summary(db_session)
    assert summary_partial["token_usage"]["total_tokens"] == 150
    assert summary_partial["token_usage"]["is_partial"] is True
    assert summary_partial["token_usage"]["known_token_requests"] == 1
    assert summary_partial["token_usage"]["unknown_token_requests"] == 1
    assert summary_partial["total_estimated_cost_usd"] == 0.000045
    assert summary_partial["cost_is_partial"] is True

    # 3. Clean database and test Measured Zero
    db_session.query(RequestMetric).delete()
    db_session.commit()

    record_request_metric(
        db=db_session,
        request_id="req-zero-01",
        user_id=1,
        document_count=1,
        retrieval_latency_ms=5.0,
        generation_latency_ms=5.0,
        total_latency_ms=10.0,
        status="success",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )
    summary_zero = get_monitoring_dashboard_summary(db_session)
    assert summary_zero["token_usage"]["total_tokens"] == 0  # Measured zero, NOT None
    assert summary_zero["total_estimated_cost_usd"] == 0.0   # Measured zero, NOT None
    assert summary_zero["token_usage"]["is_partial"] is False
    assert summary_zero["cost_is_partial"] is False
    assert summary_zero["token_usage"]["known_token_requests"] == 1
    assert summary_zero["token_usage"]["unknown_token_requests"] == 0


# ---------------------------------------------------------------------------
# Test 13: Question Whitespace Validation & Rejection
# ---------------------------------------------------------------------------

def test_question_whitespace_validation(client, auth_headers, db_session):
    """
    Issue 4: Verifies that questions containing only whitespace or < 2
    characters after trimming are rejected with client error (422 or 400),
    are NOT recorded as successful RAG queries, and valid questions succeed.
    """
    auth_hdr, _ = auth_headers

    # 1. Empty string
    res_empty = client.post("/api/rag/query", json={"question": ""}, headers=auth_hdr)
    assert res_empty.status_code in (400, 422)

    # 2. Whitespace only
    res_ws = client.post("/api/rag/query", json={"question": "     "}, headers=auth_hdr)
    assert res_ws.status_code in (400, 422)

    # 3. Single character after trimming
    res_short = client.post("/api/rag/query", json={"question": "   a   "}, headers=auth_hdr)
    assert res_short.status_code in (400, 422)

    # Verify no successful metrics recorded for rejected inputs
    success_metrics = db_session.query(RequestMetric).filter(RequestMetric.status == "success").all()
    assert len(success_metrics) == 0

    # 4. Valid question succeeds (answers with safe abstention when no docs indexed, returning HTTP 200)
    res_valid = client.post("/api/rag/query", json={"question": "What is the policy?"}, headers=auth_hdr)
    assert res_valid.status_code == 200
    valid_data = res_valid.json()
    assert valid_data["insufficient_evidence"] is True
    assert "Insufficient evidence" in valid_data["answer"]

    # Verify success metric recorded for valid abstention
    success_after = db_session.query(RequestMetric).filter(RequestMetric.status == "success").all()
    assert len(success_after) == 1
    assert success_after[0].status == "success"
