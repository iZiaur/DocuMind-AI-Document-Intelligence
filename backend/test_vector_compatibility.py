import os
import uuid
from unittest.mock import MagicMock
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException

from backend.models import Base, User, Document, DocumentChunk
from backend.embeddings import (
    get_embedding_space_id,
    IncompatibleEmbeddingSpaceError,
    get_embedding,
    EMBEDDING_DIMENSION,
)
from backend.main import retrieve_authorized_passages, migrate_schema
from backend.ingestion import reembed_document


@pytest.fixture
def sqlite_db_session():
    """Provides a fresh isolated SQLite in-memory database session."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    migrate_schema(test_engine)
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = SessionMaker()
    try:
        yield session
    finally:
        session.close()


def _create_test_user_and_doc(db: Session, username: str = "vec_user"):
    user = User(
        email=f"{username}@documind.ai",
        username=username,
        hashed_password="SecurePassword123!",
    )
    db.add(user)
    db.commit()

    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user.id,
        title="Test Compatibility Manual",
        filename="test.pdf",
        storage_path="storage/documents/test.pdf",
        file_size_bytes=1024,
        page_count=2,
        status="completed",
    )
    db.add(doc)
    db.commit()
    return user, doc


# ---------------------------------------------------------------------------
# Test 1: Gemini and OpenAI Equal Dimension Have Different Identifiers
# ---------------------------------------------------------------------------

def test_gemini_and_openai_equal_dimension_different_space_ids():
    """Gemini and OpenAI 1536-dim embeddings must have distinct embedding space IDs."""
    gemini_space = get_embedding_space_id(provider="gemini", model="gemini-embedding-001", dimension=1536)
    openai_space = get_embedding_space_id(provider="openai", model="text-embedding-3-small", dimension=1536)

    assert gemini_space == "gemini:gemini-embedding-001:1536"
    assert openai_space == "openai:text-embedding-3-small:1536"
    assert gemini_space != openai_space


# ---------------------------------------------------------------------------
# Test 2: Different Models From Same Provider Have Different Identifiers
# ---------------------------------------------------------------------------

def test_different_models_same_provider_different_space_ids():
    """Different embedding models from the same provider have distinct space IDs."""
    openai_small = get_embedding_space_id(provider="openai", model="text-embedding-3-small", dimension=1536)
    openai_large = get_embedding_space_id(provider="openai", model="text-embedding-3-large", dimension=1536)

    assert openai_small != openai_large
    assert "text-embedding-3-small" in openai_small
    assert "text-embedding-3-large" in openai_large


# ---------------------------------------------------------------------------
# Test 3: Offline and Live Embeddings Cannot Be Mixed
# ---------------------------------------------------------------------------

def test_offline_and_live_space_ids_cannot_be_mixed():
    """Offline deterministic mock spaces must never match live Gemini or OpenAI spaces."""
    offline_space = get_embedding_space_id(provider="offline")
    gemini_space = get_embedding_space_id(provider="gemini")
    openai_space = get_embedding_space_id(provider="openai")

    assert offline_space == "offline:deterministic-mock:1536"
    assert offline_space != gemini_space
    assert offline_space != openai_space


# ---------------------------------------------------------------------------
# Test 4: Compatible Embeddings Can Still Be Retrieved
# ---------------------------------------------------------------------------

def test_compatible_embeddings_retrieved_successfully(sqlite_db_session, monkeypatch):
    """Chunks matching the active embedding space are retrieved normally."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    current_space = get_embedding_space_id()

    user, doc = _create_test_user_and_doc(sqlite_db_session, "compat_user")

    chunk = DocumentChunk(
        id="chunk-compat-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="All customer data must be encrypted with AES-256-GCM at rest.",
        embedding=[0.05] * 1536,
        embedding_space=current_space,
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    passages = retrieve_authorized_passages(
        query_text="What encryption standard is required for data at rest?",
        user_id=user.id,
        top_k=2,
        document_id=doc.id,
        db=sqlite_db_session,
    )

    assert len(passages) == 1
    assert passages[0].chunk_id == "chunk-compat-1"
    assert "AES-256-GCM" in passages[0].text
    assert passages[0].metadata["embedding_space"] == current_space


# ---------------------------------------------------------------------------
# Test 5: Incompatible Embeddings Are Excluded Before Vector Scoring
# ---------------------------------------------------------------------------

def test_incompatible_embeddings_excluded_cross_doc_search(sqlite_db_session, monkeypatch):
    """In cross-document search, chunks from incompatible spaces are excluded."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    user, doc = _create_test_user_and_doc(sqlite_db_session, "incompat_user")

    # Chunk with OpenAI space while active provider is offline
    chunk_openai = DocumentChunk(
        id="chunk-foreign-openai",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Critical secret password is SuperSecretPasscode123.",
        embedding=[0.05] * 1536,
        embedding_space="openai:text-embedding-3-small:1536",
    )
    sqlite_db_session.add(chunk_openai)
    sqlite_db_session.commit()

    # Cross-document search (document_id=None)
    passages = retrieve_authorized_passages(
        query_text="What is the secret password SuperSecretPasscode123?",
        user_id=user.id,
        top_k=5,
        document_id=None,
        db=sqlite_db_session,
    )

    # Incompatible chunk must NOT be returned
    assert len(passages) == 0


def test_incompatible_embeddings_rejected_on_explicit_doc_query(sqlite_db_session, monkeypatch):
    """Querying a document indexed with an incompatible space raises 409 conflict."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    user, doc = _create_test_user_and_doc(sqlite_db_session, "conflict_user")

    # Document was indexed with Gemini space
    chunk_gemini = DocumentChunk(
        id="chunk-gemini-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Gemini indexed text passage.",
        embedding=[0.02] * 1536,
        embedding_space="gemini:gemini-embedding-001:1536",
    )
    sqlite_db_session.add(chunk_gemini)
    sqlite_db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        retrieve_authorized_passages(
            query_text="Gemini indexed text",
            user_id=user.id,
            top_k=2,
            document_id=doc.id,
            db=sqlite_db_session,
        )

    assert exc_info.value.status_code == 409
    assert "incompatible with" in exc_info.value.detail.lower()
    assert "gemini:gemini-embedding-001:1536" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Test 6: Legacy Rows With Unknown Provenance Are Handled Safely
# ---------------------------------------------------------------------------

def test_legacy_rows_with_null_space_excluded_safely(sqlite_db_session, monkeypatch):
    """Legacy chunks with embedding_space=None are safely excluded from search."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    user, doc = _create_test_user_and_doc(sqlite_db_session, "legacy_user")

    # Legacy chunk with NULL embedding_space
    legacy_chunk = DocumentChunk(
        id="chunk-legacy-null",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Legacy policy written before embedding space tracking was implemented.",
        embedding=[0.01] * 1536,
        embedding_space=None,
    )
    sqlite_db_session.add(legacy_chunk)
    sqlite_db_session.commit()

    # Querying the document with legacy chunks raises 409 explaining re-indexing is needed
    with pytest.raises(HTTPException) as exc_info:
        retrieve_authorized_passages(
            query_text="Legacy policy text",
            user_id=user.id,
            top_k=2,
            document_id=doc.id,
            db=sqlite_db_session,
        )

    assert exc_info.value.status_code == 409
    assert "legacy/unknown" in exc_info.value.detail

    # Cross-document search also excludes legacy chunks safely without crashing
    results = retrieve_authorized_passages(
        query_text="Legacy policy text",
        user_id=user.id,
        top_k=2,
        document_id=None,
        db=sqlite_db_session,
    )
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Test 7: Multi-Tenant Authorization Filtering Remains Enforced
# ---------------------------------------------------------------------------

def test_authorization_filtering_remains_enforced(sqlite_db_session, monkeypatch):
    """Even if chunks are in the exact same embedding space, User B cannot access User A's chunks."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    active_space = get_embedding_space_id()

    user_a, doc_a = _create_test_user_and_doc(sqlite_db_session, "tenant_a")
    user_b, doc_b = _create_test_user_and_doc(sqlite_db_session, "tenant_b")

    chunk_a = DocumentChunk(
        id="chunk-secret-a",
        document_id=doc_a.id,
        chunk_index=0,
        page_number=1,
        text="User A confidential salary data: $250,000 base.",
        embedding=[0.08] * 1536,
        embedding_space=active_space,
    )
    sqlite_db_session.add(chunk_a)
    sqlite_db_session.commit()

    # User B queries for User A's data
    results = retrieve_authorized_passages(
        query_text="salary data $250,000 base",
        user_id=user_b.id,
        top_k=5,
        document_id=None,
        db=sqlite_db_session,
    )
    assert len(results) == 0

    # User B querying User A's document ID gets 404
    with pytest.raises(HTTPException) as exc_info:
        retrieve_authorized_passages(
            query_text="salary",
            user_id=user_b.id,
            top_k=2,
            document_id=doc_a.id,
            db=sqlite_db_session,
        )
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test 8: Hybrid Lexical Retrieval Still Works
# ---------------------------------------------------------------------------

def test_hybrid_lexical_retrieval_with_compatible_space(sqlite_db_session, monkeypatch):
    """Hybrid lexical and vector scoring works seamlessly on compatible chunks."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    active_space = get_embedding_space_id()

    user, doc = _create_test_user_and_doc(sqlite_db_session, "hybrid_user")

    chunk_1 = DocumentChunk(
        id="chunk-h1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Production hotfixes require Severity-1 approval within 15 minutes.",
        embedding=[0.03] * 1536,
        embedding_space=active_space,
    )
    chunk_2 = DocumentChunk(
        id="chunk-h2",
        document_id=doc.id,
        chunk_index=1,
        page_number=2,
        text="Standard non-emergency tickets are processed within five business days.",
        embedding=[0.03] * 1536,
        embedding_space=active_space,
    )
    sqlite_db_session.add_all([chunk_1, chunk_2])
    sqlite_db_session.commit()

    results = retrieve_authorized_passages(
        query_text="Severity-1 15 minutes",
        user_id=user.id,
        top_k=1,
        document_id=doc.id,
        db=sqlite_db_session,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-h1"
    assert results[0].score > 0.15


# ---------------------------------------------------------------------------
# Test 9: Migration Idempotency and Repeatability
# ---------------------------------------------------------------------------

def test_migration_repeatability_sqlite(sqlite_db_session):
    """Running migrate_schema multiple times is safe, idempotent, and non-destructive."""
    test_engine = sqlite_db_session.bind
    # First run was in fixture; call it again
    migrate_schema(test_engine)
    migrate_schema(test_engine)

    # Verify column exists and is functional
    with test_engine.connect() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(document_chunks);")).fetchall()]
        assert "embedding_space" in cols


# ---------------------------------------------------------------------------
# Test 10: Safe Transactional Re-Embedding
# ---------------------------------------------------------------------------

def test_safe_transactional_reembed(sqlite_db_session, monkeypatch):
    """reembed_document updates embedding_space and embeddings atomically without partial failure."""
    user, doc = _create_test_user_and_doc(sqlite_db_session, "reembed_user")

    # Initial legacy chunk
    chunk = DocumentChunk(
        id="chunk-orig-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Engineering design doc passage.",
        embedding=[0.01] * 1536,
        embedding_space="legacy/unknown",
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    # Re-embed to offline target space
    reembed_document(
        document_id=doc.id,
        db=sqlite_db_session,
        target_space="offline:deterministic-mock:1536",
    )

    sqlite_db_session.refresh(chunk)
    assert chunk.embedding_space == "offline:deterministic-mock:1536"
    assert chunk.page_number == 1
    assert chunk.id == "chunk-orig-1"


def test_reembed_rollback_on_failure(sqlite_db_session, monkeypatch):
    """If embedding generation fails during re-embedding, database state is unchanged."""
    user, doc = _create_test_user_and_doc(sqlite_db_session, "rollback_user")

    chunk = DocumentChunk(
        id="chunk-rb-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Rollback test passage.",
        embedding=[0.02] * 1536,
        embedding_space="legacy:space",
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    # Simulate provider failure during batch embedding
    def mock_failing_batch(*args, **kwargs):
        raise RuntimeError("Simulated provider outage during re-embedding")

    monkeypatch.setattr("backend.ingestion.get_embeddings_batch", mock_failing_batch)

    with pytest.raises(RuntimeError, match="Simulated provider outage"):
        reembed_document(
            document_id=doc.id,
            db=sqlite_db_session,
            target_space="gemini:gemini-embedding-001:1536",
        )

    # Verify chunk was NOT modified
    sqlite_db_session.refresh(chunk)
    assert chunk.embedding_space == "legacy:space"
