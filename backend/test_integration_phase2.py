import os
import uuid
import tempfile
from unittest.mock import MagicMock
import pytest
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException

from backend.models import Base, User, Document, DocumentChunk
from backend.embeddings import (
    get_embedding_space_id,
    get_embedding,
    get_embeddings_batch,
    EmbeddingProviderError,
    EMBEDDING_DIMENSION,
)
from backend.main import retrieve_authorized_passages, migrate_schema
from backend.ingestion import process_document_pipeline, reembed_document
from backend.rag import generate_rag_answer
from backend.test_main import create_pdf_with_text


@pytest.fixture
def sqlite_db_session():
    """Provides a fresh isolated in-memory SQLite database session with migrated schema."""
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


def _create_user(db: Session, username: str = "int_user") -> User:
    user = User(
        email=f"{username}@documind.ai",
        username=username,
        hashed_password="TestPassword123!",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Test 1: Complete Offline Ingestion -> Storage -> Retrieval -> RAG Response
# ---------------------------------------------------------------------------

def test_end_to_end_offline_pipeline(sqlite_db_session, monkeypatch):
    """Complete end-to-end ingestion, storage, retrieval, and RAG response in offline mode."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")

    user = _create_user(sqlite_db_session, "e2e_offline_user")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "incident_response.pdf")
        pdf_bytes = create_pdf_with_text(
            [
                "Severity-1 production incidents require executive notification within 15 minutes.",
                "Severity-2 incidents must be acknowledged by engineering within 30 minutes.",
            ]
        )
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        doc = Document(
            id=str(uuid.uuid4()),
            owner_id=user.id,
            title="Incident Response Guide",
            filename="incident_response.pdf",
            storage_path=pdf_path,
            file_size_bytes=os.path.getsize(pdf_path),
            page_count=2,
            status="queued",
        )
        sqlite_db_session.add(doc)
        sqlite_db_session.commit()

        # Ingest document
        processed_doc = process_document_pipeline(doc.id, sqlite_db_session)
        assert processed_doc.status == "completed"

        # Verify chunk provenance
        chunks = sqlite_db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
        assert len(chunks) == 2
        for c in chunks:
            assert c.embedding_space == "offline:deterministic-mock:1536"
            assert len(c.embedding) == 1536

        # Retrieve passages
        passages = retrieve_authorized_passages(
            query_text="What is the executive notification deadline for Severity-1 incidents?",
            user_id=user.id,
            top_k=2,
            document_id=doc.id,
            db=sqlite_db_session,
        )
        assert len(passages) >= 1
        assert "15 minutes" in passages[0].text
        assert passages[0].metadata["embedding_space"] == "offline:deterministic-mock:1536"

        # Generate RAG answer
        response = generate_rag_answer(
            question="What is the executive notification deadline for Severity-1 incidents?",
            passages=passages,
        )
        assert response.insufficient_evidence is False
        assert "15 minutes" in response.answer
        assert len(response.citations) >= 1
        assert "(local-grounded)" in response.model_used


# ---------------------------------------------------------------------------
# Test 2: Gemini Mocked Pipeline (Embedding + Generation)
# ---------------------------------------------------------------------------

def test_end_to_end_gemini_mocked_pipeline(sqlite_db_session, monkeypatch):
    """Verifies complete pipeline using mocked Gemini embedding and Gemini generation."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

    active_space = "gemini:gemini-embedding-001:1536"
    dummy_vec = [0.03] * 1536

    def mock_post(*args, **kwargs):
        url = args[1] if len(args) > 1 else kwargs.get("url", "")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200

        if "batchEmbedContents" in url:
            json_data = kwargs.get("json", {})
            req_count = len(json_data.get("requests", []))
            mock_resp.json.return_value = {
                "embeddings": [{"values": dummy_vec} for _ in range(req_count)]
            }
        elif "embedContent" in url:
            mock_resp.json.return_value = {"embedding": {"values": dummy_vec}}
        elif "generateContent" in url:
            mock_resp.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Production deployments require two peer reviews [source: chunk-g1]."
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 95,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 115,
                },
            }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    user = _create_user(sqlite_db_session, "gemini_pipe_user")
    doc = Document(
        id="doc-gemini-pipe",
        owner_id=user.id,
        title="Engineering Standards",
        filename="eng.pdf",
        storage_path="eng.pdf",
        file_size_bytes=500,
        page_count=1,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="chunk-g1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="All production deployments require two peer code reviews before merge.",
        embedding=dummy_vec,
        embedding_space=active_space,
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    passages = retrieve_authorized_passages(
        query_text="What are the peer review requirements for pull requests?",
        user_id=user.id,
        top_k=1,
        document_id=doc.id,
        db=sqlite_db_session,
    )
    assert len(passages) == 1
    assert passages[0].metadata["embedding_space"] == active_space

    rag_res = generate_rag_answer(
        question="What are the peer review requirements for pull requests?",
        passages=passages,
    )
    assert rag_res.insufficient_evidence is False
    assert "two peer reviews" in rag_res.answer
    assert rag_res.model_used == "gemini-2.5-flash"
    assert len(rag_res.citations) == 1
    assert rag_res.token_usage["total_tokens"] == 115
    assert rag_res.estimated_cost_usd is not None


# ---------------------------------------------------------------------------
# Test 3: OpenAI Mocked Pipeline (Embedding + Generation)
# ---------------------------------------------------------------------------

def test_end_to_end_openai_mocked_pipeline(sqlite_db_session, monkeypatch):
    """Verifies complete pipeline using mocked OpenAI embedding and chat completions."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("backend.rag.OPENAI_API_KEY", "test-openai-key")

    active_space = "openai:text-embedding-3-small:1536"
    dummy_vec = [0.04] * 1536

    def mock_post(*args, **kwargs):
        url = args[1] if len(args) > 1 else kwargs.get("url", "")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200

        if "embeddings" in url:
            json_data = kwargs.get("json", {})
            inputs = json_data.get("input", [])
            if isinstance(inputs, list):
                mock_resp.json.return_value = {
                    "data": [{"index": i, "embedding": dummy_vec} for i in range(len(inputs))]
                }
            else:
                mock_resp.json.return_value = {"data": [{"embedding": dummy_vec}]}
        elif "chat/completions" in url:
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "Per diem meal allowance is capped at $75 per day [source: chunk-o1]."
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 22,
                    "total_tokens": 132,
                },
            }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    user = _create_user(sqlite_db_session, "openai_pipe_user")
    doc = Document(
        id="doc-openai-pipe",
        owner_id=user.id,
        title="Travel Handbook",
        filename="travel.pdf",
        storage_path="travel.pdf",
        file_size_bytes=500,
        page_count=1,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="chunk-o1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="The domestic per diem meal allowance is capped at $75 per day under SOC 2 compliance.",
        embedding=dummy_vec,
        embedding_space=active_space,
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    passages = retrieve_authorized_passages(
        query_text="What is the meal allowance per diem cap?",
        user_id=user.id,
        top_k=1,
        document_id=doc.id,
        db=sqlite_db_session,
    )
    assert len(passages) == 1
    assert passages[0].metadata["embedding_space"] == active_space

    rag_res = generate_rag_answer(
        question="What is the meal allowance per diem cap?",
        passages=passages,
    )
    assert rag_res.insufficient_evidence is False
    assert "$75 per day" in rag_res.answer
    assert len(rag_res.citations) == 1


# ---------------------------------------------------------------------------
# Test 4: Mixed Embedding Space Document Behavior
# ---------------------------------------------------------------------------

def test_mixed_embedding_space_document_rejected(sqlite_db_session, monkeypatch):
    """Document containing chunks from multiple distinct embedding spaces raises 409."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    user = _create_user(sqlite_db_session, "mixed_doc_user")
    doc = Document(
        id="doc-mixed-spaces",
        owner_id=user.id,
        title="Mixed Document",
        filename="mixed.pdf",
        storage_path="p",
        file_size_bytes=100,
        page_count=2,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    c1 = DocumentChunk(
        id="chunk-m1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Chunk embedded with Gemini.",
        embedding=[0.05] * 1536,
        embedding_space="gemini:gemini-embedding-001:1536",
    )
    c2 = DocumentChunk(
        id="chunk-m2",
        document_id=doc.id,
        chunk_index=1,
        page_number=2,
        text="Chunk embedded with OpenAI.",
        embedding=[0.05] * 1536,
        embedding_space="openai:text-embedding-3-small:1536",
    )
    sqlite_db_session.add_all([c1, c2])
    sqlite_db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        retrieve_authorized_passages(
            query_text="Find text",
            user_id=user.id,
            top_k=2,
            document_id=doc.id,
            db=sqlite_db_session,
        )

    assert exc_info.value.status_code == 409
    assert "mixed embedding spaces" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# Test 5: Unknown Legacy Provenance
# ---------------------------------------------------------------------------

def test_unknown_legacy_provenance_rejected(sqlite_db_session, monkeypatch):
    """Querying a legacy document with NULL embedding space raises 409 conflict."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    user = _create_user(sqlite_db_session, "legacy_user_int")

    doc = Document(
        id="doc-legacy-int",
        owner_id=user.id,
        title="Legacy Doc",
        filename="l.pdf",
        storage_path="p",
        file_size_bytes=100,
        page_count=1,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="chunk-legacy-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Legacy text with no provenance tag.",
        embedding=[0.01] * 1536,
        embedding_space=None,
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        retrieve_authorized_passages(
            query_text="Legacy text",
            user_id=user.id,
            top_k=2,
            document_id=doc.id,
            db=sqlite_db_session,
        )

    assert exc_info.value.status_code == 409
    assert "legacy/unknown" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Test 6: Re-Embedding a Legacy Document Makes it Queryable
# ---------------------------------------------------------------------------

def test_reembedding_legacy_document_makes_it_queryable(sqlite_db_session, monkeypatch):
    """Re-embedding a legacy document into the active space allows successful retrieval."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    active_space = "offline:deterministic-mock:1536"

    user = _create_user(sqlite_db_session, "reembed_success_user")
    doc = Document(
        id="doc-reembed-test",
        owner_id=user.id,
        title="Policy Manual",
        filename="p.pdf",
        storage_path="p.pdf",
        file_size_bytes=100,
        page_count=1,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="chunk-reembed-target",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Mandatory multi-factor authentication (MFA) policy.",
        embedding=[0.0] * 1536,
        embedding_space=None,
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    # Pre-reembed: querying document raises 409
    with pytest.raises(HTTPException) as pre_exc:
        retrieve_authorized_passages("MFA policy", user.id, 1, doc.id, sqlite_db_session)
    assert pre_exc.value.status_code == 409

    # Re-embed document into active offline space
    reembed_document(doc.id, sqlite_db_session, target_space=active_space)

    # Post-reembed: querying succeeds and returns passage
    passages = retrieve_authorized_passages("MFA policy", user.id, 1, doc.id, sqlite_db_session)
    assert len(passages) == 1
    assert passages[0].chunk_id == "chunk-reembed-target"
    assert passages[0].metadata["embedding_space"] == active_space


# ---------------------------------------------------------------------------
# Test 7: Re-Embedding Rollback When Second Batch Fails
# ---------------------------------------------------------------------------

def test_reembed_rollback_when_batch_fails(sqlite_db_session, monkeypatch):
    """When batch embedding generation raises an error, all chunk vectors and space tags roll back."""
    user = _create_user(sqlite_db_session, "batch_fail_user")
    doc = Document(
        id="doc-fail-batch",
        owner_id=user.id,
        title="Multi-page Doc",
        filename="m.pdf",
        storage_path="m.pdf",
        file_size_bytes=200,
        page_count=2,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    c1 = DocumentChunk(
        id="c-orig-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Original Page 1 content.",
        embedding=[0.11] * 1536,
        embedding_space="legacy:space",
    )
    c2 = DocumentChunk(
        id="c-orig-2",
        document_id=doc.id,
        chunk_index=1,
        page_number=2,
        text="Original Page 2 content.",
        embedding=[0.22] * 1536,
        embedding_space="legacy:space",
    )
    sqlite_db_session.add_all([c1, c2])
    sqlite_db_session.commit()

    def mock_failing_batch(*args, **kwargs):
        raise EmbeddingProviderError("Sub-batch 2 network timeout from provider.")

    monkeypatch.setattr("backend.ingestion.get_embeddings_batch", mock_failing_batch)

    with pytest.raises(EmbeddingProviderError, match="Sub-batch 2 network timeout"):
        reembed_document(doc.id, sqlite_db_session, target_space="offline:deterministic-mock:1536")

    sqlite_db_session.refresh(c1)
    sqlite_db_session.refresh(c2)
    assert c1.embedding_space == "legacy:space"
    assert c2.embedding_space == "legacy:space"
    assert c1.embedding[0] == pytest.approx(0.11)
    assert c2.embedding[0] == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# Test 8: Re-Embedding Preserves Chunk IDs, Page Numbers, and Ordering
# ---------------------------------------------------------------------------

def test_reembed_preserves_metadata_and_ordering(sqlite_db_session, monkeypatch):
    """Re-embedding maintains chunk primary keys, page numbers, and exact chunk indices."""
    user = _create_user(sqlite_db_session, "preserve_meta_user")
    doc = Document(
        id="doc-preserve-test",
        owner_id=user.id,
        title="Architecture Specs",
        filename="a.pdf",
        storage_path="a.pdf",
        file_size_bytes=300,
        page_count=3,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    c_ids = [f"fixed-chunk-id-{i}" for i in range(3)]
    for i in range(3):
        sqlite_db_session.add(
            DocumentChunk(
                id=c_ids[i],
                document_id=doc.id,
                chunk_index=i,
                page_number=i + 1,
                text=f"Text content for section {i+1}.",
                embedding=[0.05] * 1536,
                embedding_space="old:space",
            )
        )
    sqlite_db_session.commit()

    reembed_document(doc.id, sqlite_db_session, target_space="offline:deterministic-mock:1536")

    post_chunks = (
        sqlite_db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    assert len(post_chunks) == 3
    for i, c in enumerate(post_chunks):
        assert c.id == c_ids[i]
        assert c.chunk_index == i
        assert c.page_number == i + 1
        assert c.embedding_space == "offline:deterministic-mock:1536"


# ---------------------------------------------------------------------------
# Test 9: Repeated Re-Embedding Safety
# ---------------------------------------------------------------------------

def test_repeated_reembedding_idempotence(sqlite_db_session):
    """Running reembed_document repeatedly is safe and idempotent."""
    user = _create_user(sqlite_db_session, "repeat_user")
    doc = Document(
        id="doc-repeat-test",
        owner_id=user.id,
        title="Repeat Doc",
        filename="r.pdf",
        storage_path="r.pdf",
        file_size_bytes=100,
        page_count=1,
        status="completed",
    )
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="chunk-rep-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Content for repeat testing.",
        embedding=[0.01] * 1536,
        embedding_space="offline:deterministic-mock:1536",
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    for _ in range(3):
        reembed_document(doc.id, sqlite_db_session, target_space="offline:deterministic-mock:1536")

    sqlite_db_session.refresh(chunk)
    assert chunk.embedding_space == "offline:deterministic-mock:1536"
    assert chunk.chunk_index == 0


# ---------------------------------------------------------------------------
# Test 10: Multi-User Authorization Under Same Embedding Space
# ---------------------------------------------------------------------------

def test_multi_user_authorization_under_same_space(sqlite_db_session, monkeypatch):
    """Even when two users have chunks in the same space, cross-tenant isolation is strict."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    space = "offline:deterministic-mock:1536"

    user_a = _create_user(sqlite_db_session, "alpha_user")
    user_b = _create_user(sqlite_db_session, "beta_user")

    doc_a = Document(id="doc-a", owner_id=user_a.id, title="Doc A", filename="a.pdf", storage_path="p", file_size_bytes=10, page_count=1, status="completed")
    doc_b = Document(id="doc-b", owner_id=user_b.id, title="Doc B", filename="b.pdf", storage_path="p", file_size_bytes=10, page_count=1, status="completed")
    sqlite_db_session.add_all([doc_a, doc_b])
    sqlite_db_session.commit()

    chunk_a = DocumentChunk(
        id="c-alpha-secret",
        document_id=doc_a.id,
        chunk_index=0,
        page_number=1,
        text="Alpha secret api key is ALPHA-KEY-7788.",
        embedding=[0.05] * 1536,
        embedding_space=space,
    )
    sqlite_db_session.add(chunk_a)
    sqlite_db_session.commit()

    # User B queries for Alpha's secret
    results = retrieve_authorized_passages("ALPHA-KEY-7788", user_b.id, 5, None, sqlite_db_session)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Test 11: Citation Validation After Retrieval Filtering
# ---------------------------------------------------------------------------

def test_citation_validation_rejects_unauthorized_chunk_ids(sqlite_db_session, monkeypatch):
    """Model output containing a citation to an unauthorized chunk ID is filtered out."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")

    user_a = _create_user(sqlite_db_session, "cit_user_a")
    user_b = _create_user(sqlite_db_session, "cit_user_b")

    doc_a = Document(id="doc-a-cit", owner_id=user_a.id, title="Doc A", filename="a.pdf", storage_path="p", file_size_bytes=10, page_count=1, status="completed")
    sqlite_db_session.add(doc_a)
    sqlite_db_session.commit()

    chunk_a = DocumentChunk(
        id="c-authorized-a",
        document_id=doc_a.id,
        chunk_index=0,
        page_number=1,
        text="Company headquarters are located in New York City.",
        embedding=[0.05] * 1536,
        embedding_space="offline:deterministic-mock:1536",
    )
    sqlite_db_session.add(chunk_a)
    sqlite_db_session.commit()

    passages = retrieve_authorized_passages("headquarters location", user_a.id, 1, doc_a.id, sqlite_db_session)
    assert len(passages) == 1
    assert passages[0].chunk_id == "c-authorized-a"

    # Simulate an answer referencing an unauthorized chunk from another user
    from backend.rag import validate_and_extract_citations
    raw_answer = "Headquarters are in New York [source: c-authorized-a] and backup is in Chicago [source: c-unauthorized-b]."
    citations, clean_ans = validate_and_extract_citations(raw_answer, passages)

    # c-unauthorized-b must be rejected because it is not in passages
    assert len(citations) == 1
    assert citations[0].chunk_id == "c-authorized-a"
    assert all(c.chunk_id != "c-unauthorized-b" for c in citations)


# ---------------------------------------------------------------------------
# Test 12: Offline Mode With Keys Present Proves Zero HTTP Calls
# ---------------------------------------------------------------------------

def test_offline_mode_with_keys_present_makes_zero_http_calls(sqlite_db_session, monkeypatch):
    """When offline mode is selected, zero HTTP network calls are made even if live keys exist."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")
    monkeypatch.setenv("OPENAI_API_KEY", "real-looking-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "real-looking-gemini-key")

    http_called = False

    def mock_post_fail(*args, **kwargs):
        nonlocal http_called
        http_called = True
        raise AssertionError("Unexpected HTTP request in offline mode!")

    monkeypatch.setattr(httpx.Client, "post", mock_post_fail)

    user = _create_user(sqlite_db_session, "zero_call_user")
    doc = Document(id="doc-zero-call", owner_id=user.id, title="Zero Call Doc", filename="z.pdf", storage_path="p", file_size_bytes=10, page_count=1, status="completed")
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="c-zero-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="The server cluster has 99.99% uptime guarantee.",
        embedding=[0.05] * 1536,
        embedding_space="offline:deterministic-mock:1536",
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    passages = retrieve_authorized_passages("uptime guarantee", user.id, 1, doc.id, sqlite_db_session)
    response = generate_rag_answer("uptime guarantee", passages)

    assert http_called is False
    assert response.insufficient_evidence is False
    assert "uptime" in response.answer.lower()


# ---------------------------------------------------------------------------
# Test 13: Provider/Model Configuration Changes During Re-Embedding
# ---------------------------------------------------------------------------

def test_reembed_with_explicit_target_space_override(sqlite_db_session, monkeypatch):
    """Passing an explicit target_space invokes the correct provider override."""
    user = _create_user(sqlite_db_session, "override_space_user")
    doc = Document(id="doc-override", owner_id=user.id, title="Override Doc", filename="o.pdf", storage_path="p", file_size_bytes=10, page_count=1, status="completed")
    sqlite_db_session.add(doc)
    sqlite_db_session.commit()

    chunk = DocumentChunk(
        id="c-override-1",
        document_id=doc.id,
        chunk_index=0,
        page_number=1,
        text="Explicit space override text.",
        embedding=[0.01] * 1536,
        embedding_space="offline:deterministic-mock:1536",
    )
    sqlite_db_session.add(chunk)
    sqlite_db_session.commit()

    called_provider = None

    def mock_get_batch(texts, provider=None, model=None):
        nonlocal called_provider
        called_provider = provider
        return [[0.07] * 1536 for _ in texts]

    monkeypatch.setattr("backend.ingestion.get_embeddings_batch", mock_get_batch)

    reembed_document(doc.id, sqlite_db_session, target_space="gemini:gemini-embedding-001:1536")

    assert called_provider == "gemini"
    sqlite_db_session.refresh(chunk)
    assert chunk.embedding_space == "gemini:gemini-embedding-001:1536"


# ---------------------------------------------------------------------------
# Test 14: SQLite and PostgreSQL Migration Compatibility
# ---------------------------------------------------------------------------

def test_migration_compatibility(sqlite_db_session):
    """Schema migration executes safely and preserves all tables and columns."""
    test_engine = sqlite_db_session.bind
    migrate_schema(test_engine)

    with test_engine.connect() as conn:
        users_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(users);")).fetchall()]
        chunk_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(document_chunks);")).fetchall()]

        assert "is_admin" in users_cols
        assert "embedding_space" in chunk_cols
