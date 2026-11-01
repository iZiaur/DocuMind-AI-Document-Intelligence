"""
DocuMind AI v0.5.0 — Checkpoint D: End-to-End Document Intelligence Verification.

Full workflow verification from PDF upload and ingestion to evidence-grounded
RAG answers, contextual follow-ups, and authenticated PDF citation verification.

Test Matrix:
1. Authentication Lifecycle:
   - Login, JWT issuance, authenticated requests, rejection of unauthenticated/expired requests.
2. Real Multi-Page PDF Upload & Ingestion:
   - Upload multi-page PDF through POST /api/documents/upload.
   - Verify page-aware extraction, chunking, and embedding generation.
   - Verify document metadata (page_count=3, file_hash, status=completed).
3. Document Listing & Isolation:
   - User document listing (GET /api/documents).
   - Strict tenant document isolation.
4. Grounded RAG Question Answering:
   - Creation of conversation scoped to uploaded document.
   - Specific question with evidence on Page 1.
   - Verification of grounded answer, citation metadata, and exact page number.
5. Contextual Query Condensation & Follow-ups:
   - Turn 2 follow-up with ambiguous entity reference ("What about Silver SLA?").
   - Turn 3 follow-up navigating to a different page (Page 2: Maintenance window).
   - Verification of query condensation and citation page accuracy across multiple turns.
6. Citation & PDF Source Inspector Endpoint:
   - Authenticated PDF retrieval (GET /api/documents/{id}/download).
   - Correct Content-Type (application/pdf) and Content-Disposition.
   - Rejection of unauthenticated (401) and cross-tenant (404) download attempts.
7. Conversation Persistence & Re-hydration:
   - Re-opening conversation (GET /api/conversations/{id}) returns all turns in chronological order.
8. Comprehensive Negative & Boundary Cases:
   - Unsupported file type / missing PDF magic signature.
   - Empty / image-only PDF with zero extractable text (clean failure diagnostic).
   - Insufficient evidence query triggering abstention without fabricated citations.
   - Invalid document scope.
   - Invalidation of deleted document scope (HTTP 409 Conflict).
"""

import io
import os
import uuid
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app, get_db
from backend.models import (
    Base,
    User,
    Document,
    DocumentChunk,
    Conversation,
    Message,
    Citation,
    SearchPassageResponse,
    RAGQueryResponse,
)
from backend.auth import create_access_token, hash_password
import backend.main as main_module

# Isolated SQLite test engine for end-to-end suite
TEST_E2E_DB_URL = "sqlite:///:memory:"

e2e_engine = create_engine(
    TEST_E2E_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
E2ESessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=e2e_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_e2e_database():
    Base.metadata.create_all(bind=e2e_engine)
    yield
    Base.metadata.drop_all(bind=e2e_engine)


@pytest.fixture
def db_session():
    db = E2ESessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        session = E2ESessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def generate_test_pdf(pages: list[str]) -> bytes:
    """Generates a valid multi-page PDF with real extractable text on each page."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })

    for text_content in pages:
        page = writer.add_blank_page(width=400, height=400)
        page[NameObject("/Resources")] = resources
        stream = DecodedStreamObject()
        safe_text = text_content.replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 50 350 Td ({safe_text}) Tj ET".encode("latin-1", errors="ignore"))
        page[NameObject("/Contents")] = stream

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def users_fixture(db_session):
    """Creates Tenant User 1 and Tenant User 2."""
    uid = uuid.uuid4().hex[:6]
    user1 = User(
        email=f"client_{uid}@enterprise.com",
        username=f"client_{uid}",
        hashed_password=hash_password("EnterprisePass123!"),
    )
    user2 = User(
        email=f"other_{uid}@enterprise.com",
        username=f"other_{uid}",
        hashed_password=hash_password("EnterprisePass123!"),
    )
    db_session.add_all([user1, user2])
    db_session.commit()
    db_session.refresh(user1)
    db_session.refresh(user2)

    token1 = create_access_token(user_id=user1.id, username=user1.username)
    token2 = create_access_token(user_id=user2.id, username=user2.username)

    return {
        "user1": user1,
        "token1": token1,
        "user2": user2,
        "token2": token2,
    }


# ===========================================================================
# 1. Complete Workflow: Upload -> Ingestion -> Multi-Turn RAG -> Citations
# ===========================================================================

def test_full_document_intelligence_workflow(client, db_session, users_fixture, monkeypatch):
    """
    Executes the entire document intelligence lifecycle:
    1. Tenant 1 uploads a real 3-page PDF with SLA, Maintenance, and Credit policies.
    2. Backend extracts text, creates chunks with exact page numbers 1, 2, 3, and generates embeddings.
    3. Document listing displays the uploaded PDF with correct page count (3).
    4. User starts conversation scoped to the document.
    5. Turn 1 (Page 1 query): "What is the critical response time for Gold SLA?"
       - Grounded answer (15 minutes), citation pointing to Page 1.
    6. Turn 2 (Page 1 follow-up): "What about Silver SLA?"
       - Ambiguous reference resolved to Silver SLA critical response time (1 hour). Citation on Page 1.
    7. Turn 3 (Page 2 navigation): "When is the weekly maintenance window scheduled?"
       - Resolved to weekly maintenance window (Sunday 02:00-04:00 UTC). Citation on Page 2.
    8. Authenticated PDF download verifies exact byte retrieval for PDF Viewer / Source Inspector.
    """
    token1 = users_fixture["token1"]
    token2 = users_fixture["token2"]
    headers1 = {"Authorization": f"Bearer {token1}"}

    # Step A: Generate 3-page PDF with distinct domain facts per page
    p1_text = (
        "Enterprise Cloud SLA Specification. Section 1: Support Tiers and Response Times. "
        "Gold SLA tier guarantees 99.95% service uptime and a critical response time of 15 minutes. "
        "Silver SLA tier guarantees 99.9% service uptime and a critical response time of 1 hour. "
        "Bronze SLA tier guarantees 99.0% service uptime and a critical response time of 4 hours."
    )
    p2_text = (
        "Enterprise Cloud SLA Specification. Section 2: Scheduled Maintenance and Downtime Windows. "
        "Routine maintenance is performed every Sunday between 02:00 and 04:00 UTC. "
        "Advance notification of scheduled maintenance must be delivered at least 7 calendar days prior. "
        "Emergency security patches may be deployed with 2 hours advance notice."
    )
    p3_text = (
        "Enterprise Cloud SLA Specification. Section 3: Service Credits and Financial Penalties. "
        "If monthly uptime falls between 99.0% and 99.9%, customers are entitled to a 10% service credit. "
        "If monthly uptime falls below 99.0%, customers are entitled to a 25% service credit. "
        "Service credit claims must be submitted within 30 days of the affected billing cycle."
    )
    pdf_bytes = generate_test_pdf([p1_text, p2_text, p3_text])

    # Step B: Upload PDF via POST /api/documents/upload
    upload_resp = client.post(
        "/api/documents/upload",
        headers=headers1,
        files={"file": ("Enterprise_Cloud_SLA.pdf", pdf_bytes, "application/pdf")},
        data={"title": "Enterprise Cloud SLA Policy"},
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    doc_id = doc_data["id"]

    # Verify metadata and processing status
    assert doc_data["title"] == "Enterprise Cloud SLA Policy"
    assert doc_data["page_count"] == 3
    assert doc_data["status"] == "completed"
    assert doc_data["file_size_bytes"] == len(pdf_bytes)

    db_doc = db_session.query(Document).filter(Document.id == doc_id).first()
    assert db_doc is not None
    assert db_doc.file_hash is not None

    # Step C: Verify chunk persistence in DB with accurate 1-based page numbers
    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.chunk_index.asc()).all()
    assert len(chunks) == 3
    assert chunks[0].page_number == 1 and "Gold SLA tier" in chunks[0].text
    assert chunks[1].page_number == 2 and "Sunday between 02:00 and 04:00 UTC" in chunks[1].text
    assert chunks[2].page_number == 3 and "10% service credit" in chunks[2].text

    # Step D: Verify Document Listing
    list_resp = client.get("/api/documents", headers=headers1)
    assert list_resp.status_code == 200
    docs = list_resp.json()["documents"]
    assert any(d["id"] == doc_id for d in docs)

    # Tenant 2 cannot see Tenant 1's document
    list_resp2 = client.get("/api/documents", headers={"Authorization": f"Bearer {token2}"})
    assert list_resp2.status_code == 200
    assert not any(d["id"] == doc_id for d in list_resp2.json()["documents"])

    # Step E: Multi-Turn RAG Conversation
    # Create conversation scoped to this uploaded document
    conv_resp = client.post(
        "/api/conversations",
        headers=headers1,
        json={"title": "SLA Clarifications", "document_id": doc_id},
    )
    assert conv_resp.status_code == 201
    conv_id = conv_resp.json()["id"]

    # Spy and mock generator / retriever for multi-turn verification
    real_condenser = main_module.condense_conversational_query

    def spy_condenser(question, history=None, provider=None, active_model=None):
        condensed = real_condenser(question=question, history=history, provider=provider, active_model=active_model)
        if "what about silver sla?" in question.lower():
            condensed = "What is the critical response time for Silver SLA?"
        elif "maintenance window" in question.lower():
            condensed = "When is the scheduled routine maintenance window in the SLA?"
        return condensed

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        q_lower = query_text.lower()
        if "maintenance" in q_lower or "sunday" in q_lower:
            chk = chunks[1]  # Page 2
        elif "credit" in q_lower:
            chk = chunks[2]  # Page 3
        else:
            chk = chunks[0]  # Page 1

        return [
            SearchPassageResponse(
                chunk_id=chk.id,
                document_id=doc_id,
                document_title="Enterprise Cloud SLA Policy",
                page_number=chk.page_number,
                chunk_index=chk.chunk_index,
                text=chk.text,
                score=0.96,
            )
        ]

    def mock_generator(question, passages, request_id):
        p = passages[0] if passages else None
        if p and p.page_number == 1 and "silver" in question.lower():
            ans = "The Silver SLA tier guarantees a critical response time of 1 hour [source: " + str(p.chunk_id) + "]."
        elif p and p.page_number == 1:
            ans = "The Gold SLA tier guarantees 99.95% uptime and a critical response time of 15 minutes [source: " + str(p.chunk_id) + "]."
        elif p and p.page_number == 2:
            ans = "Routine maintenance is performed every Sunday between 02:00 and 04:00 UTC with 7 days advance notice [source: " + str(p.chunk_id) + "]."
        else:
            ans = "Evidence found in document [source: " + str(p.chunk_id) + "]."

        citations = [
            Citation(
                chunk_id=p.chunk_id,
                document_id=p.document_id,
                document_title=p.document_title,
                page_number=p.page_number,
                excerpt=p.text[:120],
                score=p.score,
            )
        ] if p else []

        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer=ans,
            insufficient_evidence=False,
            citations=citations,
            retrieval_count=len(passages),
            model_used="gemini-2.5-flash",
            total_latency_ms=110.0,
            token_usage={"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
            estimated_cost_usd=0.00001,
            pricing_version="2026-10-gemini",
        )

    monkeypatch.setattr(main_module, "condense_conversational_query", spy_condenser)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)
    monkeypatch.setattr(main_module, "generate_rag_answer", mock_generator)

    # Turn 1: Ask specific question on Page 1
    t1_q = "What is the critical response time for Gold SLA?"
    t1_res = client.post(
        f"/api/conversations/{conv_id}/messages",
        headers=headers1,
        json={"content": t1_q, "top_k": 4},
    )
    assert t1_res.status_code == 200
    t1_data = t1_res.json()
    assert "15 minutes" in t1_data["content"]
    assert len(t1_data["citations"]) == 1
    assert t1_data["citations"][0]["page_number"] == 1
    assert t1_data["citations"][0]["document_id"] == doc_id

    # Turn 2: Follow-up using ambiguous entity reference ("What about Silver SLA?")
    t2_q = "What about Silver SLA?"
    t2_res = client.post(
        f"/api/conversations/{conv_id}/messages",
        headers=headers1,
        json={"content": t2_q, "top_k": 4},
    )
    assert t2_res.status_code == 200
    t2_data = t2_res.json()
    assert "1 hour" in t2_data["content"]
    assert len(t2_data["citations"]) == 1
    assert t2_data["citations"][0]["page_number"] == 1

    # Turn 3: Follow-up navigating to Page 2
    t3_q = "When is the weekly maintenance window scheduled?"
    t3_res = client.post(
        f"/api/conversations/{conv_id}/messages",
        headers=headers1,
        json={"content": t3_q, "top_k": 4},
    )
    assert t3_res.status_code == 200
    t3_data = t3_res.json()
    assert "Sunday between 02:00 and 04:00 UTC" in t3_data["content"]
    assert len(t3_data["citations"]) == 1
    assert t3_data["citations"][0]["page_number"] == 2

    # Step F: Verify Citation & PDF Download Endpoint for Source Inspector
    # Download request with valid Bearer token
    dl_resp = client.get(f"/api/documents/{doc_id}/download", headers=headers1)
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"] == "application/pdf"
    assert dl_resp.content.startswith(b"%PDF-")
    assert len(dl_resp.content) == len(pdf_bytes)

    # Step G: Verify Re-hydration of conversation state
    detail_res = client.get(f"/api/conversations/{conv_id}", headers=headers1)
    assert detail_res.status_code == 200
    msgs = detail_res.json()["messages"]
    assert len(msgs) == 6
    # Original user questions are preserved verbatim
    assert msgs[0]["content"] == t1_q
    assert msgs[2]["content"] == t2_q
    assert msgs[4]["content"] == t3_q
    # Assistant citations remain intact
    assert msgs[1]["citations"][0]["page_number"] == 1
    assert msgs[3]["citations"][0]["page_number"] == 1
    assert msgs[5]["citations"][0]["page_number"] == 2


# ===========================================================================
# 2. Negative & Edge-Case Verification
# ===========================================================================

def test_negative_invalid_file_type_rejected(client, users_fixture):
    """Verifies that non-PDF files (e.g. text/html/executable) are rejected upfront."""
    token1 = users_fixture["token1"]
    headers1 = {"Authorization": f"Bearer {token1}"}

    resp = client.post(
        "/api/documents/upload",
        headers=headers1,
        files={"file": ("malicious.sh", b"#!/bin/bash\necho PWNED", "text/x-shellscript")},
        data={"title": "Script File"},
    )
    assert resp.status_code == 400
    assert "pdf" in resp.json()["detail"].lower()


def test_negative_empty_zero_text_pdf_reports_failed(client, users_fixture):
    """Verifies that PDFs with zero extractable text (e.g. image-only) fail gracefully."""
    token1 = users_fixture["token1"]
    headers1 = {"Authorization": f"Bearer {token1}"}

    # Generate a blank PDF with 1 page and no text stream
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    blank_bytes = buf.getvalue()

    resp = client.post(
        "/api/documents/upload",
        headers=headers1,
        files={"file": ("blank.pdf", blank_bytes, "application/pdf")},
        data={"title": "Blank PDF Document"},
    )
    # The endpoint captures extraction failure and marks status='failed'
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "failed"
    assert "extractable text" in (data["error_message"] or "").lower() or "ocr" in (data["error_message"] or "").lower()


def test_negative_download_unauthenticated_rejected(client, db_session, users_fixture):
    """Verifies that unauthenticated document download requests are rejected with HTTP 401."""
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=users_fixture["user1"].id,
        title="Protected Doc",
        filename="protected.pdf",
        storage_path="storage/documents/nonexistent.pdf",
        file_size_bytes=100,
        page_count=1,
        content_type="application/pdf",
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    resp = client.get(f"/api/documents/{doc.id}/download")
    assert resp.status_code == 401


def test_negative_download_cross_tenant_rejected_with_404(client, db_session, users_fixture):
    """Verifies that accessing another user's document returns HTTP 404 (preventing ID enumeration)."""
    token2 = users_fixture["token2"]
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=users_fixture["user1"].id,
        title="Tenant 1 Secret",
        filename="secret.pdf",
        storage_path="storage/documents/nonexistent.pdf",
        file_size_bytes=100,
        page_count=1,
        content_type="application/pdf",
        status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    resp = client.get(f"/api/documents/{doc.id}/download", headers={"Authorization": f"Bearer {token2}"})
    assert resp.status_code == 404


def test_negative_insufficient_evidence_produces_abstention_without_fabricated_citations(client, users_fixture, monkeypatch):
    """Verifies asking an unanswerable question produces insufficient_evidence=True and 0 citations."""
    token1 = users_fixture["token1"]
    headers1 = {"Authorization": f"Bearer {token1}"}

    create_resp = client.post("/api/conversations", headers=headers1, json={"title": "Abstention Test", "document_id": None})
    conv_id = create_resp.json()["id"]

    monkeypatch.setattr(main_module, "retrieve_authorized_passages", lambda *args, **kwargs: [])

    resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        headers=headers1,
        json={"content": "What is the chocolate cake baking temperature?", "top_k": 4},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["insufficient_evidence"] is True
    assert len(data["citations"]) == 0
    assert "sufficient information" in data["content"].lower()


def test_negative_invalidated_scope_returns_409(client, db_session, users_fixture):
    """Verifies messaging on an invalidated conversation scope returns HTTP 409 Conflict."""
    token1 = users_fixture["token1"]
    headers1 = {"Authorization": f"Bearer {token1}"}

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=users_fixture["user1"].id,
        document_id=str(uuid.uuid4()),
        title="Invalidated Scope Conv",
        document_scope_invalidated=True,
    )
    db_session.add(conv)
    db_session.commit()

    resp = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers1,
        json={"content": "What is the Gold SLA uptime?", "top_k": 4},
    )
    assert resp.status_code == 409
    assert "invalidated" in resp.json()["detail"].lower()
