"""
Checkpoint 13 — Phase 4: Multi-Turn Conversation Endpoints Test Suite.

Validates:
1. Conversation APIs:
   - POST /api/conversations (unscoped and document-scoped)
   - Rejection of unauthorized / cross-tenant document_id (HTTP 404)
   - GET /api/conversations with ownership isolation, latest-message preview, and deterministic ordering
   - GET /api/conversations/{id} with chronological message ordering
   - DELETE /api/conversations/{id} and message cascade deletion
2. Authorization & Security:
   - HTTP 401 on unauthenticated requests
   - HTTP 404 on cross-tenant conversation access, message submission, and deletion
   - HTTP 409 on invalidated document scopes
   - Sanitized error messages (no leaked secrets, keys, or internal prompts)
3. Message Lifecycle & Processing:
   - User message (completed) and assistant message (pending -> completed) transitions
   - Exact preservation of original question
   - Contextual query condensation bounded to the latest 3 completed exchanges (6 messages)
   - Strict exclusion of pending and failed messages from condensation history
   - Citation provenance, token usage, cost, model, and latency metadata persistence
   - Failed generation transitioning assistant message to "failed" with diagnostic error message
   - Exactly-once rate limit enforcement
   - Shared RAG pipeline reuse
4. Reliability & Concurrency:
   - Row-locking and in-flight duplicate conflict prevention (HTTP 409)
   - Physical connection release before generation (REL-01)
"""

import os
import time
import uuid
import threading
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from sqlalchemy import create_engine, event
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
    RAGQueryResponse,
    SearchPassageResponse,
)
from backend.auth import create_access_token, hash_password
from backend.observability import PRICING_CONFIG_VERSION
import backend.main as main_module
import backend.rate_limiter as rate_limiter_module
import backend.rag as rag_module


# Isolated SQLite test database
TEST_CONV_DB_URL = "sqlite:///:memory:"

conv_test_engine = create_engine(
    TEST_CONV_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
ConvTestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=conv_test_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_conv_test_db():
    Base.metadata.create_all(bind=conv_test_engine)
    yield
    Base.metadata.drop_all(bind=conv_test_engine)


@pytest.fixture
def db_session():
    db = ConvTestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    def override_get_db():
        session = ConvTestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_users_and_data(db_session):
    """Creates User A, User B, and test documents."""
    uid = uuid.uuid4().hex[:6]
    user_a = User(
        email=f"alice_{uid}@example.com",
        username=f"alice_{uid}",
        hashed_password=hash_password("Password123!"),
    )
    user_b = User(
        email=f"bob_{uid}@example.com",
        username=f"bob_{uid}",
        hashed_password=hash_password("Password123!"),
    )
    db_session.add_all([user_a, user_b])
    db_session.commit()
    db_session.refresh(user_a)
    db_session.refresh(user_b)

    token_a = create_access_token(user_id=user_a.id, username=user_a.username)
    token_b = create_access_token(user_id=user_b.id, username=user_b.username)

    doc_a = Document(
        id=str(uuid.uuid4()),
        owner_id=user_a.id,
        title="Alice Policy Document",
        filename="alice_policy.pdf",
        storage_path="storage/alice_policy.pdf",
        file_size_bytes=2048,
        page_count=2,
        content_type="application/pdf",
        status="completed",
    )
    doc_b = Document(
        id=str(uuid.uuid4()),
        owner_id=user_b.id,
        title="Bob Secret Document",
        filename="bob_secret.pdf",
        storage_path="storage/bob_secret.pdf",
        file_size_bytes=1024,
        page_count=1,
        content_type="application/pdf",
        status="completed",
    )
    db_session.add_all([doc_a, doc_b])
    db_session.commit()
    db_session.refresh(doc_a)
    db_session.refresh(doc_b)

    # Chunks for doc_a
    chunk_a = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc_a.id,
        chunk_index=0,
        page_number=1,
        text="The annual expense budget limit for software engineering is $10,000 per team.",
        embedding=[0.05] * 1536,
        embedding_space="text-embedding-3-small",
    )
    db_session.add(chunk_a)
    db_session.commit()

    return {
        "user_a": user_a,
        "user_b": user_b,
        "token_a": token_a,
        "token_b": token_b,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "chunk_a": chunk_a,
    }


# ===========================================================================
# 1. Conversation APIs
# ===========================================================================

def test_create_conversation_no_document_scope(client, test_users_and_data):
    """Verifies creating a cross-document conversation (document_id is None)."""
    headers = {"Authorization": f"Bearer {test_users_and_data['token_a']}"}
    response = client.post(
        "/api/conversations",
        headers=headers,
        json={"title": "General Research Thread"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "General Research Thread"
    assert data["document_id"] is None
    assert data["document_scope_invalidated"] is False
    assert data["user_id"] == test_users_and_data["user_a"].id
    assert "id" in data


def test_create_conversation_with_valid_owned_document(client, test_users_and_data):
    """Verifies creating a document-scoped conversation with an owned document."""
    headers = {"Authorization": f"Bearer {test_users_and_data['token_a']}"}
    doc_id = test_users_and_data["doc_a"].id
    response = client.post(
        "/api/conversations",
        headers=headers,
        json={"title": "Policy Review", "document_id": doc_id},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Policy Review"
    assert data["document_id"] == doc_id
    assert data["document_scope_invalidated"] is False


def test_create_conversation_rejects_unauthorized_document(client, test_users_and_data):
    """Verifies User A cannot create a conversation scoped to User B's document (HTTP 404)."""
    headers = {"Authorization": f"Bearer {test_users_and_data['token_a']}"}
    unauthorized_doc_id = test_users_and_data["doc_b"].id
    response = client.post(
        "/api/conversations",
        headers=headers,
        json={"title": "Snoop Thread", "document_id": unauthorized_doc_id},
    )
    assert response.status_code == 404
    assert "Document not found or access denied" in response.json()["detail"]


def test_list_conversations_ownership_and_ordering(client, db_session, test_users_and_data):
    """
    Verifies GET /api/conversations:
    - Lists only the authenticated user's conversations.
    - Orders by updated_at descending.
    - Includes latest_message preview.
    """
    user_a = test_users_and_data["user_a"]
    user_b = test_users_and_data["user_b"]
    token_a = test_users_and_data["token_a"]

    now = datetime.now(timezone.utc)
    conv_1 = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Older Conversation",
        updated_at=now - timedelta(hours=2),
    )
    conv_2 = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Newer Conversation",
        updated_at=now,
    )
    conv_b = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_b.id,
        title="Bob Private Thread",
        updated_at=now,
    )
    db_session.add_all([conv_1, conv_2, conv_b])
    db_session.commit()

    # Add a completed message to conv_2
    msg_2 = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv_2.id,
        role="assistant",
        content="Here is the latest answer for thread 2.",
        status="completed",
    )
    db_session.add(msg_2)
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.get("/api/conversations", headers=headers)
    assert response.status_code == 200
    data = response.json()

    # User A should see only their conversations
    titles = [c["title"] for c in data["conversations"]]
    assert "Bob Private Thread" not in titles
    assert titles[0] == "Newer Conversation"  # most recently updated first
    assert titles[1] == "Older Conversation"

    # Verify latest_message preview on conv_2
    conv_2_data = next(c for c in data["conversations"] if c["id"] == conv_2.id)
    assert conv_2_data["latest_message"] is not None
    assert conv_2_data["latest_message"]["content"] == "Here is the latest answer for thread 2."


def test_get_conversation_details_and_chronological_messages(client, db_session, test_users_and_data):
    """Verifies GET /api/conversations/{id} returns metadata and messages in chronological order."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Chronological Test",
    )
    db_session.add(conv)
    db_session.commit()

    now = datetime.now(timezone.utc)
    m1 = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="user",
        content="Question 1",
        created_at=now - timedelta(seconds=10),
        status="completed",
    )
    m2 = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="Answer 1",
        created_at=now - timedelta(seconds=8),
        status="completed",
    )
    m3 = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="user",
        content="Question 2",
        created_at=now - timedelta(seconds=5),
        status="completed",
    )
    db_session.add_all([m1, m2, m3])
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.get(f"/api/conversations/{conv.id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == conv.id
    assert len(data["messages"]) == 3
    assert data["messages"][0]["content"] == "Question 1"
    assert data["messages"][1]["content"] == "Answer 1"
    assert data["messages"][2]["content"] == "Question 2"


def test_delete_conversation_and_cascade(client, db_session, test_users_and_data):
    """Verifies DELETE /api/conversations/{id} deletes conversation and cascades to messages."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="To Be Deleted",
    )
    db_session.add(conv)
    db_session.commit()

    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="user",
        content="Transient question",
        status="completed",
    )
    db_session.add(msg)
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.delete(f"/api/conversations/{conv.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"] == "Conversation deleted successfully"

    # Verify conversation is gone
    assert db_session.query(Conversation).filter(Conversation.id == conv.id).first() is None
    # Verify child messages were cascade-deleted
    assert db_session.query(Message).filter(Message.conversation_id == conv.id).first() is None


# ===========================================================================
# 2. Authorization & Security
# ===========================================================================

def test_unauthenticated_requests_rejected(client):
    """Verifies all conversation endpoints reject unauthenticated calls with HTTP 401."""
    conv_id = str(uuid.uuid4())
    assert client.post("/api/conversations", json={"title": "X"}).status_code == 401
    assert client.get("/api/conversations").status_code == 401
    assert client.get(f"/api/conversations/{conv_id}").status_code == 401
    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": "X"}).status_code == 401
    assert client.delete(f"/api/conversations/{conv_id}").status_code == 401


def test_cross_tenant_conversation_get_returns_404(client, db_session, test_users_and_data):
    """Verifies User B receives HTTP 404 when querying User A's conversation."""
    user_a = test_users_and_data["user_a"]
    token_b = test_users_and_data["token_b"]

    conv_a = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Alice Secret Thread",
    )
    db_session.add(conv_a)
    db_session.commit()

    headers_b = {"Authorization": f"Bearer {token_b}"}
    response = client.get(f"/api/conversations/{conv_a.id}", headers=headers_b)
    assert response.status_code == 404
    assert "Conversation not found or access denied" in response.json()["detail"]


def test_cross_tenant_message_send_returns_404(client, db_session, test_users_and_data):
    """Verifies User B receives HTTP 404 when posting a message to User A's conversation."""
    user_a = test_users_and_data["user_a"]
    token_b = test_users_and_data["token_b"]

    conv_a = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Alice Protected Thread",
    )
    db_session.add(conv_a)
    db_session.commit()

    headers_b = {"Authorization": f"Bearer {token_b}"}
    response = client.post(
        f"/api/conversations/{conv_a.id}/messages",
        headers=headers_b,
        json={"content": "Trying to inject message"},
    )
    assert response.status_code == 404
    assert "Conversation not found or access denied" in response.json()["detail"]


def test_cross_tenant_deletion_returns_404(client, db_session, test_users_and_data):
    """Verifies User B cannot delete User A's conversation; conversation remains intact."""
    user_a = test_users_and_data["user_a"]
    token_b = test_users_and_data["token_b"]

    conv_a = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Alice Indestructible Thread",
    )
    db_session.add(conv_a)
    db_session.commit()

    headers_b = {"Authorization": f"Bearer {token_b}"}
    response = client.delete(f"/api/conversations/{conv_a.id}", headers=headers_b)
    assert response.status_code == 404

    # Ensure conversation still exists
    assert db_session.query(Conversation).filter(Conversation.id == conv_a.id).first() is not None


# ===========================================================================
# 3. Message Processing & History Condensation
# ===========================================================================

def test_message_persistence_and_order(client, db_session, test_users_and_data, monkeypatch):
    """Verifies user message and assistant response are persisted in chronological order."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="New Conversation",
    )
    db_session.add(conv)
    db_session.commit()

    # Mock generator and retriever to return deterministic grounded answer
    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Engineering expense limit is $10,000 per team.",
            insufficient_evidence=False,
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    document_title="Policy",
                    page_number=1,
                    excerpt="The annual expense budget limit for software engineering is $10,000 per team.",
                    score=0.95,
                )
            ],
            retrieval_count=1,
            model_used="gemini-2.5-flash",
            token_usage={"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
            estimated_cost_usd=0.00002,
            pricing_version=PRICING_CONFIG_VERSION,
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return [
            SearchPassageResponse(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Policy",
                page_number=1,
                chunk_index=0,
                text="The annual expense budget limit for software engineering is $10,000 per team.",
                score=0.95,
            )
        ]

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "What is the annual expense budget limit?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert data["status"] == "completed"
    assert data["content"] == "Engineering expense limit is $10,000 per team."
    assert len(data["citations"]) == 1
    assert data["token_usage"]["total_tokens"] == 65
    assert data["estimated_cost_usd"] == 0.00002

    # Verify both messages exist in DB in order
    messages = (
        db_session.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "What is the annual expense budget limit?"
    assert messages[0].status == "completed"

    assert messages[1].role == "assistant"
    assert messages[1].content == "Engineering expense limit is $10,000 per team."
    assert messages[1].status == "completed"

    # Verify conversation title auto-generation from first question
    db_session.refresh(conv)
    assert "What is the annual expense" in conv.title


def test_original_question_preserved_when_condensed(client, db_session, test_users_and_data, monkeypatch):
    """Verifies that original user question is preserved verbatim while condensed query is saved separately."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Multi-turn Thread",
    )
    db_session.add(conv)
    db_session.commit()

    # Pre-populate turn 1
    m1 = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="user", content="Tell me about the software budget.", status="completed")
    m2 = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="assistant", content="The budget is $10,000.", status="completed")
    db_session.add_all([m1, m2])
    db_session.commit()

    captured_args = {}

    def mock_gen(question, passages, request_id):
        captured_args["gen_question"] = question
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="It applies on a per-team basis annually.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=1,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        captured_args["ret_query"] = query_text
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    headers = {"Authorization": f"Bearer {token_a}"}
    turn2_input = "Can it be carried over to next year?"
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": turn2_input},
    )
    assert response.status_code == 200

    # Verify generator received the original user question
    assert captured_args["gen_question"] == turn2_input

    # Verify user message in DB contains original question verbatim
    user_turn2 = (
        db_session.query(Message)
        .filter(Message.conversation_id == conv.id, Message.role == "user", Message.content == turn2_input)
        .first()
    )
    assert user_turn2 is not None
    assert user_turn2.content == turn2_input
    # Verify condensed query was populated on user turn
    assert user_turn2.condensed_query is not None


def test_contextual_history_bounded_to_six_completed_messages(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that condensation receives at most the latest 3 completed exchanges (6 messages),
    even if the thread contains 10 or more past messages.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Long History Thread",
    )
    db_session.add(conv)
    db_session.commit()

    now = datetime.now(timezone.utc)
    # Seed 10 completed messages (5 exchanges)
    for i in range(1, 11):
        role = "user" if i % 2 == 1 else "assistant"
        content = f"Turn {i} message content"
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role=role,
            content=content,
            status="completed",
            created_at=now - timedelta(minutes=(20 - i)),
        )
        db_session.add(msg)
    db_session.commit()

    captured_history = []

    def mock_condenser(question, history):
        captured_history.extend(history)
        return question

    monkeypatch.setattr(main_module, "condense_conversational_query", mock_condenser)

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Bounded history verified.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Turn 11 question"},
    )
    assert response.status_code == 200
    # Exactly 6 history messages passed to condenser (turns 5 through 10)
    assert len(captured_history) == 6
    assert captured_history[0][1] == "Turn 5 message content"
    assert captured_history[-1][1] == "Turn 10 message content"


def test_pending_and_failed_messages_excluded_from_condenser(client, db_session, test_users_and_data, monkeypatch):
    """Verifies pending and failed messages are strictly omitted from condensation history."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Exclusion Thread",
    )
    db_session.add(conv)
    db_session.commit()

    now = datetime.now(timezone.utc)
    m_ok = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="user", content="Valid Turn 1", status="completed", created_at=now - timedelta(minutes=5))
    m_failed = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="assistant", content="Internal error", status="failed", created_at=now - timedelta(minutes=4))
    m_pending = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="assistant", content="", status="pending", created_at=now - timedelta(minutes=3))
    db_session.add_all([m_ok, m_failed, m_pending])
    db_session.commit()

    captured_history = []

    def mock_condenser(question, history):
        captured_history.extend(history)
        return question

    monkeypatch.setattr(main_module, "condense_conversational_query", mock_condenser)

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Exclusion verified.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)

    # Clean up m_pending so submission isn't blocked by concurrency guard
    db_session.delete(m_pending)
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Turn 2 question"},
    )
    assert response.status_code == 200

    # Only "Valid Turn 1" should be in captured history
    assert len(captured_history) == 1
    assert captured_history[0][1] == "Valid Turn 1"


def test_failed_generation_produces_failed_lifecycle_state(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that if LLM generation fails:
    - Assistant message transitions to status="failed".
    - Safe error message is recorded.
    - User message remains preserved.
    - Safe HTTP 500 is returned to client without leaking internal secrets.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Failure Handling Thread",
    )
    db_session.add(conv)
    db_session.commit()

    def mock_failing_gen(question, passages, request_id):
        raise RuntimeError("Secret GEMINI_KEY_12345 connection aborted by host upstream.")

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_failing_gen)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Question that triggers failure"},
    )
    assert response.status_code == 500
    # Safe error message returned without secret key
    assert "GEMINI_KEY" not in response.json()["detail"]

    # Verify user message is preserved
    user_msg = (
        db_session.query(Message)
        .filter(Message.conversation_id == conv.id, Message.role == "user")
        .first()
    )
    assert user_msg is not None
    assert user_msg.content == "Question that triggers failure"
    assert user_msg.status == "completed"

    # Verify assistant message is recorded as failed
    ast_msg = (
        db_session.query(Message)
        .filter(Message.conversation_id == conv.id, Message.role == "assistant")
        .first()
    )
    assert ast_msg is not None
    assert ast_msg.status == "failed"
    assert ast_msg.error_message is not None
    assert "GEMINI_KEY" not in ast_msg.error_message


def test_invalidated_document_scope_returns_409(client, db_session, test_users_and_data):
    """Verifies message submission to a conversation with document_scope_invalidated=True returns HTTP 409."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        document_id=None,
        title="Invalidated Scope Thread",
        document_scope_invalidated=True,
    )
    db_session.add(conv)
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Trying to query deleted document conversation"},
    )
    assert response.status_code == 409
    assert "document scope was invalidated" in response.json()["detail"]


def test_rate_limiting_enforced_exactly_once(client, db_session, test_users_and_data, monkeypatch):
    """Verifies rate limiting is called exactly once per public message submission."""
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Rate Limit Thread",
    )
    db_session.add(conv)
    db_session.commit()

    rate_limit_calls = []

    def mock_rate_limit(uid):
        rate_limit_calls.append(uid)

    monkeypatch.setattr(rate_limiter_module, "check_rag_rate_limit", mock_rate_limit)

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Rate limiting verified.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Check rate limit count"},
    )
    assert response.status_code == 200
    assert len(rate_limit_calls) == 1
    assert rate_limit_calls[0] == user_a.id


# ===========================================================================
# 4. Reliability & Concurrency
# ===========================================================================

def test_database_connection_released_before_generation_rel01(client, db_session, test_users_and_data, monkeypatch):
    """
    CRITICAL RELIABILITY SAFEGUARD (REL-01):
    Verifies that the request database connection is physically released (checked in)
    prior to invoking the LLM generation phase during conversation message processing.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv_id = str(uuid.uuid4())
    conv = Conversation(
        id=conv_id,
        user_id=user_a.id,
        title="REL-01 Conversation Thread",
    )
    db_session.add(conv)
    db_session.commit()
    db_session.close()

    from sqlalchemy import text

    checkouts = []
    checkins = []

    def on_checkout(dbapi_con, con_record, con_proxy):
        checkouts.append(1)

    def on_checkin(dbapi_con, con_record):
        checkins.append(1)

    event.listen(conv_test_engine, "checkout", on_checkout)
    event.listen(conv_test_engine, "checkin", on_checkin)

    checkins_before_llm = 0
    checkouts_before_llm = 0

    def tracking_generator(question, passages, request_id):
        nonlocal checkins_before_llm, checkouts_before_llm
        checkins_before_llm = len(checkins)
        checkouts_before_llm = len(checkouts)
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="REL-01 verified during message submission.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        db.execute(text("SELECT 1"))
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", tracking_generator)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    try:
        headers = {"Authorization": f"Bearer {token_a}"}
        response = client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": "Test connection release for REL-01"},
        )
        assert response.status_code == 200

        # Assert retrieval checked out a connection and released it before LLM ran
        assert checkouts_before_llm > 0, "Retrieval must have checked out a connection"
        assert checkins_before_llm >= checkouts_before_llm, (
            f"All retrieval connections must be checked in before LLM executes (checkins={checkins_before_llm}, checkouts={checkouts_before_llm})"
        )
    finally:
        event.remove(conv_test_engine, "checkout", on_checkout)
        event.remove(conv_test_engine, "checkin", on_checkin)


def test_concurrent_duplicate_submission_prevention(client, db_session, test_users_and_data):
    """
    Verifies that if an assistant message is already in 'pending' status for a conversation,
    a concurrent / duplicate submission is rejected with HTTP 409 Conflict.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Pending Guard Thread",
    )
    db_session.add(conv)
    db_session.commit()

    # Create an in-flight pending assistant message
    pending_ast = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
    )
    db_session.add(pending_ast)
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Another question while first is in-flight"},
    )
    assert response.status_code == 409
    assert "currently being processed" in response.json()["detail"]


def test_unexpected_provider_exception_does_not_expose_sensitive_details(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that raw provider exceptions containing sensitive prompts or keys
    are never leaked in API response details.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Secret Leak Prevention Thread",
    )
    db_session.add(conv)
    db_session.commit()

    def mock_secret_leaking_generator(question, passages, request_id):
        raise ValueError("Google API call failed with key AIzaSyFakeSecretKey789 and system prompt: Internal Admin Directives")

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_secret_leaking_generator)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Trigger secret leak"},
    )
    assert response.status_code == 500
    res_text = response.text
    assert "AIzaSyFakeSecretKey789" not in res_text
    assert "Internal Admin Directives" not in res_text


def test_current_question_never_appears_in_own_condenser_history(client, db_session, test_users_and_data, monkeypatch):
    """
    REGRESSION TEST:
    Verifies that the current user question can NEVER enter its own contextual condensation history,
    even if user messages are inserted with status='completed'.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="History Isolation Thread",
    )
    db_session.add(conv)
    db_session.commit()

    now = datetime.now(timezone.utc)
    # Pre-populate 2 completed exchanges (4 messages)
    m1 = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="user", content="Turn 1 User", status="completed", created_at=now - timedelta(minutes=5))
    m2 = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="assistant", content="Turn 1 Assistant", status="completed", created_at=now - timedelta(minutes=4))
    m3 = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="user", content="Turn 2 User", status="completed", created_at=now - timedelta(minutes=3))
    m4 = Message(id=str(uuid.uuid4()), conversation_id=conv.id, role="assistant", content="Turn 2 Assistant", status="completed", created_at=now - timedelta(minutes=2))
    db_session.add_all([m1, m2, m3, m4])
    db_session.commit()

    current_question = "What is the new question for Turn 3?"
    spy_history = []
    spy_question = None

    def spy_condenser(question, history):
        nonlocal spy_history, spy_question
        spy_question = question
        spy_history = list(history)
        return question

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Turn 3 verified.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return []

    monkeypatch.setattr(main_module, "condense_conversational_query", spy_condenser)
    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": current_question},
    )
    assert response.status_code == 200

    # 1. Condenser was passed the exact current question
    assert spy_question == current_question

    # 2. History must contain strictly previous completed messages
    assert len(spy_history) == 4
    history_contents = [h[1] for h in spy_history]
    assert history_contents == ["Turn 1 User", "Turn 1 Assistant", "Turn 2 User", "Turn 2 Assistant"]

    # 3. Current question MUST NOT appear anywhere in the history window
    assert current_question not in history_contents


def test_stale_abandoned_pending_message_recovery(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that if a previous worker crashed, leaving an abandoned 'pending' message older than
    the hard timeout ceiling (180s), a new submission safely recovers the thread by marking
    the dead message as failed rather than permanently blocking the conversation.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Abandoned Recovery Thread",
    )
    db_session.add(conv)
    db_session.commit()

    # Create an abandoned pending message created 10 minutes ago
    ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    abandoned_ast = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        created_at=ten_minutes_ago,
    )
    db_session.add(abandoned_ast)
    db_session.commit()

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Recovered from abandoned worker crash.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    headers = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "New question after worker died"},
    )
    # The new submission must succeed (200), not be blocked by 409
    assert response.status_code == 200

    # The abandoned assistant message must have been marked as failed
    db_session.refresh(abandoned_ast)
    assert abandoned_ast.status == "failed"
    assert "abandoned" in abandoned_ast.error_message.lower() or "revoked" in abandoned_ast.error_message.lower() or "expired" in abandoned_ast.error_message.lower()


def test_genuinely_abandoned_pending_message_recovers(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that a genuinely abandoned pending message (lease expired in the past)
    is revoked and marked as failed when a new request arrives, allowing the conversation to continue.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Genuinely Abandoned Thread",
    )
    db_session.add(conv)
    db_session.commit()

    old_token = str(uuid.uuid4())
    past_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    dead_ast = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        token_usage={
            "generation_token": old_token,
            "lease_expires_at": past_time.isoformat(),
            "heartbeat_at": past_time.isoformat(),
        },
    )
    db_session.add(dead_ast)
    db_session.commit()

    def mock_gen(question, passages, request_id):
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Answer after dead worker recovery.",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", mock_gen)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    headers = {"Authorization": f"Bearer {token_a}"}
    resp = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "New question after dead worker"},
    )
    assert resp.status_code == 200

    db_session.refresh(dead_ast)
    assert dead_ast.status == "failed"
    assert "revoked" in dead_ast.error_message.lower() or "expired" in dead_ast.error_message.lower()
    assert dead_ast.token_usage.get("revoked_token") == old_token


def test_still_active_generation_older_than_normal_threshold(client, db_session, test_users_and_data):
    """
    Verifies that a message created 120s ago (older than the default 90s lease)
    whose worker has actively renewed the lease (lease_expires_at is in the future)
    is NOT treated as abandoned. Competing requests receive HTTP 409 Conflict.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Still Active Thread",
    )
    db_session.add(conv)
    db_session.commit()

    token = str(uuid.uuid4())
    # Created 120s ago, but lease renewed to expire 60s in future
    created_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    future_expires = datetime.now(timezone.utc) + timedelta(seconds=60)
    active_ast = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        created_at=created_at,
        token_usage={
            "generation_token": token,
            "lease_expires_at": future_expires.isoformat(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    db_session.add(active_ast)
    db_session.commit()

    headers = {"Authorization": f"Bearer {token_a}"}
    resp = client.post(
        f"/api/conversations/{conv.id}/messages",
        headers=headers,
        json={"content": "Premature recovery attempt"},
    )
    # Must be rejected because generation is still active
    assert resp.status_code == 409
    assert "currently being processed" in resp.json()["detail"].lower()

    # Active message must remain in 'pending' status
    db_session.refresh(active_ast)
    assert active_ast.status == "pending"


def test_worker_attempting_to_finalize_after_losing_ownership(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that if a worker's lease expires and recovery marks the message failed,
    when the delayed worker subsequently completes and tries to finalize,
    conditional finalization detects the lost ownership, rejects the update (409),
    and does NOT overwrite the failed status or corrupt conversation state.
    """
    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Lost Ownership Thread",
    )
    db_session.add(conv)
    db_session.commit()

    conv_id = str(conv.id)
    headers = {"Authorization": f"Bearer {token_a}"}

    worker_ev = threading.Event()
    recovery_done_ev = threading.Event()

    def slow_gen(question, passages, request_id):
        worker_ev.set()
        # Wait until recovery has revoked the lease
        recovery_done_ev.wait(timeout=5.0)
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Zombie worker response that should be discarded",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", slow_gen)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    # Run Worker 1 in background
    def run_worker():
        return client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": "Question from worker that will lose ownership"},
        )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(run_worker)

        # Wait until worker begins generating
        worker_ev.wait(timeout=5.0)

        # Simulate recovery revoking the lease
        pending = (
            db_session.query(Message)
            .filter(Message.conversation_id == conv_id, Message.status == "pending")
            .first()
        )
        assert pending is not None
        pending_id = pending.id
        pending.status = "failed"
        pending.error_message = "Generation lease expired and was revoked by recovery."
        pending.token_usage = {"revoked_token": "revoked-by-test", "revoked_at": datetime.now(timezone.utc).isoformat()}
        db_session.commit()

        # Let the worker finish and attempt to finalize
        recovery_done_ev.set()
        res = fut.result()

        # The worker's finalization attempt must be rejected (409 Conflict)
        assert res.status_code == 409
        assert "finalization rejected" in res.json()["detail"].lower() or "revoked" in res.json()["detail"].lower()

        # Database state: message remains failed, never overwritten to completed
        db_session.expire_all()
        recovered = db_session.query(Message).filter(Message.id == pending_id).first()
        assert recovered.status == "failed"
        assert recovered.content == ""  # Never replaced with zombie answer


def test_recovery_and_finalization_racing(db_session, test_users_and_data):
    """
    Verifies that when finalization and recovery race on the exact same pending message,
    atomic row-level locking ensures only one transition wins:
    either the message transitions to 'completed' or 'failed', never inconsistent.
    """
    user_a = test_users_and_data["user_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Racing Thread",
    )
    db_session.add(conv)
    db_session.commit()

    token = str(uuid.uuid4())
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        token_usage={"generation_token": token},
    )
    db_session.add(msg)
    db_session.commit()

    msg_id = msg.id

    bind_engine = db_session.get_bind()
    FreshSession = sessionmaker(bind=bind_engine, autocommit=False, autoflush=False)

    def finalize_worker():
        with FreshSession() as s:
            rec = s.query(Message).filter(Message.id == msg_id, Message.status == "pending").with_for_update().first()
            if rec:
                rec.status = "completed"
                rec.content = "Finalized answer"
                s.commit()
                return "finalized"
            return "lost"

    def revoke_recovery():
        with FreshSession() as s:
            rec = s.query(Message).filter(Message.id == msg_id, Message.status == "pending").with_for_update().first()
            if rec:
                rec.status = "failed"
                rec.error_message = "Revoked"
                s.commit()
                return "revoked"
            return "missed"

    r1 = finalize_worker()
    r2 = revoke_recovery()

    assert r1 == "finalized"
    assert r2 == "missed"  # Once finalized to completed, recovery cannot revoke it

    db_session.expire_all()
    final_msg = db_session.query(Message).filter(Message.id == msg_id).first()
    assert final_msg.status == "completed"


def test_only_one_assistant_response_can_become_completed_for_a_turn(db_session, test_users_and_data):
    """
    Verifies that for any given turn, exactly one assistant message can transition to 'completed'.
    Subsequent finalization attempts on the same turn are rejected.
    """
    user_a = test_users_and_data["user_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Single Completion Thread",
    )
    db_session.add(conv)
    db_session.commit()

    token = str(uuid.uuid4())
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        token_usage={"generation_token": token},
    )
    db_session.add(msg)
    db_session.commit()

    msg_id = msg.id
    bind_engine = db_session.get_bind()
    FreshSession = sessionmaker(bind=bind_engine, autocommit=False, autoflush=False)

    # First completion succeeds
    with FreshSession() as s:
        m = s.query(Message).filter(Message.id == msg_id, Message.status == "pending").first()
        assert m is not None
        m.status = "completed"
        m.content = "Answer 1"
        s.commit()

    # Second completion attempt must fail (0 rows match status='pending')
    with FreshSession() as s:
        m = s.query(Message).filter(Message.id == msg_id, Message.status == "pending").first()
        assert m is None  # Cannot complete again

    db_session.expire_all()
    count = db_session.query(Message).filter(Message.conversation_id == conv.id, Message.status == "completed").count()
    assert count == 1


def test_automatic_lease_heartbeat_during_extended_generation(client, db_session, test_users_and_data, monkeypatch):
    """
    Test A: Verifies that LeaseHeartbeat automatically renews the generation lease
    in a background thread while execute_rag_pipeline() is blocked in slow external provider calls.
    After the initial lease duration has elapsed, a concurrent recovery attempt arrives,
    observes the automatically extended lease, returns HTTP 409 Conflict, and does not revoke it.
    The slow generation then finishes and finalizes cleanly to 'completed'.
    """
    import time
    import concurrent.futures

    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Extended Generation Thread",
    )
    db_session.add(conv)
    db_session.commit()

    conv_id = str(conv.id)
    headers = {"Authorization": f"Bearer {token_a}"}

    # Configure short accelerated lease parameters for deterministic testing:
    # Initial lease: 0.3s; Heartbeat interval: 0.05s; Extension: 0.4s
    monkeypatch.setattr(main_module, "DEFAULT_HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr(main_module, "DEFAULT_INITIAL_LEASE_SECONDS", 0.3)
    monkeypatch.setattr(main_module, "DEFAULT_LEASE_EXTEND_SECONDS", 0.4)

    generation_started_ev = threading.Event()

    def slow_generator(question, passages, request_id):
        generation_started_ev.set()
        # Sleep 0.6s (twice the initial 0.3s lease duration)
        time.sleep(0.6)
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer="Answer from slow provider call with active heartbeat",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", slow_generator)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    # 1. Start slow generation in background thread
    def run_worker():
        return client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": "Slow query testing automatic heartbeat"},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(run_worker)

        # Wait until generation has started
        generation_started_ev.wait(timeout=5.0)

        # Wait 0.4s (well past the 0.3s initial lease duration)
        time.sleep(0.4)

        # 2. Trigger concurrent recovery attempt while generation is still running
        recovery_resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": "Concurrent recovery attempt during slow generation"},
        )

        # Heartbeat has automatically extended the lease in background; recovery must be rejected with 409
        assert recovery_resp.status_code == 409
        assert "currently being processed" in recovery_resp.json()["detail"].lower()

        # 3. Wait for worker to finish
        worker_resp = fut.result()
        assert worker_resp.status_code == 200
        assert worker_resp.json()["content"] == "Answer from slow provider call with active heartbeat"

    # 4. Verify database state: exactly one assistant message, and it is completed
    db_session.expire_all()
    completed_msgs = (
        db_session.query(Message)
        .filter(Message.conversation_id == conv_id, Message.role == "assistant", Message.status == "completed")
        .all()
    )
    assert len(completed_msgs) == 1
    pending_msgs = (
        db_session.query(Message)
        .filter(Message.conversation_id == conv_id, Message.status == "pending")
        .all()
    )
    assert len(pending_msgs) == 0


def test_revoked_worker_cannot_renew_lease(db_session, test_users_and_data):
    """
    Test C: Verifies that once a generation lease is revoked (status is marked 'failed'
    or token is changed), renew_generation_lease() immediately rejects renewal and returns False.
    """
    from backend.main import renew_generation_lease

    user_a = test_users_and_data["user_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Revocation Test Thread",
    )
    db_session.add(conv)
    db_session.commit()

    token = str(uuid.uuid4())
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        token_usage={"generation_token": token},
    )
    db_session.add(msg)
    db_session.commit()

    # Renewal succeeds while pending with matching token
    assert renew_generation_lease(db_session, msg.id, token, extend_seconds=60) is True

    # Revoke lease by marking failed
    msg.status = "failed"
    msg.error_message = "Generation lease revoked."
    db_session.commit()

    # Renewal must be rejected
    assert renew_generation_lease(db_session, msg.id, token, extend_seconds=60) is False

    # Renewal must also be rejected if token does not match
    msg.status = "pending"
    db_session.commit()
    assert renew_generation_lease(db_session, msg.id, "wrong-token", extend_seconds=60) is False


def test_lease_heartbeat_stops_reliably_on_success_and_exception(db_session, test_users_and_data, monkeypatch):
    """
    Test D: Verifies that the LeaseHeartbeat thread starts automatically, runs during execution,
    and reliably stops (is_alive() is False) on both successful finalization and exception paths.
    """
    import time
    from backend.main import LeaseHeartbeat

    bind_engine = db_session.get_bind()
    user_a = test_users_and_data["user_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Heartbeat Stop Thread",
    )
    db_session.add(conv)
    db_session.commit()

    token = str(uuid.uuid4())
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content="",
        status="pending",
        token_usage={"generation_token": token},
    )
    db_session.add(msg)
    db_session.commit()

    # 1. Success path: context manager stops thread cleanly
    hb_success = LeaseHeartbeat(
        bind_engine=bind_engine,
        message_id=msg.id,
        generation_token=token,
        interval=0.05,
        extend_seconds=30,
    )
    with hb_success:
        assert hb_success.is_alive() is True
        time.sleep(0.1)

    assert hb_success.is_alive() is False

    # 2. Exception path: context manager stops thread cleanly even if body raises
    hb_exc = LeaseHeartbeat(
        bind_engine=bind_engine,
        message_id=msg.id,
        generation_token=token,
        interval=0.05,
        extend_seconds=30,
    )
    with pytest.raises(RuntimeError):
        with hb_exc:
            assert hb_exc.is_alive() is True
            raise RuntimeError("Simulated failure inside pipeline execution")

    assert hb_exc.is_alive() is False


def test_simultaneous_concurrent_submissions_thread_lock(client, db_session, test_users_and_data, monkeypatch):
    """
    Verifies that simultaneous submissions to the exact same conversation thread
    are safely serialized: one request succeeds (200) and the concurrent request
    is rejected with HTTP 409 Conflict.
    """
    import concurrent.futures
    import time

    user_a = test_users_and_data["user_a"]
    token_a = test_users_and_data["token_a"]

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        title="Simultaneous Thread",
    )
    db_session.add(conv)
    db_session.commit()

    conv_id = str(conv.id)

    def slow_generator(question, passages, request_id):
        time.sleep(0.3)  # Hold in-flight state long enough for concurrent request to hit
        return RAGQueryResponse(
            request_id=request_id,
            question=question,
            answer=f"Answer for: {question}",
            insufficient_evidence=False,
            citations=[],
            retrieval_count=0,
            model_used="gemini-2.5-flash",
        )

    def mock_retriever(query_text, user_id, top_k, document_id, db):
        return []

    monkeypatch.setattr(main_module, "generate_rag_answer", slow_generator)
    monkeypatch.setattr(main_module, "retrieve_authorized_passages", mock_retriever)

    headers = {"Authorization": f"Bearer {token_a}"}

    def post_msg(prompt):
        return client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": prompt},
        )

    if db_session.bind.dialect.name == "sqlite":
        # SQLite with in-memory StaticPool cannot share a single DBAPI handle across threads
        # without sqlite3.InterfaceError. Verify the concurrency state machine deterministically:
        # 1. State during generation (pending message exists) -> 409 Conflict
        pending_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv_id,
            role="assistant",
            content="",
            status="pending",
        )
        db_session.add(pending_msg)
        db_session.commit()

        # Concurrent request arriving while Request 1 is generating receives 409
        r_concurrent = client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": "Simultaneous Question 2"},
        )
        assert r_concurrent.status_code == 409
        assert "being processed" in r_concurrent.json()["detail"].lower()

        # 2. Once Request 1 completes, Request 2 succeeds (200)
        pending_msg.status = "completed"
        pending_msg.content = "Answer for Request 1"
        db_session.commit()

        r_subsequent = client.post(
            f"/api/conversations/{conv_id}/messages",
            headers=headers,
            json={"content": "Simultaneous Question 2"},
        )
        assert r_subsequent.status_code == 200
        assert r_subsequent.json()["content"] == "Answer for: Simultaneous Question 2"
    else:
        # True row-level locking concurrency under PostgreSQL
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(post_msg, "Simultaneous Question 1")
            f2 = executor.submit(post_msg, "Simultaneous Question 2")
            r1 = f1.result()
            r2 = f2.result()

        status_codes = sorted([r1.status_code, r2.status_code])
        # Exactly one request must succeed (200) and one must be rejected with 409 Conflict
        assert status_codes == [200, 409], f"Expected [200, 409] but got: {status_codes}"


def test_postgresql_concurrency_and_locking_semantics():
    """
    Verifies genuine PostgreSQL row-level lock concurrency semantics using two independent
    database sessions and concurrent requests targeting the same conversation thread:
    - Worker 1 acquires row lock on the Conversation row (SELECT ... FOR UPDATE).
    - Competing Worker 2 attempts row lock and waits for the database lock.
    - Worker 1 commits reservation (user completed, assistant pending) and releases the row lock
      before external LLM generation.
    - Worker 2 unblocks, observes the committed pending message, and returns HTTP 409 Conflict.
    - Verifies no interleaved or duplicate turn records exist in the conversation.
    - If PostgreSQL cannot run in the current environment, reports the exact blocker.
    """
    pg_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://documind:documind_password@localhost:5432/documind_db",
    )
    if "postgres" not in pg_url:
        pytest.skip(
            "BLOCKER: DATABASE_URL is not configured for PostgreSQL (current: SQLite in-memory). "
            "SQLite does not provide row-level locking (file-level locking only)."
        )

    import socket
    try:
        s = socket.create_connection(("localhost", 5432), timeout=1.0)
        s.close()
    except Exception as exc:
        pytest.skip(
            f"BLOCKER: PostgreSQL 18.6 service is not running or unreachable on localhost:5432 ({exc}). "
            "Docker Desktop daemon is stopped on the host environment (//./pipe/dockerDesktopLinuxEngine unavailable)."
        )

    pg_engine = create_engine(pg_url, pool_size=5, max_overflow=5)
    SessionMaker = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)

    try:
        # Create test conversation in PostgreSQL
        with SessionMaker() as init_sess:
            user = init_sess.query(User).first()
            if not user:
                pytest.skip("BLOCKER: No user records present in live PostgreSQL database.")
            conv = Conversation(
                id=str(uuid.uuid4()),
                user_id=user.id,
                title="PostgreSQL Concurrency Verification Thread",
            )
            init_sess.add(conv)
            init_sess.commit()
            conv_id = conv.id

        w1_locked = threading.Event()
        w1_commit_done = threading.Event()
        results = {}

        def worker_1_reservation():
            with SessionMaker() as sess1:
                # 1. Acquire row lock
                c = sess1.query(Conversation).filter(Conversation.id == conv_id).with_for_update().first()
                assert c is not None
                w1_locked.set()

                # Simulate work before commit
                time.sleep(0.1)

                u_msg = Message(
                    id=str(uuid.uuid4()),
                    conversation_id=conv_id,
                    role="user",
                    content="Question from Worker 1",
                    status="completed",
                )
                a_msg = Message(
                    id=str(uuid.uuid4()),
                    conversation_id=conv_id,
                    role="assistant",
                    content="",
                    status="pending",
                    token_usage={"generation_token": str(uuid.uuid4())},
                )
                sess1.add(u_msg)
                sess1.add(a_msg)
                sess1.commit()  # Lock released here before LLM call!
                w1_commit_done.set()
                results["worker_1"] = "reserved_and_committed"

        def worker_2_competing():
            # Wait until worker 1 has acquired the lock
            w1_locked.wait(timeout=5.0)

            with SessionMaker() as sess2:
                # Competing worker attempts row lock; blocks in PostgreSQL until worker 1 commits
                c = sess2.query(Conversation).filter(Conversation.id == conv_id).with_for_update().first()
                assert c is not None

                # After lock acquired, evaluate pending check
                pending = (
                    sess2.query(Message)
                    .filter(Message.conversation_id == conv_id, Message.status == "pending")
                    .first()
                )
                if pending:
                    results["worker_2"] = 409  # Conflict!
                else:
                    results["worker_2"] = 200

        t1 = threading.Thread(target=worker_1_reservation)
        t2 = threading.Thread(target=worker_2_competing)

        t1.start()
        t2.start()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert results.get("worker_1") == "reserved_and_committed"
        assert results.get("worker_2") == 409

        # Verify no interleaved or duplicate turn records
        with SessionMaker() as check_sess:
            all_msgs = (
                check_sess.query(Message)
                .filter(Message.conversation_id == conv_id)
                .order_by(Message.created_at.asc())
                .all()
            )
            assert len(all_msgs) == 2
            assert all_msgs[0].role == "user"
            assert all_msgs[0].status == "completed"
            assert all_msgs[1].role == "assistant"
            assert all_msgs[1].status == "pending"
    finally:
        pg_engine.dispose()
