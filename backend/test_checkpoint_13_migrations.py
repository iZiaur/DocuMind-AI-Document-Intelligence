"""
Checkpoint 13 Migration Test Suite: Multi-Turn Conversations & Messages Schema.

Tests:
1. Migration 0002 & 0003 upgrade cleanly and create conversations/messages tables.
2. Verified non-redundant primary key indexes and presence of composite indexes.
3. Redundant standalone index on messages.conversation_id is eliminated.
4. Stable tie-breaking index ix_messages_conv_created covers (conversation_id, created_at, id).
5. Sidebar index ix_conversations_user_updated orders by (user_id, updated_at DESC).
6. Scope Invalidation:
   - BEFORE DELETE trigger on documents sets document_scope_invalidated = True
     before document_id is cleared to NULL by ON DELETE SET NULL foreign key.
   - Cross-document conversations (document_id is NULL initially) remain unaffected.
7. Cascade delete behavior: deleting user cascades to conversations and messages.
8. Live PostgreSQL trigger and schema verification.
"""

import os
import uuid
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, text, inspect, event
from sqlalchemy.orm import sessionmaker

from backend.models import Base, User, Document, DocumentChunk, Conversation, Message
from backend.migrations import (
    upgrade_database,
    get_current_revision,
    get_head_revision,
    check_migration_status,
)


def test_checkpoint_13_migration_on_isolated_sqlite(tmp_path):
    """Verifies migrations up to 0003 upgrade cleanly on an isolated SQLite database."""
    db_file = tmp_path / "test_cp13.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)

    # 1. Upgrade from scratch to head (0001 -> 0002 -> 0003)
    upgrade_database(db_url=db_url, revision="head")
    assert get_current_revision(engine) == "0003_scope_and_index_fixes"

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "conversations" in tables
    assert "messages" in tables

    # 2. Verify conversations columns including document_scope_invalidated
    conv_cols = {c["name"]: c for c in inspector.get_columns("conversations")}
    assert "id" in conv_cols
    assert "user_id" in conv_cols
    assert "document_id" in conv_cols
    assert "title" in conv_cols
    assert "document_scope_invalidated" in conv_cols
    assert "created_at" in conv_cols
    assert "updated_at" in conv_cols

    # 3. Verify messages columns
    msg_cols = {c["name"]: c for c in inspector.get_columns("messages")}
    assert "id" in msg_cols
    assert "conversation_id" in msg_cols
    assert "role" in msg_cols
    assert "content" in msg_cols
    assert "condensed_query" in msg_cols
    assert "status" in msg_cols
    assert "error_message" in msg_cols
    assert "citations" in msg_cols
    assert "token_usage" in msg_cols
    assert "estimated_cost_usd" in msg_cols
    assert "pricing_version" in msg_cols
    assert "retrieval_latency_ms" in msg_cols
    assert "generation_latency_ms" in msg_cols
    assert "total_latency_ms" in msg_cols
    assert "model_used" in msg_cols
    assert "insufficient_evidence" in msg_cols
    assert "created_at" in msg_cols

    # 4. Verify indexes
    conv_indexes = {idx["name"]: idx for idx in inspector.get_indexes("conversations")}
    assert "ix_conversations_user_updated" in conv_indexes
    assert conv_indexes["ix_conversations_user_updated"]["column_names"] == ["user_id", "updated_at"]
    assert "ix_conversations_user_id" in conv_indexes
    assert "ix_conversations_document_id" in conv_indexes
    # Ensure there is no redundant PK index
    assert "ix_conversations_id" not in conv_indexes

    msg_indexes = {idx["name"]: idx for idx in inspector.get_indexes("messages")}
    assert "ix_messages_conv_created" in msg_indexes
    assert msg_indexes["ix_messages_conv_created"]["column_names"] == ["conversation_id", "created_at", "id"]
    # Standalone index on conversation_id was dropped as redundant
    assert "ix_messages_conversation_id" not in msg_indexes
    assert "ix_messages_id" not in msg_indexes


def test_document_scope_invalidation_trigger_on_sqlite(tmp_path):
    """
    Verifies that deleting a document triggers scope invalidation:
    - Sets document_scope_invalidated = True (1)
    - Sets document_id = NULL via foreign key action
    - An explicitly cross-document conversation (document_id was NULL) is NOT invalidated.
    """
    db_file = tmp_path / "test_scope.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)

    # Enable SQLite foreign keys on all engine connections
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    upgrade_database(db_url=db_url, revision="head")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Create user
        user = User(
            email="scope_user@example.com",
            username="scope_user",
            hashed_password="pw_hash_test",
            is_active=True,
            is_admin=False,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # Create two documents
        doc1 = Document(
            id=str(uuid.uuid4()),
            owner_id=user.id,
            title="Doc1.pdf",
            filename="Doc1.pdf",
            storage_path="storage/documents/Doc1.pdf",
            file_size_bytes=1000,
            status="completed",
        )
        doc2 = Document(
            id=str(uuid.uuid4()),
            owner_id=user.id,
            title="Doc2.pdf",
            filename="Doc2.pdf",
            storage_path="storage/documents/Doc2.pdf",
            file_size_bytes=2000,
            status="completed",
        )
        session.add_all([doc1, doc2])
        session.commit()

        # Conversation 1: Scoped to Doc 1
        conv_doc1 = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            document_id=doc1.id,
            title="Doc1 Chat",
        )
        # Conversation 2: Scoped to Doc 2
        conv_doc2 = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            document_id=doc2.id,
            title="Doc2 Chat",
        )
        # Conversation 3: Cross-document (no document scope initially)
        conv_cross = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            document_id=None,
            title="Cross Doc Chat",
        )
        session.add_all([conv_doc1, conv_doc2, conv_cross])
        session.commit()

        # Verify initial states
        session.refresh(conv_doc1)
        session.refresh(conv_doc2)
        session.refresh(conv_cross)
        assert conv_doc1.document_scope_invalidated is False
        assert conv_doc2.document_scope_invalidated is False
        assert conv_cross.document_scope_invalidated is False

        # Delete Doc 1
        session.delete(doc1)
        session.commit()

        # Refresh conversations
        session.refresh(conv_doc1)
        session.refresh(conv_doc2)
        session.refresh(conv_cross)

        # conv_doc1 must now be marked invalidated, with document_id cleared
        assert conv_doc1.document_scope_invalidated is True, (
            "Deleting doc1 must trigger document_scope_invalidated = True on conv_doc1"
        )
        assert conv_doc1.document_id is None, "Foreign key ON DELETE SET NULL must clear document_id"

        # conv_doc2 must remain valid and bound to doc2
        assert conv_doc2.document_scope_invalidated is False
        assert conv_doc2.document_id == doc2.id

        # conv_cross must remain valid
        assert conv_cross.document_scope_invalidated is False
        assert conv_cross.document_id is None
    finally:
        session.close()


def test_orm_document_deletion_with_loaded_conversations_relationship(tmp_path):
    """
    Critical verification requested by user:
    SQLAlchemy ORM document deletion with the conversations relationship already loaded.

    Test procedure:
    1. Create a user, document, and document-scoped conversation.
    2. Also create an explicitly cross-document conversation (document_id is None).
    3. Load the document and explicitly access document.conversations in the SQLAlchemy session.
    4. Delete the document using session.delete(document).
    5. Commit the transaction.
    6. Query the conversations directly from the database.
    7. Verify for scoped conversation: document_id IS NULL and document_scope_invalidated IS TRUE.
    8. Verify for cross-document conversation: document_id IS NULL and document_scope_invalidated IS FALSE.
    """
    db_file = tmp_path / "test_orm_del.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    upgrade_database(db_url=db_url, revision="head")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # 1. Create user, document, and conversations
        user = User(
            email="orm_del@example.com",
            username="orm_del_user",
            hashed_password="pw_hash_test",
            is_active=True,
            is_admin=False,
        )
        session.add(user)
        session.commit()

        doc = Document(
            id=str(uuid.uuid4()),
            owner_id=user.id,
            title="ScopedDoc.pdf",
            filename="ScopedDoc.pdf",
            storage_path="storage/documents/ScopedDoc.pdf",
            file_size_bytes=500,
            status="completed",
        )
        session.add(doc)
        session.commit()

        scoped_conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            document_id=doc.id,
            title="Scoped Thread",
        )
        cross_conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            document_id=None,
            title="Cross Doc Thread",
        )
        session.add_all([scoped_conv, cross_conv])
        session.commit()

        # 2. Query document and EXPLICITLY load doc.conversations into session memory
        loaded_doc = session.query(Document).filter_by(id=doc.id).first()
        assert loaded_doc is not None
        # Access relationship to load into session identity map
        child_convs = loaded_doc.conversations
        assert len(child_convs) == 1
        assert child_convs[0].id == scoped_conv.id

        # 3. Delete document via session.delete()
        session.delete(loaded_doc)

        # 4. Commit transaction
        session.commit()

        scoped_id = scoped_conv.id
        cross_id = cross_conv.id

        # 5. Query conversations directly from database using fresh session
        session.close()
        fresh_session = SessionLocal()

        db_scoped_conv = fresh_session.query(Conversation).filter_by(id=scoped_id).first()
        assert db_scoped_conv is not None
        # 6. Verify scoped conversation has document_id IS NULL and document_scope_invalidated IS TRUE
        assert db_scoped_conv.document_id is None, "Document ID must be set to NULL via foreign key"
        assert db_scoped_conv.document_scope_invalidated is True, (
            "document_scope_invalidated must be TRUE after ORM deletion with loaded relationship"
        )

        # 7. Verify cross-document conversation remains unaffected
        db_cross_conv = fresh_session.query(Conversation).filter_by(id=cross_id).first()
        assert db_cross_conv is not None
        assert db_cross_conv.document_id is None
        assert db_cross_conv.document_scope_invalidated is False, (
            "Cross-document conversation must remain document_scope_invalidated=FALSE"
        )
        fresh_session.close()
    finally:
        session.close()


def test_live_postgresql_orm_document_deletion_with_loaded_conversations():
    """
    Verifies the exact 7-step ORM deletion procedure against the live PostgreSQL 18.6 database.
    """
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://documind:documind_password@localhost:5432/documind_db",
    )
    if "postgres" not in db_url:
        pytest.skip("Skipping PostgreSQL test: DATABASE_URL is not PostgreSQL")

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    test_uid = 99998
    test_doc_id = str(uuid.uuid4())
    test_scoped_cid = str(uuid.uuid4())
    test_cross_cid = str(uuid.uuid4())

    try:
        # Setup
        user = User(
            id=test_uid,
            email="live_orm@example.com",
            username="live_orm_u",
            hashed_password="pw",
            is_active=True,
            is_admin=False,
        )
        session.add(user)
        session.commit()

        doc = Document(
            id=test_doc_id,
            owner_id=user.id,
            title="LiveORM.pdf",
            filename="LiveORM.pdf",
            storage_path="storage/LiveORM.pdf",
            file_size_bytes=100,
            status="completed",
        )
        session.add(doc)
        session.commit()

        scoped_conv = Conversation(
            id=test_scoped_cid,
            user_id=user.id,
            document_id=doc.id,
            title="Live Scoped Thread",
        )
        cross_conv = Conversation(
            id=test_cross_cid,
            user_id=user.id,
            document_id=None,
            title="Live Cross Thread",
        )
        session.add_all([scoped_conv, cross_conv])
        session.commit()

        # Step 2: Load document and explicitly access document.conversations
        loaded_doc = session.query(Document).filter_by(id=test_doc_id).first()
        assert loaded_doc is not None
        assert len(loaded_doc.conversations) == 1
        assert loaded_doc.conversations[0].id == test_scoped_cid

        # Step 3: Delete using session.delete(document)
        session.delete(loaded_doc)

        # Step 4: Commit transaction
        session.commit()

        # Step 5 & 6: Verify in PostgreSQL
        session.close()
        verify_session = SessionLocal()

        c_scoped = verify_session.query(Conversation).filter_by(id=test_scoped_cid).first()
        assert c_scoped is not None
        assert c_scoped.document_id is None, "PostgreSQL foreign key must set document_id to NULL"
        assert c_scoped.document_scope_invalidated is True, (
            "PostgreSQL trigger must set document_scope_invalidated to TRUE on ORM delete"
        )

        # Step 7: Verify cross-document conversation
        c_cross = verify_session.query(Conversation).filter_by(id=test_cross_cid).first()
        assert c_cross is not None
        assert c_cross.document_id is None
        assert c_cross.document_scope_invalidated is False, (
            "Cross-document conversation invalidation flag must remain FALSE"
        )
        verify_session.close()

    finally:
        # Cleanup
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": test_uid})
            conn.commit()
        session.close()
        engine.dispose()


def test_cascade_delete_behavior_with_0003(tmp_path):
    """Verifies user cascade-deletion cleanly removes conversations and messages."""
    db_file = tmp_path / "test_cascade.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    upgrade_database(db_url=db_url, revision="head")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        user = User(
            email="cascade_u@example.com",
            username="cascade_u",
            hashed_password="pw",
            is_active=True,
            is_admin=False,
        )
        session.add(user)
        session.commit()

        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Thread 1",
        )
        session.add(conv)
        session.commit()

        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="user",
            content="Hello",
            status="completed",
        )
        session.add(msg)
        session.commit()

        assert session.query(Conversation).count() == 1
        assert session.query(Message).count() == 1

        session.delete(user)
        session.commit()

        assert session.query(Conversation).count() == 0
        assert session.query(Message).count() == 0
    finally:
        session.close()


def test_live_postgresql_trigger_and_scope_invalidation():
    """
    Verifies that the live PostgreSQL 18.6 database has the trigger installed,
    and deleting a document correctly executes scope invalidation.
    """
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://documind:documind_password@localhost:5432/documind_db",
    )
    if "postgres" not in db_url:
        pytest.skip("Skipping PostgreSQL trigger test: DATABASE_URL is not PostgreSQL")

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            # 1. Verify trigger function exists
            fn = conn.execute(
                text("SELECT proname FROM pg_proc WHERE proname = 'trg_fn_invalidate_conversation_scope'")
            ).fetchone()
            assert fn is not None, "trg_fn_invalidate_conversation_scope must exist in PostgreSQL"

            # 2. Verify trigger exists on documents table
            trg = conn.execute(
                text("SELECT tgname FROM pg_trigger WHERE tgname = 'trg_documents_scope_invalidation'")
            ).fetchone()
            assert trg is not None, "trg_documents_scope_invalidation must exist on documents table"

            # 3. Verify index order on ix_conversations_user_updated
            idx_def = conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_conversations_user_updated'")
            ).fetchone()
            assert idx_def is not None
            assert "DESC" in idx_def[0], f"ix_conversations_user_updated must be descending: {idx_def[0]}"

            # 4. Verify standalone index on messages.conversation_id is dropped
            st_idx = conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE indexname = 'ix_messages_conversation_id'")
            ).fetchone()
            assert st_idx is None, "ix_messages_conversation_id must be removed"

            # 5. Live PostgreSQL trigger behavior verification
            test_uid = 99999
            test_doc_id = str(uuid.uuid4())
            test_conv_id = str(uuid.uuid4())
            try:
                # Insert test user
                conn.execute(
                    text(
                        "INSERT INTO users (id, email, username, hashed_password, is_active, is_admin, created_at) "
                        "VALUES (:uid, 'live_pg_trg@example.com', 'live_pg_trg', 'hash', true, false, NOW())"
                    ),
                    {"uid": test_uid},
                )
                # Insert test document
                conn.execute(
                    text(
                        "INSERT INTO documents (id, owner_id, title, filename, storage_path, file_size_bytes, page_count, content_type, status, created_at) "
                        "VALUES (:did, :uid, 'TriggerDoc.pdf', 'TriggerDoc.pdf', 'storage/TriggerDoc.pdf', 100, 1, 'application/pdf', 'completed', NOW())"
                    ),
                    {"did": test_doc_id, "uid": test_uid},
                )
                # Insert test conversation scoped to document
                conn.execute(
                    text(
                        "INSERT INTO conversations (id, user_id, document_id, title, document_scope_invalidated, created_at, updated_at) "
                        "VALUES (:cid, :uid, :did, 'PG Trigger Thread', false, NOW(), NOW())"
                    ),
                    {"cid": test_conv_id, "uid": test_uid, "did": test_doc_id},
                )
                conn.commit()

                # Verify conversation initial state in PostgreSQL
                row = conn.execute(
                    text("SELECT document_id, document_scope_invalidated FROM conversations WHERE id = :cid"),
                    {"cid": test_conv_id},
                ).fetchone()
                assert row[0] == test_doc_id
                assert row[1] is False

                # Delete document from live PostgreSQL
                conn.execute(
                    text("DELETE FROM documents WHERE id = :did"),
                    {"did": test_doc_id},
                )
                conn.commit()

                # Verify trigger executed in PostgreSQL: document_scope_invalidated must be True and document_id must be NULL
                row_after = conn.execute(
                    text("SELECT document_id, document_scope_invalidated FROM conversations WHERE id = :cid"),
                    {"cid": test_conv_id},
                ).fetchone()
                assert row_after is not None
                assert row_after[0] is None, "PostgreSQL foreign key must set document_id to NULL"
                assert row_after[1] is True, "PostgreSQL trigger must set document_scope_invalidated to TRUE"
            finally:
                conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": test_uid})
                conn.commit()
    finally:
        engine.dispose()
