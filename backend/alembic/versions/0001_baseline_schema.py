"""Baseline database schema with pgvector support.

Revision ID: 0001_baseline_schema
Revises:
Create Date: 2026-10-31 23:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '0001_baseline_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name != "sqlite"

    # 1. Enable pgvector extension on PostgreSQL
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true' if is_postgres else '1')),
        sa.Column('is_admin', sa.Boolean(), nullable=False, server_default=sa.text('false' if is_postgres else '0')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_users_id', 'users', ['id'], unique=False)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    # 3. Documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=True),
        sa.Column('storage_path', sa.String(length=512), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('content_type', sa.String(length=64), nullable=False, server_default=sa.text("'application/pdf'")),
        sa.Column('status', sa.String(length=32), nullable=False, server_default=sa.text("'queued'")),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_documents_id', 'documents', ['id'], unique=False)
    op.create_index('ix_documents_owner_id', 'documents', ['owner_id'], unique=False)
    op.create_index('ix_documents_file_hash', 'documents', ['file_hash'], unique=False)
    op.create_index('ix_documents_owner_created', 'documents', ['owner_id', 'created_at'], unique=False)
    op.create_index('ix_documents_owner_hash', 'documents', ['owner_id', 'file_hash'], unique=False)

    # 4. Document Chunks table with 1536-dimensional vector embedding
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('embedding_space', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_document_chunks_id', 'document_chunks', ['id'], unique=False)
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'], unique=False)
    op.create_index('ix_chunks_doc_idx', 'document_chunks', ['document_id', 'chunk_index'], unique=False)
    op.create_index('ix_chunks_embedding_space', 'document_chunks', ['embedding_space'], unique=False)

    # Fast approximate nearest-neighbor HNSW index on PostgreSQL
    if is_postgres:
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
        )

    # 5. Request Metrics table
    op.create_table(
        'request_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('request_id', sa.String(length=36), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('document_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('retrieval_latency_ms', sa.Float(), nullable=False),
        sa.Column('generation_latency_ms', sa.Float(), nullable=False),
        sa.Column('total_latency_ms', sa.Float(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default=sa.text("'success'")),
        sa.Column('error_category', sa.String(length=64), nullable=True),
        sa.Column('safe_message', sa.String(length=255), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('pricing_version', sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_request_metrics_id', 'request_metrics', ['id'], unique=False)
    op.create_index('ix_request_metrics_request_id', 'request_metrics', ['request_id'], unique=True)
    op.create_index('ix_request_metrics_timestamp', 'request_metrics', ['timestamp'], unique=False)
    op.create_index('ix_request_metrics_time', 'request_metrics', ['timestamp'], unique=False)
    op.create_index('ix_request_metrics_status', 'request_metrics', ['status'], unique=False)
    op.create_index('ix_request_metrics_user_id', 'request_metrics', ['user_id'], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name != "sqlite"

    if is_postgres:
        op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw;")

    op.drop_table('request_metrics')
    op.drop_table('document_chunks')
    op.drop_table('documents')
    op.drop_table('users')
