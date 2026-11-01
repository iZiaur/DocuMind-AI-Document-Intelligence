"""Add document_scope_invalidated flag, trigger, and index optimizations.

Revision ID: 0003_conversation_scope_and_index_fixes
Revises: 0002_conversations_and_messages
Create Date: 2026-11-01 00:56:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003_scope_and_index_fixes'
down_revision: Union[str, None] = '0002_conversations_and_messages'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name != "sqlite"

    # 1. Add document_scope_invalidated column to conversations
    op.add_column(
        'conversations',
        sa.Column(
            'document_scope_invalidated',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false' if is_postgres else '0'),
        ),
    )

    # 2. Remove redundant standalone index on messages.conversation_id
    # (ix_messages_conv_created already starts with conversation_id)
    op.drop_index('ix_messages_conversation_id', table_name='messages')

    # 3. Update ix_messages_conv_created to include 'id' for stable tie-breaking
    op.drop_index('ix_messages_conv_created', table_name='messages')
    op.create_index(
        'ix_messages_conv_created',
        'messages',
        ['conversation_id', 'created_at', 'id'],
        unique=False,
    )

    # 4. Update ix_conversations_user_updated to order by updated_at DESC
    op.drop_index('ix_conversations_user_updated', table_name='conversations')
    op.execute(
        "CREATE INDEX ix_conversations_user_updated ON conversations (user_id, updated_at DESC);"
    )

    # 5. Database trigger to mark document_scope_invalidated BEFORE document deletion
    if is_postgres:
        op.execute(
            """
            CREATE OR REPLACE FUNCTION trg_fn_invalidate_conversation_scope()
            RETURNS TRIGGER AS $$
            BEGIN
                UPDATE conversations
                SET document_scope_invalidated = TRUE
                WHERE document_id = OLD.id;
                RETURN OLD;
            END;
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS trg_documents_scope_invalidation ON documents;
            CREATE TRIGGER trg_documents_scope_invalidation
            BEFORE DELETE ON documents
            FOR EACH ROW
            EXECUTE FUNCTION trg_fn_invalidate_conversation_scope();
            """
        )
    else:
        op.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_documents_scope_invalidation
            BEFORE DELETE ON documents
            BEGIN
                UPDATE conversations
                SET document_scope_invalidated = 1
                WHERE document_id = OLD.id;
            END;
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name != "sqlite"

    # 1. Drop trigger and trigger function
    if is_postgres:
        op.execute("DROP TRIGGER IF EXISTS trg_documents_scope_invalidation ON documents;")
        op.execute("DROP FUNCTION IF EXISTS trg_fn_invalidate_conversation_scope();")
    else:
        op.execute("DROP TRIGGER IF EXISTS trg_documents_scope_invalidation;")

    # 2. Revert ix_conversations_user_updated to ASC
    op.drop_index('ix_conversations_user_updated', table_name='conversations')
    op.create_index(
        'ix_conversations_user_updated',
        'conversations',
        ['user_id', 'updated_at'],
        unique=False,
    )

    # 3. Revert ix_messages_conv_created without 'id'
    op.drop_index('ix_messages_conv_created', table_name='messages')
    op.create_index(
        'ix_messages_conv_created',
        'messages',
        ['conversation_id', 'created_at'],
        unique=False,
    )

    # 4. Recreate standalone index on messages.conversation_id
    op.create_index('ix_messages_conversation_id', 'messages', ['conversation_id'], unique=False)

    # 5. Drop document_scope_invalidated column
    op.drop_column('conversations', 'document_scope_invalidated')
