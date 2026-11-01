import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    desc,
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
from backend.embeddings import get_embedding_space_id

Base = declarative_base()


class User(Base):
    """
    User entity representing an authenticated account.
    Passwords are stored strictly as salted bcrypt hashes.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    request_metrics = relationship("RequestMetric", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    """
    Document entity representing an uploaded PDF document.
    Enforces foreign key ownership to the uploading user for document-level authorization.
    Tracks ingestion lifecycle: queued -> processing -> completed / failed.
    """
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), index=True, nullable=True)  # SHA-256 content hash for idempotency
    storage_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    page_count = Column(Integer, default=0, nullable=False)
    content_type = Column(String(64), default="application/pdf", nullable=False)
    status = Column(String(32), default="queued", nullable=False)  # "queued", "processing", "completed", "failed"
    error_message = Column(Text, nullable=True)  # Detailed reason if extraction/parsing failed
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="document", passive_deletes="all")

    __table_args__ = (
        Index("ix_documents_owner_created", "owner_id", "created_at"),
        Index("ix_documents_owner_hash", "owner_id", "file_hash"),
    )


class DocumentChunk(Base):
    """
    DocumentChunk entity storing page-aware text passages and 1536-dimensional vector embeddings.
    Enforces foreign key ownership back to the parent Document.
    """
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)  # 1-indexed page for accurate citation
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)  # 1536-dimensional embedding
    embedding_space = Column(String(64), nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")

    def __init__(self, **kwargs):
        if "embedding_space" not in kwargs:
            kwargs["embedding_space"] = get_embedding_space_id()
        super().__init__(**kwargs)

    __table_args__ = (
        Index("ix_chunks_doc_idx", "document_id", "chunk_index"),
        Index("ix_chunks_embedding_space", "embedding_space"),
    )


class RequestMetric(Base):
    """
    RequestMetric entity storing lightweight production observability records.
    Tracks lifecycle latency, token usage, estimated costs, and failures per document query.
    Sensitive document text, full prompts, stack traces, and API keys are strictly excluded.
    """
    __tablename__ = "request_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    request_id = Column(String(36), unique=True, index=True, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    document_count = Column(Integer, default=0, nullable=False)
    retrieval_latency_ms = Column(Float, nullable=False)
    generation_latency_ms = Column(Float, nullable=False)
    total_latency_ms = Column(Float, nullable=False)
    status = Column(String(16), default="success", nullable=False)  # "success" or "failed"
    error_category = Column(String(64), nullable=True)
    safe_message = Column(String(255), nullable=True)

    # Token usage (None if unknown / unavailable)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # Cost tracking (None if tokens unknown)
    estimated_cost_usd = Column(Float, nullable=True)
    pricing_version = Column(String(32), nullable=True)

    user = relationship("User", back_populates="request_metrics")

    __table_args__ = (
        Index("ix_request_metrics_time", "timestamp"),
        Index("ix_request_metrics_status", "status"),
    )


class Conversation(Base):
    """
    Conversation entity representing a persistent multi-turn chat thread.
    Enforces foreign key ownership to the authenticated User for tenant isolation.
    Optionally scoped to a specific Document (cross-document if document_id is NULL).
    Tracks document_scope_invalidated to prevent silent broadening of retrieval
    if the scoped document is deleted.
    """
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(255), default="New Conversation", nullable=False)
    document_scope_invalidated = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="conversations")
    document = relationship("Document", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="[Message.created_at, Message.id]",
    )

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", desc("updated_at")),
    )


class Message(Base):
    """
    Message entity representing a single turn within a conversation thread.
    Belongs strictly to a parent Conversation.
    Stores original question, condensed retrieval query, grounded answer, citations, and status.
    """
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # conversation_id is indexed as the leftmost prefix of ix_messages_conv_created (no redundant index)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)  # Original user question or assistant answer
    condensed_query = Column(Text, nullable=True)  # Reformulated retrieval query (used for vector search)
    status = Column(String(16), default="completed", nullable=False)  # "pending", "completed", "failed"
    error_message = Column(Text, nullable=True)  # Detailed failure reason if generation failed
    citations = Column(JSON, nullable=True)  # Serialized list of citation dicts
    token_usage = Column(JSON, nullable=True)  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    estimated_cost_usd = Column(Float, nullable=True)
    pricing_version = Column(String(32), nullable=True)
    retrieval_latency_ms = Column(Float, nullable=True)
    generation_latency_ms = Column(Float, nullable=True)
    total_latency_ms = Column(Float, nullable=True)
    model_used = Column(String(64), nullable=True)
    insufficient_evidence = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conv_created", "conversation_id", "created_at", "id"),
    )


# ---------------------------------------------------------------------------
# Pydantic Schemas for Validation and API Contracts
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def strip_username(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class UserLoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
    is_admin: bool = False
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    filename: str
    file_size_bytes: int
    page_count: int
    content_type: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]


# ---------------------------------------------------------------------------
# Search Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = None


class SearchPassageResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    total_results: int
    passages: List[SearchPassageResponse]


# ---------------------------------------------------------------------------
# RAG Question-Answering Schemas
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    excerpt: str
    score: float


class RAGQueryRequest(BaseModel):
    question: str = Field(...)
    document_id: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Question must be a string.")
        trimmed = v.strip()
        if len(trimmed) < 2:
            raise ValueError("Question must contain at least 2 non-whitespace characters.")
        if len(trimmed) > 1500:
            raise ValueError("Question cannot exceed 1500 characters after trimming.")
        return trimmed


class RAGQueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    insufficient_evidence: bool
    citations: List[Citation]
    retrieval_count: int
    model_used: str
    token_usage: Optional[Dict[str, Any]] = None
    estimated_cost_usd: Optional[float] = None
    pricing_version: Optional[str] = None
    retrieval_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Observability and Monitoring Schemas
# ---------------------------------------------------------------------------

class RequestMetricItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    timestamp: datetime
    user_id: Optional[int] = None
    document_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    status: str
    error_category: Optional[str] = None
    safe_message: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    pricing_version: Optional[str] = None


class MonitoringDashboardResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    avg_total_latency_ms: float
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    p95_total_latency_ms: float
    token_usage: Dict[str, Any]
    total_estimated_cost_usd: Optional[float] = None
    cost_is_partial: bool = False
    known_cost_requests: int = 0
    unknown_cost_requests: int = 0
    pricing_version: str
    recent_requests: List[RequestMetricItemResponse]


# ---------------------------------------------------------------------------
# Conversation & Multi-Turn Message Schemas (Checkpoint 13)
# ---------------------------------------------------------------------------

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    condensed_query: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    citations: Optional[List[Citation]] = None
    token_usage: Optional[Dict[str, Any]] = None
    estimated_cost_usd: Optional[float] = None
    pricing_version: Optional[str] = None
    retrieval_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    insufficient_evidence: bool = False
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: int
    document_id: Optional[str] = None
    title: str
    document_scope_invalidated: bool = False
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None
    latest_message: Optional[MessageResponse] = None


class ConversationListResponse(BaseModel):
    total: int
    conversations: List[ConversationResponse]


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    document_id: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    document_id: Optional[str] = None
    reset_document_scope: Optional[bool] = None


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=1500)
    top_k: int = Field(default=4, ge=1, le=10)
