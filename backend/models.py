import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

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
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


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
    embedding = Column(Vector(1536), nullable=False)  # OpenAI text-embedding-3-small (1536 dim)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_idx", "document_id", "chunk_index"),
    )


# ---------------------------------------------------------------------------
# Pydantic Schemas for Validation and API Contracts
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
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
    question: str = Field(..., min_length=2, max_length=1500)
    document_id: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    insufficient_evidence: bool
    citations: List[Citation]
    retrieval_count: int
    model_used: str
