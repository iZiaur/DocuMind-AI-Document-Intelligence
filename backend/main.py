import os
import math
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, Generator, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from backend.models import (
    Base,
    User,
    Document,
    DocumentChunk,
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    DocumentResponse,
    DocumentListResponse,
    SearchRequest,
    SearchResponse,
    SearchPassageResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from backend.rag import generate_rag_answer
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    security,
)
from backend.storage import save_pdf_securely, remove_pdf_from_storage
from backend.embeddings import get_embedding, EMBEDDING_MODEL, EMBEDDING_DIMENSION
from backend.ingestion import process_document_pipeline

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://documind:documind_password@localhost:5432/documind_db",
)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

# SQLAlchemy database engine (connection pool)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"connect_timeout": 2},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a clean database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_health() -> Dict[str, Any]:
    """Checks PostgreSQL connection and tests pgvector extension availability."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            vector_check = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()

            has_vector = vector_check is not None
            if not has_vector:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    has_vector = True
                except Exception:
                    pass

            return {
                "status": "connected",
                "pgvector_available": has_vector,
                "message": "Database connection established successfully",
            }
    except SQLAlchemyError as err:
        return {
            "status": "disconnected",
            "pgvector_available": False,
            "message": f"Database unreachable: {str(err.__class__.__name__)}",
        }
    except Exception as err:
        return {
            "status": "disconnected",
            "pgvector_available": False,
            "message": f"Database check failed: {str(err)}",
        }


def init_db():
    """Initializes tables and HNSW index in PostgreSQL schema."""
    try:
        Base.metadata.create_all(bind=engine)
        # Attempt to create HNSW index for high-speed cosine vector queries
        with engine.connect() as conn:
            try:
                conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);")
                )
                conn.commit()
            except Exception:
                pass
    except Exception as err:
        print(f"[DocuMind AI Backend] Table initialization deferred (DB offline): {err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    db_health = check_database_health()
    print(f"[DocuMind AI Backend] Startup check - Database: {db_health['status']}")
    if db_health["status"] == "connected":
        init_db()
    yield
    engine.dispose()
    print("[DocuMind AI Backend] Shutdown complete - Connection pool disposed")


app = FastAPI(
    title="DocuMind AI Backend",
    description="Enterprise Document Intelligence & RAG Platform API",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Authentication Security Dependency
# ---------------------------------------------------------------------------

def get_current_active_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Validates JWT Bearer token and returns authenticated User entity.
    Derives user identity strictly from token signature.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id_str: Optional[str] = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token user identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account associated with this token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ---------------------------------------------------------------------------
# Health & Base Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "DocuMind AI Backend",
        "version": "0.3.0",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "docs_url": "/docs",
        "health_url": "/api/health",
    }


@app.get("/health")
@app.get("/api/health")
def health_check():
    db_health = check_database_health()
    overall_status = "healthy" if db_health["status"] == "connected" else "degraded"
    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_health,
        "environment": {
            "service": "documind-api",
            "version": "0.3.0",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
    }


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(
        (User.email == request.email.lower().strip()) |
        (User.username == request.username.strip())
    ).first()

    if existing_user:
        if existing_user.email.lower() == request.email.lower().strip():
            detail = "An account with this email address already exists."
        else:
            detail = "This username is already taken. Please choose another."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    hashed = hash_password(request.password)

    new_user = User(
        email=request.email.lower().strip(),
        username=request.username.strip(),
        hashed_password=hashed,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(user_id=new_user.id, username=new_user.username)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(new_user),
    )


@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    identifier = request.username_or_email.strip()
    user = db.query(User).filter(
        (User.email == identifier.lower()) | (User.username == identifier)
    ).first()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user account has been disabled.",
        )

    access_token = create_access_token(user_id=user.id, username=user.username)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_active_user)):
    return UserResponse.model_validate(current_user)


# ---------------------------------------------------------------------------
# Document Management & Ingestion Routes
# ---------------------------------------------------------------------------

@app.post("/api/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Uploads, validates, and processes a PDF document through the ingestion pipeline:
    - Verifies %PDF- magic signature and structural integrity
    - Enforces 15MB file-size limit
    - Stores file under safe server-side UUID identifier
    - Checks for duplicate upload using SHA-256 content hash (idempotency)
    - Performs page-aware text extraction and chunking
    - Generates 1536-dimensional vector embeddings and stores in pgvector
    """
    file_bytes = await file.read()
    cleaned_title = title.strip() if title and title.strip() else file.filename or "Untitled Document"

    # Save to disk and obtain safe identifier + SHA-256 content hash
    doc_id, disk_path, page_count, file_hash = save_pdf_securely(
        file_bytes=file_bytes,
        original_filename=file.filename or "uploaded.pdf",
    )

    # Idempotency / Duplicate Detection: Check if user already uploaded identical file
    existing_doc = db.query(Document).filter(
        Document.owner_id == current_user.id,
        Document.file_hash == file_hash,
    ).first()

    if existing_doc and existing_doc.status == "completed":
        # Remove newly saved duplicate file to conserve disk
        remove_pdf_from_storage(disk_path)
        return DocumentResponse.model_validate(existing_doc)

    document = Document(
        id=doc_id,
        owner_id=current_user.id,
        title=cleaned_title,
        filename=os.path.basename(file.filename or "uploaded.pdf"),
        file_hash=file_hash,
        storage_path=disk_path,
        file_size_bytes=len(file_bytes),
        page_count=page_count,
        content_type="application/pdf",
        status="queued",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # Ingest document text, create chunks, and generate vector embeddings
    try:
        process_document_pipeline(document.id, db)
    except Exception as err:
        # Error details are captured in document.error_message and document.status="failed"
        pass

    db.refresh(document)
    return DocumentResponse.model_validate(document)


@app.post("/api/documents/{document_id}/process", response_model=DocumentResponse)
def retry_process_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Retries ingestion for a document.
    Enforces authorization: Returns 404 if document is not owned by the caller.
    """
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == current_user.id)
        .first()
    )

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        process_document_pipeline(document.id, db)
    except Exception as err:
        pass

    db.refresh(document)
    return DocumentResponse.model_validate(document)


@app.get("/api/documents", response_model=DocumentListResponse)
def list_user_documents(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    docs = (
        db.query(Document)
        .filter(Document.owner_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )

    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentResponse.model_validate(doc) for doc in docs],
    )


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
def get_document_details(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == current_user.id)
        .first()
    )

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return DocumentResponse.model_validate(document)


@app.get("/api/documents/{document_id}/download")
def download_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == current_user.id)
        .first()
    )

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if not os.path.exists(document.storage_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document file missing from server storage.")

    return FileResponse(
        path=document.storage_path,
        media_type="application/pdf",
        filename=document.filename,
        headers={"Content-Disposition": f"inline; filename=\"{document.filename}\""},
    )


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == current_user.id)
        .first()
    )

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    remove_pdf_from_storage(document.storage_path)
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully", "id": document_id}


# ---------------------------------------------------------------------------
# Retrieval & RAG Question-Answering Endpoints
# ---------------------------------------------------------------------------

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Helper for evaluating cosine similarity on SQLite test environments."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def retrieve_authorized_passages(
    query_text: str,
    user_id: int,
    top_k: int,
    document_id: Optional[str],
    db: Session,
) -> List[SearchPassageResponse]:
    """
    Core retrieval engine enforcing multi-tenant isolation at the database query level.
    Uses pgvector's <=> cosine distance operator on PostgreSQL and in-memory cosine fallback on SQLite.
    """
    if document_id:
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.owner_id == user_id,
        ).first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or access denied.",
            )

    query_vector = get_embedding(query_text)
    passages: List[SearchPassageResponse] = []
    is_sqlite = db.bind.dialect.name == "sqlite"

    if not is_sqlite:
        distance_expr = DocumentChunk.embedding.cosine_distance(query_vector)
        query = (
            db.query(
                DocumentChunk,
                Document.title.label("document_title"),
                (1 - distance_expr).label("similarity"),
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(Document.owner_id == user_id)
        )
        if document_id:
            query = query.filter(Document.id == document_id)

        results = query.order_by(distance_expr).limit(top_k).all()

        for chunk, doc_title, similarity in results:
            clamped_score = round(max(0.0, min(1.0, float(similarity))), 4)
            passages.append(
                SearchPassageResponse(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=doc_title,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=clamped_score,
                    metadata={
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dimension": EMBEDDING_DIMENSION,
                        "character_count": len(chunk.text),
                    },
                )
            )
    else:
        query = (
            db.query(DocumentChunk, Document.title.label("document_title"))
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(Document.owner_id == user_id)
        )
        if document_id:
            query = query.filter(Document.id == document_id)

        chunks_with_titles = query.all()
        scored_chunks = []

        for chunk, doc_title in chunks_with_titles:
            emb = list(chunk.embedding) if hasattr(chunk.embedding, "__iter__") else chunk.embedding
            sim = _cosine_similarity(query_vector, emb)
            scored_chunks.append((chunk, doc_title, sim))

        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        top_matches = scored_chunks[:top_k]

        for chunk, doc_title, sim in top_matches:
            clamped_score = round(max(0.0, min(1.0, float(sim))), 4)
            passages.append(
                SearchPassageResponse(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=doc_title,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=clamped_score,
                    metadata={
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dimension": EMBEDDING_DIMENSION,
                        "character_count": len(chunk.text),
                    },
                )
            )

    return passages


@app.post("/api/search", response_model=SearchResponse)
def semantic_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Semantic vector search across authorized documents."""
    passages = retrieve_authorized_passages(
        query_text=request.query,
        user_id=current_user.id,
        top_k=request.top_k,
        document_id=request.document_id,
        db=db,
    )
    return SearchResponse(
        query=request.query,
        total_results=len(passages),
        passages=passages,
    )


@app.post("/api/rag/query", response_model=RAGQueryResponse)
def query_rag(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    RAG Question-Answering Endpoint:
    1. Authenticates user and enforces document permissions in database query.
    2. Retrieves relevant evidence passages from authorized documents.
    3. Synthesizes a grounded answer using the configured LLM with prompt-injection defense.
    4. Validates citations against the passages actually retrieved.
    5. Returns structured response with answer, citations, page citations, and request ID.
    """
    passages = retrieve_authorized_passages(
        query_text=request.question,
        user_id=current_user.id,
        top_k=request.top_k,
        document_id=request.document_id,
        db=db,
    )

    response = generate_rag_answer(
        question=request.question,
        passages=passages,
    )

    return response
