import os
import math
import asyncio
import logging
import threading
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager
from typing import Dict, Any, Generator, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.exc import SQLAlchemyError

from backend.models import (
    Base,
    User,
    Document,
    DocumentChunk,
    Conversation,
    Message,
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    SendMessageRequest,
    MessageResponse,
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
    RequestMetric,
    MonitoringDashboardResponse,
    RequestMetricItemResponse,
)
from backend.rag import (
    generate_rag_answer,
    validate_generation_provider_configuration,
    _cosine_similarity,
    retrieve_authorized_passages,
    condense_conversational_query,
    execute_rag_pipeline,
)
from backend.observability import (
    categorize_error,
    log_request_event,
    record_request_metric,
    get_monitoring_dashboard_summary,
    PRICING_CONFIG_VERSION,
)
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    validate_jwt_secret,
    security,
)
from backend.storage import save_pdf_securely, remove_pdf_from_storage, stream_and_validate_upload
from backend.embeddings import (
    get_embedding,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    get_embedding_space_id,
    IncompatibleEmbeddingSpaceError,
    validate_embedding_provider_configuration,
)
from backend.rate_limiter import check_login_rate_limit, check_rag_rate_limit, rate_limiter
from backend.ingestion import process_document_pipeline
from backend.bootstrap import provision_admin_account, is_username_reserved
from backend.retrieval import compute_lexical_score, compute_hybrid_score

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


def migrate_schema(bind_engine):
    """
    Safely and idempotently adds missing columns (such as users.is_admin,
    and document_chunks.embedding_space) to existing databases created in earlier
    stages without data loss.
    Supports both SQLite and PostgreSQL engines.
    """
    dialect = bind_engine.dialect.name
    with bind_engine.connect() as conn:
        if dialect == "sqlite":
            table_info = conn.execute(text("PRAGMA table_info(users);")).fetchall()
            if table_info:
                cols = [row[1] for row in table_info]
                if "is_admin" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0;"))
                    conn.commit()
            chunk_info = conn.execute(text("PRAGMA table_info(document_chunks);")).fetchall()
            if chunk_info:
                chunk_cols = [row[1] for row in chunk_info]
                if "embedding_space" not in chunk_cols:
                    conn.execute(text("ALTER TABLE document_chunks ADD COLUMN embedding_space VARCHAR(64);"))
                    conn.commit()
        else:
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_space VARCHAR(64);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_embedding_space ON document_chunks (embedding_space);"))
                conn.commit()
            except Exception:
                conn.rollback()


def bootstrap_admin_user(
    db: Session,
    username: str,
    secret: Optional[str] = None,
    email: Optional[str] = None,
    force_promote: bool = False,
) -> bool:
    """
    Controlled administrative provisioning function.
    Delegates to provision_admin_account.
    Returns True if successfully provisioned or promoted, False otherwise.
    """
    success, _ = provision_admin_account(
        db=db,
        username=username,
        secret=secret,
        email=email,
        force_promote=force_promote,
    )
    return success


def init_db(target_engine=None):
    """
    Initializes database schema and verifies migration status (Phase E / Checkpoint 10).
    - In production: avoids uncontrolled DDL creation; verifies Alembic revision status.
    - In development / tests / SQLite: ensures tables exist and applies local schema migrations.
    """
    db_engine = target_engine or engine
    try:
        from backend.migrations import check_migration_status
        from backend.embeddings import is_production_mode

        # 1. Schema initialization / verification
        if is_production_mode() and db_engine.dialect.name != "sqlite":
            # Production: Verify Alembic migration status rather than executing uncontrolled DDL
            status = check_migration_status(db_engine)
            if not status["is_up_to_date"]:
                print(
                    f"[DocuMind AI Backend] Notice: Database schema migration status in production: "
                    f"current='{status['current_revision']}', head='{status['head_revision']}'. "
                    f"Run 'alembic upgrade head' as a distinct deployment pipeline step."
                )
        else:
            # Development / Test / SQLite environments: Ensure defined tables exist
            Base.metadata.create_all(bind=db_engine)
            migrate_schema(db_engine)

        # 3. Check for environment-controlled admin bootstrap
        bootstrap_admin = os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_USERNAME", "").strip()
        bootstrap_secret = os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_SECRET", "").strip()
        bootstrap_email = os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_EMAIL", "").strip() or None

        if bootstrap_admin:
            if not bootstrap_secret:
                print(
                    f"[DocuMind AI Admin Bootstrap] Notice: DOCUMIND_ADMIN_BOOTSTRAP_USERNAME is configured ('{bootstrap_admin}'), "
                    "but DOCUMIND_ADMIN_BOOTSTRAP_SECRET is not provided. Admin bootstrap skipped."
                )
            else:
                SessionMaker = sessionmaker(bind=db_engine)
                with SessionMaker() as session:
                    success, message = provision_admin_account(
                        db=session,
                        username=bootstrap_admin,
                        secret=bootstrap_secret,
                        email=bootstrap_email,
                        force_promote=False,  # Automatic startup NEVER automatically elevates existing unprivileged accounts
                    )
                    print(f"[DocuMind AI Admin Bootstrap] {message}")

        # 4. Attempt to create HNSW index for high-speed cosine vector queries on PostgreSQL
        if db_engine.dialect.name != "sqlite":
            with db_engine.connect() as conn:
                try:
                    conn.execute(
                        text("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);")
                    )
                    conn.commit()
                except Exception:
                    pass
    except Exception as err:
        print(f"[DocuMind AI Backend] Table initialization deferred (DB offline): {err}")


def validate_provider_configuration() -> None:
    """
    Validates embedding and generation provider configurations at startup (SEC-05).
    Guarantees that production environments fail fast on startup if providers
    are missing, set to offline, or missing required credentials.
    """
    validate_embedding_provider_configuration()
    validate_generation_provider_configuration()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # 1. Enforce strict JWT security configuration upfront (SEC-01)
    validate_jwt_secret()

    # 2. Enforce strict provider configurations in production (SEC-05)
    validate_provider_configuration()

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


def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Ensures the authenticated user has administrator privileges.
    Enforces role-based access control based strictly on the persisted is_admin role.
    """
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required to access monitoring dashboard.",
        )
    return current_user


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
    clean_username = request.username.strip()
    clean_email = request.email.lower().strip()

    # Prevent public squatting on reserved administrative usernames
    if is_username_reserved(clean_username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This username is reserved for system administration. Please choose another username.",
        )

    existing_user = db.query(User).filter(
        (User.email == clean_email) |
        (User.username == clean_username)
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
        is_admin=False,  # Public registration NEVER grants admin privileges
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
def login(request: UserLoginRequest, http_request: Request, db: Session = Depends(get_db)):
    # Enforce anti-credential-stuffing IP-based rate limiting (SEC-04)
    check_login_rate_limit(http_request)

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

def _run_ingestion_in_worker(document_id: str, bind_engine) -> None:
    """
    Executes synchronous document ingestion inside an isolated worker thread (REL-02).
    Uses its own dedicated database session bound to the active engine to guarantee thread safety.
    """
    WorkerSession = sessionmaker(bind=bind_engine)
    with WorkerSession() as worker_db:
        process_document_pipeline(document_id, worker_db)


@app.post("/api/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Uploads, validates, and processes a PDF document through the ingestion pipeline:
    - Verifies %PDF- magic signature and structural integrity
    - Enforces 10MB file-size limit (SEC-03)
    - Incrementally streams chunked upload to bounded disk temporary file
    - Stores file under safe server-side UUID identifier
    - Checks for duplicate upload using SHA-256 content hash (idempotency)
    - Performs page-aware text extraction and chunking
    - Generates 1536-dimensional vector embeddings and stores in pgvector
    """
    cleaned_title = title.strip() if title and title.strip() else file.filename or "Untitled Document"

    # Save to disk and obtain safe identifier + SHA-256 content hash (SEC-03: Incremental streaming upload)
    content_length = request.headers.get("content-length")
    doc_id, disk_path, page_count, file_hash, file_size_bytes = await stream_and_validate_upload(
        file,
        content_length=content_length,
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
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        content_type="application/pdf",
        status="queued",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # Ingest document text, create chunks, and generate vector embeddings
    # REL-02: Run CPU and network intensive ingestion in a worker thread so the main asyncio event loop is never blocked
    try:
        await asyncio.to_thread(_run_ingestion_in_worker, document.id, db.bind)
    except Exception as err:
        # Error details are captured in document.error_message and document.status="failed"
        pass

    db.refresh(document)
    return DocumentResponse.model_validate(document)


@app.post("/api/documents/{document_id}/process", response_model=DocumentResponse)
async def retry_process_document(
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
        await asyncio.to_thread(_run_ingestion_in_worker, document.id, db.bind)
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
    RAG Question-Answering Endpoint with Request Lifecycle Observability:
    Reuses the unified execute_rag_pipeline() shared service.
    """
    return execute_rag_pipeline(
        question=request.question,
        user_id=current_user.id,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
        enforce_rate_limit=True,
        generator_fn=generate_rag_answer,
        retriever_fn=retrieve_authorized_passages,
    )


@app.get("/api/monitoring/dashboard", response_model=MonitoringDashboardResponse)
def get_monitoring_dashboard(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """
    Admin-only dashboard providing real-time in-app observability metrics:
    total requests, average and P95 latency, error rates, token usage,
    estimated API costs, and recent request performance logs.
    """
    summary = get_monitoring_dashboard_summary(db)
    return summary


# ---------------------------------------------------------------------------
# Multi-Turn Conversation Endpoints (Checkpoint 13 - Phase 4)
# ---------------------------------------------------------------------------

@app.post("/api/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Creates a new persistent conversation thread.
    Optionally scopes to a specific document owned by the authenticated user.
    If payload.document_id is provided, validates that the document exists and belongs to current_user.
    """
    effective_document_id = None
    if payload.document_id:
        doc = db.query(Document).filter(
            Document.id == payload.document_id,
            Document.owner_id == current_user.id,
        ).first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or access denied.",
            )
        effective_document_id = payload.document_id

    conv_title = (payload.title.strip() if payload.title and payload.title.strip() else "New Conversation")[:255]

    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        document_id=effective_document_id,
        title=conv_title,
        document_scope_invalidated=False,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return ConversationResponse.model_validate(conversation)


@app.get("/api/conversations", response_model=ConversationListResponse)
def list_conversations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Lists conversations for the authenticated user, ordered most recently updated first.
    Populates latest_message for sidebar previews.
    """
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .all()
    )

    result = []
    for conv in conversations:
        conv_resp = ConversationResponse.model_validate(conv)
        latest_msg = (
            db.query(Message)
            .filter(Message.conversation_id == conv.id, Message.status == "completed")
            .order_by(Message.created_at.desc(), Message.id.desc())
            .first()
        )
        if latest_msg:
            conv_resp.latest_message = MessageResponse.model_validate(latest_msg)
        result.append(conv_resp)

    return ConversationListResponse(total=len(result), conversations=result)


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Returns conversation metadata and its chronological message history.
    Enforces tenant isolation (HTTP 404 for missing or cross-tenant access).
    """
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

    conv_resp = ConversationResponse.model_validate(conv)
    conv_resp.messages = [MessageResponse.model_validate(m) for m in messages]
    if messages:
        conv_resp.latest_message = MessageResponse.model_validate(messages[-1])
    return conv_resp


DEFAULT_HEARTBEAT_INTERVAL: float = 25.0
DEFAULT_LEASE_EXTEND_SECONDS: int = 90
DEFAULT_INITIAL_LEASE_SECONDS: int = 90


def renew_generation_lease(
    db: Session,
    message_id: str,
    generation_token: str,
    extend_seconds: int = 90,
) -> bool:
    """
    Renews the generation lease for an active in-flight assistant message.
    Requires matching generation_token and status='pending'.
    Returns True if lease was successfully extended, False if ownership was lost or revoked.
    """
    record = (
        db.query(Message)
        .filter(Message.id == message_id, Message.status == "pending")
        .with_for_update()
        .first()
    )
    if not record:
        return False
    meta = record.token_usage if isinstance(record.token_usage, dict) else {}
    if meta.get("generation_token") != generation_token:
        return False

    now_utc = datetime.now(timezone.utc)
    new_meta = dict(meta)
    new_meta["heartbeat_at"] = now_utc.isoformat()
    new_meta["lease_expires_at"] = (now_utc + timedelta(seconds=extend_seconds)).isoformat()
    record.token_usage = new_meta
    flag_modified(record, "token_usage")
    db.commit()
    return True


class LeaseHeartbeat:
    """
    Background daemon thread context manager that automatically renews the generation lease
    at periodic intervals while execute_rag_pipeline() and finalization run.
    Uses independent short-lived SQLAlchemy sessions from the existing application engine.
    """
    def __init__(
        self,
        bind_engine,
        message_id: str,
        generation_token: str,
        interval: Optional[float] = None,
        extend_seconds: Optional[int] = None,
    ):
        self.bind_engine = bind_engine
        self.message_id = message_id
        self.generation_token = generation_token
        self.interval = interval if interval is not None else DEFAULT_HEARTBEAT_INTERVAL
        self.extend_seconds = extend_seconds if extend_seconds is not None else DEFAULT_LEASE_EXTEND_SECONDS
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        FreshSession = (
            sessionmaker(bind=self.bind_engine, autocommit=False, autoflush=False)
            if self.bind_engine is not None
            else SessionLocal
        )
        while not self._stop_event.wait(timeout=self.interval):
            try:
                with FreshSession() as db:
                    success = renew_generation_lease(
                        db=db,
                        message_id=self.message_id,
                        generation_token=self.generation_token,
                        extend_seconds=self.extend_seconds,
                    )
                    if not success:
                        logger.warning(
                            "Lease heartbeat stopped: generation ownership was revoked or message is no longer pending (%s).",
                            self.message_id,
                        )
                        break
            except Exception as exc:
                logger.error(
                    "Transient error during generation lease heartbeat renewal for %s: %r",
                    self.message_id,
                    exc,
                )

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"lease-heartbeat-{self.message_id[:8]}",
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_alive(self) -> bool:
        return self._thread.is_alive() if self._thread else False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@app.post("/api/conversations/{conversation_id}/messages", response_model=MessageResponse)
def send_conversation_message(
    conversation_id: str,
    payload: SendMessageRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Submits a user message to a multi-turn conversation thread:
    1. Validates ownership and row-locks conversation (with_for_update) to prevent history races.
    2. Enforces scope validity (HTTP 409 if document_scope_invalidated is True).
    3. Safe recovery check on in-flight pending assistant message: verifies generation lease activity.
    4. Loads latest completed history (up to 3 exchanges / 6 messages) for contextual condensation.
    5. Persists the user message (completed) and pending assistant message (pending) with lease token.
    6. Condenses retrieval query using bounded history if multi-turn.
    7. Invokes execute_rag_pipeline(), respecting REL-01 connection release and single rate-limiting.
    8. Conditional finalization: ensures only the active lease owner can finalize the assistant record.
    """
    raw_question = payload.content.strip()
    if not raw_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    # 1. Row-lock conversation to serialize concurrent messages on the same thread
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .with_for_update()
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )

    # 2. Scope validation
    if conv.document_scope_invalidated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation document scope was invalidated because the document was deleted. Re-scoping is required.",
        )

    if conv.document_id:
        doc = db.query(Document).filter(
            Document.id == conv.document_id,
            Document.owner_id == current_user.id,
        ).first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Conversation document scope is invalid or referenced document was removed.",
            )

    # Prevent concurrent duplicate in-flight processing and enforce safe lease recovery
    pending_msg = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.status == "pending",
        )
        .first()
    )
    if pending_msg:
        now_utc = datetime.now(timezone.utc)
        meta = pending_msg.token_usage if isinstance(pending_msg.token_usage, dict) else {}
        lease_expires_str = meta.get("lease_expires_at")
        lease_is_active = False

        if lease_expires_str:
            try:
                lease_expires = datetime.fromisoformat(lease_expires_str)
                if lease_expires.tzinfo is None:
                    lease_expires = lease_expires.replace(tzinfo=timezone.utc)
                if lease_expires > now_utc:
                    lease_is_active = True
            except (ValueError, TypeError):
                pass
        else:
            # Fallback for legacy rows without lease_expires_at
            created_time = pending_msg.created_at
            if created_time.tzinfo is None:
                created_time = created_time.replace(tzinfo=timezone.utc)
            if (now_utc - created_time).total_seconds() <= 180.0:
                lease_is_active = True

        if lease_is_active:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A message is currently being processed for this conversation. Please wait for it to complete.",
            )
        else:
            # Safe recovery contract: lease expired or worker crashed. Revoke ownership and mark failed.
            revoked_token = meta.get("generation_token")
            pending_msg.status = "failed"
            pending_msg.error_message = "Generation lease expired and was revoked by recovery."
            pending_msg.token_usage = {
                "revoked_token": revoked_token,
                "revoked_at": now_utc.isoformat(),
            }
            flag_modified(pending_msg, "token_usage")
            db.commit()

    # 3. Pre-generate IDs, generation lease token, and load previous completed history
    user_msg_id = str(uuid.uuid4())
    assistant_msg_id = str(uuid.uuid4())
    generation_token = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)
    lease_expires_at = now_utc + timedelta(seconds=DEFAULT_INITIAL_LEASE_SECONDS)

    # Strictly load only previous completed messages; exclude current user_msg_id
    past_messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.status == "completed",
            Message.role.in_(["user", "assistant"]),
            Message.id != user_msg_id,
        )
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    history_window = past_messages[-6:] if len(past_messages) > 6 else past_messages
    history_tuples = [(m.role, m.content) for m in history_window]

    # 4. Contextual query condensation using strictly previous turns
    condensed_query = None
    if history_tuples:
        condensed_query = condense_conversational_query(
            question=raw_question,
            history=history_tuples,
        )

    # 5. Persist user message and pending assistant message with active generation lease
    user_msg = Message(
        id=user_msg_id,
        conversation_id=conversation_id,
        role="user",
        content=raw_question,
        condensed_query=condensed_query,
        status="completed",
    )
    assistant_msg = Message(
        id=assistant_msg_id,
        conversation_id=conversation_id,
        role="assistant",
        content="",
        status="pending",
        token_usage={
            "generation_token": generation_token,
            "lease_expires_at": lease_expires_at.isoformat(),
            "heartbeat_at": now_utc.isoformat(),
        },
    )

    db.add(user_msg)
    db.add(assistant_msg)

    # Update conversation title if default
    if conv.title == "New Conversation" or not conv.title:
        candidate_title = raw_question[:45].strip()
        conv.title = candidate_title + ("..." if len(raw_question) > 45 else "")
    conv.updated_at = datetime.now(timezone.utc)

    db.commit()

    # Save bind engine for fresh session creation after execute_rag_pipeline releases db
    bind_engine = db.get_bind() if hasattr(db, "get_bind") else getattr(db, "bind", None)

    # 6. Execute shared RAG pipeline with automatic background lease heartbeat
    heartbeat = LeaseHeartbeat(
        bind_engine=bind_engine,
        message_id=assistant_msg_id,
        generation_token=generation_token,
        interval=DEFAULT_HEARTBEAT_INTERVAL,
        extend_seconds=DEFAULT_LEASE_EXTEND_SECONDS,
    )
    heartbeat.start()

    rag_response = None
    pipeline_exception = None

    try:
        try:
            rag_response = execute_rag_pipeline(
                question=raw_question,
                user_id=current_user.id,
                document_id=conv.document_id,
                conversation_id=conv.id,
                retrieval_query=condensed_query,
                top_k=payload.top_k,
                db=db,
                enforce_rate_limit=True,
                generator_fn=generate_rag_answer,
                retriever_fn=retrieve_authorized_passages,
            )
        except Exception as exc:
            pipeline_exception = exc

        # 7. Post-generation conditional finalization using fresh DB session and lease token validation
        FreshSession = (
            sessionmaker(bind=bind_engine, autocommit=False, autoflush=False)
            if bind_engine is not None
            else SessionLocal
        )

        with FreshSession() as fresh_db:
            # Atomic conditional select: record MUST still be in 'pending' status
            ast_record = (
                fresh_db.query(Message)
                .filter(
                    Message.id == assistant_msg_id,
                    Message.status == "pending",
                )
                .with_for_update()
                .first()
            )

            if not ast_record:
                # Worker lost ownership: recovery revoked the lease or record was modified
                logger.warning(
                    "Worker lost generation ownership for message %s (not pending). Finalization rejected.",
                    assistant_msg_id,
                )
                if pipeline_exception is not None and isinstance(pipeline_exception, HTTPException):
                    raise pipeline_exception
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Generation lease expired or was revoked by recovery. Finalization rejected.",
                )

            meta = ast_record.token_usage if isinstance(ast_record.token_usage, dict) else {}
            stored_token = meta.get("generation_token")
            if stored_token and stored_token != generation_token:
                logger.warning(
                    "Worker generation token mismatch for message %s: expected %s, found %s.",
                    assistant_msg_id,
                    generation_token,
                    stored_token,
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Generation lease ownership token mismatch. Finalization rejected.",
                )

            if pipeline_exception is not None:
                ast_record.status = "failed"
                err_cat, safe_msg = categorize_error(pipeline_exception)
                ast_record.error_message = safe_msg or "Failed to generate answer."
                fresh_db.commit()

                if isinstance(pipeline_exception, HTTPException):
                    raise pipeline_exception
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unexpected error occurred during message processing.",
                )

            # Transition to completed
            ast_record.status = "completed"
            ast_record.content = rag_response.answer
            ast_record.condensed_query = condensed_query
            ast_record.citations = [c.model_dump() for c in rag_response.citations] if rag_response.citations else []
            ast_record.token_usage = rag_response.token_usage
            ast_record.estimated_cost_usd = rag_response.estimated_cost_usd
            ast_record.pricing_version = rag_response.pricing_version
            ast_record.retrieval_latency_ms = rag_response.retrieval_latency_ms
            ast_record.generation_latency_ms = rag_response.generation_latency_ms
            ast_record.total_latency_ms = rag_response.total_latency_ms
            ast_record.model_used = rag_response.model_used
            ast_record.insufficient_evidence = rag_response.insufficient_evidence

            # Update conversation updated_at
            c_update = (
                fresh_db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )
            if c_update:
                c_update.updated_at = datetime.now(timezone.utc)

            fresh_db.commit()
            fresh_db.refresh(ast_record)
            return MessageResponse.model_validate(ast_record)
    finally:
        heartbeat.stop()

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Assistant message record could not be finalized.",
    )


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Deletes a conversation and its associated messages (via database cascade).
    Enforces tenant ownership (returns HTTP 404 for cross-tenant or missing).
    """
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )

    db.delete(conv)
    db.commit()

    return {"message": "Conversation deleted successfully", "id": conversation_id}
