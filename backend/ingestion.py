import os
import uuid
import re
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from sqlalchemy.orm import Session

from backend.models import Document, DocumentChunk
from backend.embeddings import get_embeddings_batch

# Chunking Strategy Configuration
DEFAULT_CHUNK_SIZE = 800       # Target characters per chunk (~150-200 words)
DEFAULT_CHUNK_OVERLAP = 150    # Characters of shared context between adjacent chunks


def clean_text(text: str) -> str:
    """Normalizes excessive whitespace, carriage returns, and control characters."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace sequences of 3+ newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces/tabs into single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pages_text(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text page-by-page from a PDF file using pypdf.
    Returns: List of (page_number, text) tuples.
    Raises ValueError for encrypted PDFs or documents with zero extractable text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found at {file_path}")

    reader = PdfReader(file_path)

    # 1. Check for encryption/password protection
    if reader.is_encrypted:
        raise ValueError("Encrypted or password-protected PDFs are not supported.")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("Invalid PDF: Document contains zero pages.")

    pages_data: List[Tuple[int, str]] = []
    total_characters = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        pages_data.append((page_num, cleaned))
        total_characters += len(cleaned)

    # 2. Check for empty or image-only scanned PDFs (No OCR support disclaimer)
    if total_characters == 0:
        raise ValueError(
            "No extractable text found. The document may be empty, corrupted, or scanned (image-only). "
            "OCR (Optical Character Recognition) is not supported."
        )

    return pages_data


def chunk_page_text(
    text: str,
    page_number: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Splits a single page's text into overlapping chunks while preserving page numbers.
    Uses a sliding window approach with punctuation snapping to avoid cutting sentences.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [{
            "page_number": page_number,
            "text": text,
        }]

    chunks: List[Dict[str, Any]] = []
    start = 0
    text_len = len(text)
    step = chunk_size - chunk_overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Attempt to snap boundary to natural sentence/paragraph end if not at the end of text
        if end < text_len:
            # Look backwards up to 80 characters for punctuation
            lookback_zone = text[max(start, end - 80):end]
            punctuation_match = re.search(r"(\. |\.\n|\n\n|\n|\? |\! )", lookback_zone)
            if punctuation_match:
                # Adjust end to boundary
                last_idx = lookback_zone.rfind(punctuation_match.group(0))
                if last_idx != -1:
                    adjusted_end = max(start, end - 80) + last_idx + len(punctuation_match.group(0))
                    if adjusted_end > start + 50:  # Avoid overly tiny chunks
                        end = adjusted_end

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "page_number": page_number,
                "text": chunk_text,
            })

        if end >= text_len:
            break

        # Advance start using overlap: guarantees shared context between adjacent chunks
        start = max(start + 1, end - chunk_overlap)

    return chunks


def process_document_pipeline(document_id: str, db: Session) -> Document:
    """
    Synchronous / in-process execution pipeline:
    1. Sets status to 'processing'
    2. Extracts text per page
    3. Generates overlapping chunks with page metadata
    4. Computes 1536-dimensional vector embeddings
    5. Stores chunks idempotently in PostgreSQL / pgvector
    6. Updates status to 'completed' or 'failed'
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ValueError(f"Document {document_id} not found in database.")

    document.status = "processing"
    document.error_message = None
    db.commit()

    try:
        # Extract text per page
        pages_data = extract_pages_text(document.storage_path)

        # Build chunks across all pages
        all_chunks: List[Dict[str, Any]] = []
        for page_num, page_text in pages_data:
            page_chunks = chunk_page_text(page_text, page_num)
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise ValueError(
                "No extractable text found. Document may be scanned (image-only). OCR is not supported."
            )

        # Generate embeddings in batch
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = get_embeddings_batch(chunk_texts)

        # Idempotency guard: Clear any prior chunks for this document if re-processing
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

        # Insert new chunks
        for idx, (chunk_data, embedding) in enumerate(zip(all_chunks, embeddings)):
            chunk_record = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                chunk_index=idx,
                page_number=chunk_data["page_number"],
                text=chunk_data["text"],
                embedding=embedding,
            )
            db.add(chunk_record)

        document.status = "completed"
        document.error_message = None
        db.commit()
        db.refresh(document)
        return document

    except Exception as err:
        db.rollback()
        # Mark document as failed with clean diagnostic message
        failed_doc = db.query(Document).filter(Document.id == document_id).first()
        if failed_doc:
            failed_doc.status = "failed"
            failed_doc.error_message = str(err)
            db.commit()
            db.refresh(failed_doc)
        raise err
