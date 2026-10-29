import io
import os
import uuid
from typing import Tuple
from pathlib import Path
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pypdf import PdfReader
from pypdf.errors import PdfReadError

load_dotenv()

# Storage configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "storage/documents")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Ensure storage directory exists
storage_path = Path(UPLOAD_DIR)
storage_path.mkdir(parents=True, exist_ok=True)


def validate_pdf_content(file_bytes: bytes) -> int:
    """
    Validates true file content:
    1. Checks PDF magic header signature (%PDF-).
    2. Validates document structure using pypdf reader.
    3. Returns the valid page count.
    
    Rejects spoofed extensions, empty files, or corrupted PDF binaries.
    """
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Magic signature verification: All legitimate PDFs start with %PDF- (hex 25 50 44 46 2D)
    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: File header does not match PDF signature (%PDF-). Spoofed files are rejected.",
        )

    try:
        # Validate internal PDF stream & dictionary structure
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)

        # Ensure pages can be parsed
        page_count = len(reader.pages)
        if page_count < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF: Document contains 0 valid pages.",
            )

        return page_count
    except PdfReadError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed or corrupted PDF structure: {str(err)}",
        )
    except Exception as err:
        if isinstance(err, HTTPException):
            raise err
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse document. Ensure the file is a valid, uncorrupted PDF.",
        )


def save_pdf_securely(file_bytes: bytes, original_filename: str) -> Tuple[str, str, int, str]:
    """
    Validates file size, validates PDF structure, and writes to isolated server storage.
    Uses UUID4 identifiers to prevent directory traversal and file collision attacks.
    Returns: (document_id, storage_path, page_count, file_hash)
    """
    import hashlib

    # 1. Enforce strict size limit
    file_size = len(file_bytes)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB ({file_size} bytes received).",
        )

    # 2. Deep content validation (magic bytes + PDF structure)
    page_count = validate_pdf_content(file_bytes)

    # 3. Compute SHA-256 hash for deduplication
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # 4. Generate safe server-side identifier (never trust client filename for disk location)
    doc_id = str(uuid.uuid4())
    safe_disk_filename = f"{doc_id}.pdf"
    destination = storage_path / safe_disk_filename

    # 5. Write binary to isolated storage
    with open(destination, "wb") as f:
        f.write(file_bytes)

    return doc_id, str(destination.resolve()), page_count, file_hash


def remove_pdf_from_storage(file_path_str: str) -> None:
    """
    Safely removes a file from disk during document deletion.
    Handles non-existent files gracefully.
    """
    try:
        path = Path(file_path_str)
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass
