import io
import os
import uuid
import hashlib
import tempfile
from typing import Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import HTTPException, status, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

load_dotenv()

# Storage configuration (SEC-03: Default 10MB maximum upload size)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "storage/documents")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KB read buffer to prevent memory exhaustion

# Ensure storage directory exists
storage_path = Path(UPLOAD_DIR)
storage_path.mkdir(parents=True, exist_ok=True)


async def stream_and_validate_upload(
    upload_file: UploadFile,
    target_dir: Optional[str] = None,
    content_length: Optional[str] = None,
) -> Tuple[str, str, int, str, int]:
    """
    Incrementally streams an uploaded file to disk and validates its integrity (SEC-03).

    Security & Memory Safeguards:
    1. Checks Content-Length header upfront if present (fast reject before reading).
    2. Streams file in 64 KB chunks directly to a temporary file on disk.
    3. Memory footprint stays constant (~64 KB) regardless of file size (prevents OOM).
    4. Aborts stream immediately if cumulative bytes exceed MAX_UPLOAD_SIZE_BYTES (HTTP 413).
    5. Validates %PDF- magic signature and structural integrity using pypdf.
    6. Atomically moves the temporary file to its final UUID location on success.
    7. Always cleans up temporary files on success, failure, or exception.

    Returns: (document_id, storage_path, page_count, file_hash, file_size_bytes)
    """
    # 1. Upfront Content-Length header verification
    content_length_header = content_length or upload_file.headers.get("content-length")
    if content_length_header:
        try:
            declared_size = int(content_length_header)
            if declared_size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB ({declared_size} bytes declared).",
                )
        except ValueError:
            pass

    dest_dir = Path(target_dir) if target_dir else storage_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 2. Open a safe temporary file in the destination directory
    temp_file = tempfile.NamedTemporaryFile(
        dir=dest_dir,
        delete=False,
        prefix="documind_upload_",
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    hasher = hashlib.sha256()
    total_bytes = 0

    try:
        # 3. Stream incrementally in chunks
        while True:
            chunk = await upload_file.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                temp_file.close()
                if temp_path.exists():
                    temp_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB ({total_bytes} bytes received).",
                )
            temp_file.write(chunk)
            hasher.update(chunk)

        temp_file.flush()
        temp_file.close()

        # 4. Check for empty files
        if total_bytes == 0:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # 5. Magic byte signature check
        with open(temp_path, "rb") as f:
            magic_bytes = f.read(5)
        if magic_bytes != b"%PDF-":
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format: File header does not match PDF signature (%PDF-). Spoofed files are rejected.",
            )

        # 6. Deep PDF structural validation
        try:
            reader = PdfReader(str(temp_path))
            if reader.is_encrypted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Encrypted or password-protected PDFs are not supported.",
                )
            page_count = len(reader.pages)
            if page_count < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid PDF: Document contains 0 valid pages.",
                )
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

        # 7. Atomically move temporary file to final UUID-named file
        doc_id = str(uuid.uuid4())
        safe_disk_filename = f"{doc_id}.pdf"
        destination = dest_dir / safe_disk_filename
        os.replace(temp_path, destination)

        return doc_id, str(destination.resolve()), page_count, hasher.hexdigest(), total_bytes

    finally:
        # Guarantee cleanup of temporary file if an exception occurred before os.replace
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


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


def save_pdf_securely(
    file_bytes: bytes,
    original_filename: str,
    target_dir: Optional[str] = None,
) -> Tuple[str, str, int, str]:
    """
    Validates file size, validates PDF structure, and writes to isolated server storage.
    Uses UUID4 identifiers to prevent directory traversal and file collision attacks.
    If target_dir is provided, writes to target_dir (useful for isolated evaluation runs);
    otherwise defaults to standard application storage_path (production upload dir).
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
    dest_dir = Path(target_dir) if target_dir else storage_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / safe_disk_filename

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
