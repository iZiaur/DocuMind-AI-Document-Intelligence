import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.models import User

load_dotenv()

# Security Configurations
INSECURE_DEV_JWT_SECRET = "documind_super_secret_jwt_key_for_development_change_in_production"
MIN_JWT_SECRET_LENGTH = 32  # Minimum 256 bits of entropy required for HS256
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# FastAPI Bearer token extractor
security = HTTPBearer(auto_error=False)


def is_test_or_development_mode() -> bool:
    """
    Determines whether the application is running under pytest or explicit local development.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    documind_env = os.getenv("DOCUMIND_ENV", "").strip().lower()
    if documind_env in ("development", "dev", "test", "testing"):
        return True
    app_env = os.getenv("ENVIRONMENT", "").strip().lower()
    if app_env in ("development", "dev", "test", "testing"):
        return True
    return False


def get_jwt_secret() -> str:
    """
    Retrieves and validates the JWT secret key.

    Security Policy (SEC-01):
    1. Reads strictly from JWT_SECRET environment variable.
    2. In production (non-test/non-dev):
       - Refuses to start if JWT_SECRET is missing or empty.
       - Refuses to start if JWT_SECRET equals the insecure development placeholder.
       - Refuses to start if len(JWT_SECRET) < 32 characters.
    3. In explicit test/development mode:
       - Allows the development secret or a deterministic test key if unset.
    4. Never prints or logs the secret value.
    """
    secret = os.getenv("JWT_SECRET", "").strip()
    is_dev = is_test_or_development_mode()

    if not secret:
        if is_dev:
            return INSECURE_DEV_JWT_SECRET
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: JWT_SECRET environment variable is missing or empty. "
            "DocuMind AI refuses to start without a cryptographically secure JWT secret."
        )

    if not is_dev:
        if secret == INSECURE_DEV_JWT_SECRET:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: Insecure default JWT_SECRET detected in production. "
                "You must configure a unique, high-entropy secret in your environment."
            )
        if len(secret) < MIN_JWT_SECRET_LENGTH:
            raise RuntimeError(
                f"CRITICAL SECURITY CONFIGURATION ERROR: JWT_SECRET is insufficiently strong ({len(secret)} characters). "
                f"It must be at least {MIN_JWT_SECRET_LENGTH} characters for HS256 security."
            )

    return secret


def validate_jwt_secret() -> None:
    """Validates the JWT secret at application startup."""
    get_jwt_secret()


# Backwards compatibility alias for module-level readers
def _get_active_secret() -> str:
    try:
        return get_jwt_secret()
    except RuntimeError:
        return ""

JWT_SECRET = _get_active_secret()


def hash_password(password: str) -> str:
    """
    Hashes a plain text password using bcrypt with a randomly generated salt.
    Never stores plaintext passwords.
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain password against the stored bcrypt hash.
    Constant-time comparison protects against timing attacks.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(user_id: int, username: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generates a cryptographically signed JWT containing user ID (subject) and expiry.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }

    secret = get_jwt_secret()
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and cryptographically validates a JWT token.
    Raises HTTPException(401) on expiration or signature tampering.
    """
    try:
        secret = get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = None,  # Provided by caller endpoint via db dependency
) -> User:
    """
    FastAPI security dependency.
    Validates token, extracts authenticated user from DB, and rejects unauthenticated requests.
    CRITICAL: Derives user identity strictly from token, never client-submitted input.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)

    user_id_str: Optional[str] = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid subject in token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # When db session is injected, fetch user from DB
    if db is not None:
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account not found or deactivated.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    # Fallback placeholder if db is queried directly in route
    return user_id  # type: ignore
