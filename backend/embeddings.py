import os
import math
import hashlib
import logging
from typing import List, Optional, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("documind.embeddings")

# Dimensionality constraint: All embeddings in DocuMind AI must conform to 1536 dimensions.
EMBEDDING_DIMENSION = 1536

# Provider & Model configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
try:
    GEMINI_EMBEDDING_DIMENSION = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", str(EMBEDDING_DIMENSION)))
except ValueError:
    GEMINI_EMBEDDING_DIMENSION = EMBEDDING_DIMENSION

# Backwards compatibility alias
EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL

# Maximum items allowed in a single Gemini batchEmbedContents call
GEMINI_MAX_BATCH_SIZE = 100


class EmbeddingProviderError(Exception):
    """Raised when an external embedding provider call fails in live mode."""
    pass


class EmbeddingDimensionError(ValueError):
    """Raised when an embedding vector does not match the required dimensionality."""
    pass


class IncompatibleEmbeddingSpaceError(Exception):
    """Raised when query vector space does not match document embedding space."""
    pass


def get_embedding_space_id(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dimension: Optional[int] = None,
) -> str:
    """
    Returns a canonical, deterministic identifier representing an embedding space.
    Format: '{provider}:{model}:{dimension}'
    Examples:
      - 'gemini:gemini-embedding-001:1536'
      - 'openai:text-embedding-3-small:1536'
      - 'offline:deterministic-mock:1536'
    Prevents cross-provider vector comparison between mathematically incompatible spaces.
    """
    prov = (provider or get_embedding_provider()).strip().lower()
    dim = dimension or EMBEDDING_DIMENSION
    if prov == "gemini":
        mod = (model or os.getenv("GEMINI_EMBEDDING_MODEL", "").strip() or GEMINI_EMBEDDING_MODEL).strip()
    elif prov == "openai":
        mod = (model or os.getenv("EMBEDDING_MODEL", "").strip() or OPENAI_EMBEDDING_MODEL).strip()
    elif prov == "offline":
        mod = (model or "deterministic-mock").strip()
    else:
        mod = (model or "unknown").strip()
    return f"{prov}:{mod}:{dim}"


def validate_embedding_dimension(vector: List[float], expected_dim: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Validates that the provided vector is a non-empty list of numerical values
    with exactly the expected dimensionality.
    """
    if not isinstance(vector, list):
        raise EmbeddingDimensionError(f"Expected embedding vector to be a list, got {type(vector).__name__}")
    if len(vector) != expected_dim:
        raise EmbeddingDimensionError(
            f"Embedding dimensionality mismatch: expected {expected_dim}, got {len(vector)}"
        )
    return vector


def is_production_mode() -> bool:
    """
    Returns True if the application is running in production mode (SEC-05).
    Production mode is active when DOCUMIND_ENV or ENVIRONMENT is set to 'production'/'prod',
    or when running outside of pytest without an explicit development/test environment.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        documind_env = os.getenv("DOCUMIND_ENV", "").strip().lower()
        app_env = os.getenv("ENVIRONMENT", "").strip().lower()
        return documind_env in ("production", "prod") or app_env in ("production", "prod")

    documind_env = os.getenv("DOCUMIND_ENV", "").strip().lower()
    if documind_env in ("development", "dev", "test", "testing"):
        return False
    app_env = os.getenv("ENVIRONMENT", "").strip().lower()
    if app_env in ("development", "dev", "test", "testing"):
        return False
    return True


def get_embedding_provider() -> str:
    """
    Resolves the active embedding provider.
    Supported options: 'gemini', 'openai', 'offline'.

    Security & Production Enforcement (SEC-05):
    - In production mode:
      1. EMBEDDING_PROVIDER must be explicitly configured ('gemini' or 'openai').
      2. 'offline' fallback is strictly prohibited.
      3. The API key for the selected provider must be non-empty.
    - In development/test mode:
      1. Explicit EMBEDDING_PROVIDER environment variable.
      2. Hermetic test runner guard: defaults to 'offline' if running under pytest.
      3. Default based on configured credentials: 'openai' if OPENAI_API_KEY, else 'gemini' if GEMINI_API_KEY, else 'offline'.
    """
    is_prod = is_production_mode()
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()

    if is_prod:
        if not provider:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR (SEC-05): EMBEDDING_PROVIDER must be explicitly "
                "configured in production ('gemini' or 'openai'). Offline provider fallback is prohibited in production."
            )
        if provider == "offline":
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR (SEC-05): EMBEDDING_PROVIDER cannot be set to 'offline' in production mode."
            )
        if provider not in ("gemini", "openai"):
            raise ValueError(
                f"Unsupported EMBEDDING_PROVIDER '{provider}'. Must be one of: 'gemini', 'openai'."
            )
        if provider == "gemini":
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR (SEC-05): GEMINI_API_KEY is missing or empty "
                    "for EMBEDDING_PROVIDER='gemini' in production."
                )
        elif provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR (SEC-05): OPENAI_API_KEY is missing or empty "
                    "for EMBEDDING_PROVIDER='openai' in production."
                )
        return provider

    if provider:
        if provider not in ("gemini", "openai", "offline"):
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{provider}'. Must be one of: 'gemini', 'openai', 'offline'.")
        return provider

    # Hermetic test runner guard
    if "PYTEST_CURRENT_TEST" in os.environ:
        return "offline"

    # Default based on configured credentials
    if OPENAI_API_KEY:
        return "openai"
    elif GEMINI_API_KEY:
        return "gemini"
    return "offline"


def validate_embedding_provider_configuration() -> None:
    """
    Validates embedding provider configuration at startup (SEC-05).
    In production mode, confirms explicit provider configuration and credentials.
    """
    if is_production_mode():
        get_embedding_provider()


def is_live_embedding_provider() -> bool:
    """
    Returns True if the effective embedding provider is a live external service
    (e.g., 'gemini' or 'openai'), and False if operating in 'offline' mode.
    """
    try:
        return get_embedding_provider() in ("gemini", "openai")
    except ValueError:
        return False


def generate_deterministic_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Generates a deterministic, normalized unit vector of fixed dimension from text.
    Used exclusively when explicitly configured for offline/test operation.
    Ensures mathematical consistency without runtime dependencies on external APIs.
    """
    base_hash = hashlib.sha256(text.encode("utf-8")).digest()
    vector: List[float] = []

    for i in range(dimension):
        byte_idx = i % len(base_hash)
        val = (base_hash[byte_idx] + (i * 31)) % 256
        vector.append(float(val - 128))

    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0:
        vector = [x / norm for x in vector]
    else:
        vector[0] = 1.0

    return validate_embedding_dimension(vector, dimension)


# ---------------------------------------------------------------------------
# Gemini Provider Implementation
# ---------------------------------------------------------------------------

def _get_gemini_api_key() -> str:
    """Retrieves and validates Gemini API key."""
    key = os.getenv("GEMINI_API_KEY")
    if key is None:
        key = GEMINI_API_KEY
    key = key.strip()
    if not key:
        raise EmbeddingProviderError(
            "GEMINI_API_KEY environment variable is missing or empty. Cannot generate Gemini embeddings."
        )
    return key


def _get_gemini_embedding_single(text: str) -> List[float]:
    """
    Generates an embedding vector using Gemini API's embedContent endpoint.
    Passes x-goog-api-key header and requests 1536 dimensions.
    """
    api_key = _get_gemini_api_key()
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip() or GEMINI_EMBEDDING_MODEL
    dim = GEMINI_EMBEDDING_DIMENSION
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    payload = {
        "content": {
            "parts": [{"text": text}]
        },
        "outputDimensionality": dim,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            res.raise_for_status()
            data = res.json()

            if not isinstance(data, dict) or "embedding" not in data:
                raise EmbeddingProviderError("Malformed response from Gemini API: missing 'embedding' field.")

            values = data["embedding"].get("values")
            if not isinstance(values, list):
                raise EmbeddingProviderError("Malformed response from Gemini API: 'values' is not a list.")

            return validate_embedding_dimension(values, EMBEDDING_DIMENSION)

    except httpx.HTTPStatusError as err:
        status_code = err.response.status_code
        if status_code in (401, 403):
            raise EmbeddingProviderError(
                f"Gemini authentication failed (HTTP {status_code}). Verify your GEMINI_API_KEY."
            ) from err
        elif status_code == 429:
            raise EmbeddingProviderError("Gemini rate limit exceeded (HTTP 429). Please retry shortly.") from err
        else:
            raise EmbeddingProviderError(f"Gemini API returned HTTP status {status_code}.") from err
    except httpx.TimeoutException as err:
        raise EmbeddingProviderError("Gemini API request timed out after 30 seconds.") from err
    except (EmbeddingProviderError, EmbeddingDimensionError):
        raise
    except Exception as err:
        raise EmbeddingProviderError(f"Gemini API communication error: {type(err).__name__}") from err


def _get_gemini_embeddings_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Generates embedding vectors for multiple passages using Gemini API's batchEmbedContents endpoint.
    Maintains exact input/output ordering.
    """
    if not texts:
        return []

    api_key = _get_gemini_api_key()
    active_model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "").strip() or GEMINI_EMBEDDING_MODEL
    dim = GEMINI_EMBEDDING_DIMENSION
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:batchEmbedContents"

    all_embeddings: List[List[float]] = []

    # Process in chunks up to GEMINI_MAX_BATCH_SIZE to avoid exceeding payload limits
    for start in range(0, len(texts), GEMINI_MAX_BATCH_SIZE):
        sub_texts = texts[start:start + GEMINI_MAX_BATCH_SIZE]
        requests_payload = [
            {
                "model": f"models/{active_model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": dim,
            }
            for t in sub_texts
        ]

        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(
                    url,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"requests": requests_payload},
                )
                res.raise_for_status()
                data = res.json()

                if not isinstance(data, dict) or "embeddings" not in data:
                    raise EmbeddingProviderError("Malformed response from Gemini batch API: missing 'embeddings' field.")

                embeddings_list = data["embeddings"]
                if not isinstance(embeddings_list, list) or len(embeddings_list) != len(sub_texts):
                    raise EmbeddingProviderError(
                        f"Gemini batch response count mismatch: expected {len(sub_texts)}, got {len(embeddings_list) if isinstance(embeddings_list, list) else 'invalid'}"
                    )

                for item in embeddings_list:
                    if not isinstance(item, dict) or "values" not in item:
                        raise EmbeddingProviderError("Malformed item in Gemini batch response: missing 'values'.")
                    values = item["values"]
                    validated = validate_embedding_dimension(values, EMBEDDING_DIMENSION)
                    all_embeddings.append(validated)

        except httpx.HTTPStatusError as err:
            status_code = err.response.status_code
            if status_code in (401, 403):
                raise EmbeddingProviderError(
                    f"Gemini batch authentication failed (HTTP {status_code}). Verify your GEMINI_API_KEY."
                ) from err
            elif status_code == 429:
                raise EmbeddingProviderError("Gemini batch rate limit exceeded (HTTP 429). Please retry shortly.") from err
            else:
                raise EmbeddingProviderError(f"Gemini batch API returned HTTP status {status_code}.") from err
        except httpx.TimeoutException as err:
            raise EmbeddingProviderError("Gemini batch API request timed out.") from err
        except (EmbeddingProviderError, EmbeddingDimensionError):
            raise
        except Exception as err:
            raise EmbeddingProviderError(f"Gemini batch API communication error: {type(err).__name__}") from err

    return all_embeddings


# ---------------------------------------------------------------------------
# OpenAI Provider Implementation
# ---------------------------------------------------------------------------

def _get_openai_api_key() -> str:
    """Retrieves and validates OpenAI API key."""
    key = os.getenv("OPENAI_API_KEY")
    if key is None:
        key = OPENAI_API_KEY
    key = key.strip()
    if not key:
        raise EmbeddingProviderError(
            "OPENAI_API_KEY environment variable is missing or empty. Cannot generate OpenAI embeddings."
        )
    return key


def _get_openai_embedding_single(text: str) -> List[float]:
    """Generates an embedding vector using OpenAI text-embedding-3-small."""
    api_key = _get_openai_api_key()
    model = os.getenv("EMBEDDING_MODEL", "").strip() or OPENAI_EMBEDDING_MODEL

    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": text,
                    "model": model,
                },
            )
            res.raise_for_status()
            data = res.json()

            if not isinstance(data, dict) or "data" not in data or not data["data"]:
                raise EmbeddingProviderError("Malformed response from OpenAI API: missing 'data' field.")

            embedding = data["data"][0].get("embedding")
            return validate_embedding_dimension(embedding, EMBEDDING_DIMENSION)

    except httpx.HTTPStatusError as err:
        status_code = err.response.status_code
        if status_code in (401, 403):
            raise EmbeddingProviderError(
                f"OpenAI authentication failed (HTTP {status_code}). Verify your OPENAI_API_KEY."
            ) from err
        elif status_code == 429:
            raise EmbeddingProviderError("OpenAI rate limit exceeded (HTTP 429). Please retry shortly.") from err
        else:
            raise EmbeddingProviderError(f"OpenAI API returned HTTP status {status_code}.") from err
    except httpx.TimeoutException as err:
        raise EmbeddingProviderError("OpenAI API request timed out after 30 seconds.") from err
    except (EmbeddingProviderError, EmbeddingDimensionError):
        raise
    except Exception as err:
        raise EmbeddingProviderError(f"OpenAI API communication error: {type(err).__name__}") from err


def _get_openai_embeddings_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Generates embedding vectors for multiple passages using OpenAI embeddings API."""
    if not texts:
        return []

    api_key = _get_openai_api_key()
    active_model = model or os.getenv("EMBEDDING_MODEL", "").strip() or OPENAI_EMBEDDING_MODEL

    try:
        with httpx.Client(timeout=60.0) as client:
            res = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": active_model,
                },
            )
            res.raise_for_status()
            data = res.json()

            if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
                raise EmbeddingProviderError("Malformed response from OpenAI batch API: missing 'data' field.")

            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            if len(sorted_data) != len(texts):
                raise EmbeddingProviderError(
                    f"OpenAI batch count mismatch: expected {len(texts)}, got {len(sorted_data)}"
                )

            results: List[List[float]] = []
            for item in sorted_data:
                embedding = item.get("embedding")
                results.append(validate_embedding_dimension(embedding, EMBEDDING_DIMENSION))
            return results

    except httpx.HTTPStatusError as err:
        status_code = err.response.status_code
        if status_code in (401, 403):
            raise EmbeddingProviderError(
                f"OpenAI batch authentication failed (HTTP {status_code}). Verify your OPENAI_API_KEY."
            ) from err
        elif status_code == 429:
            raise EmbeddingProviderError("OpenAI batch rate limit exceeded (HTTP 429). Please retry shortly.") from err
        else:
            raise EmbeddingProviderError(f"OpenAI batch API returned HTTP status {status_code}.") from err
    except httpx.TimeoutException as err:
        raise EmbeddingProviderError("OpenAI batch API request timed out.") from err
    except (EmbeddingProviderError, EmbeddingDimensionError):
        raise
    except Exception as err:
        raise EmbeddingProviderError(f"OpenAI batch API communication error: {type(err).__name__}") from err


# ---------------------------------------------------------------------------
# Public Facade API
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Generates a 1536-dimensional vector embedding for a single text passage.
    Delegates to the configured provider ('gemini', 'openai', or 'offline').

    Critical Safety Rules:
    1. If the live provider fails, raises EmbeddingProviderError.
    2. Never silently falls back to pseudo-vectors in live mode.
    3. Validates output dimension strictly to 1536.
    """
    cleaned = text.replace("\n", " ").strip() if isinstance(text, str) else ""
    if not cleaned:
        cleaned = "empty"

    provider = get_embedding_provider()

    if provider == "gemini":
        return _get_gemini_embedding_single(cleaned)
    elif provider == "openai":
        return _get_openai_embedding_single(cleaned)
    elif provider == "offline":
        return generate_deterministic_embedding(cleaned, EMBEDDING_DIMENSION)
    else:
        raise EmbeddingProviderError(f"Unsupported embedding provider: {provider}")


def get_embeddings_batch(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Generates 1536-dimensional vector embeddings for multiple text passages.
    Preserves exact input/output order.
    Delegates to the configured provider ('gemini', 'openai', or 'offline')
    or explicit provider/model override if supplied.

    Critical Safety Rules:
    1. If the live provider fails, raises EmbeddingProviderError.
    2. Never silently falls back to pseudo-vectors in live mode.
    3. Validates output dimension strictly to 1536.
    """
    if not texts:
        return []

    cleaned = [t.replace("\n", " ").strip() or "empty" if isinstance(t, str) else "empty" for t in texts]
    active_provider = (provider or get_embedding_provider()).strip().lower()

    if active_provider == "gemini":
        return _get_gemini_embeddings_batch(cleaned, model=model)
    elif active_provider == "openai":
        return _get_openai_embeddings_batch(cleaned, model=model)
    elif active_provider == "offline":
        return [generate_deterministic_embedding(t, EMBEDDING_DIMENSION) for t in cleaned]
    else:
        raise EmbeddingProviderError(f"Unsupported embedding provider: {active_provider}")
