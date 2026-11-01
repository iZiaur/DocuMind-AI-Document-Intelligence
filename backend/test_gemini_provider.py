import os
from unittest.mock import patch, MagicMock
import pytest
import httpx

from backend.embeddings import (
    get_embedding,
    get_embeddings_batch,
    get_embedding_provider,
    validate_embedding_dimension,
    generate_deterministic_embedding,
    EmbeddingProviderError,
    EmbeddingDimensionError,
    EMBEDDING_DIMENSION,
)


def _make_dummy_vector(dim: int = 1536, seed: float = 0.01) -> list:
    """Creates a deterministic dummy vector of specified dimension."""
    return [(seed + i * 0.001) % 1.0 for i in range(dim)]


# ---------------------------------------------------------------------------
# Test 1: Gemini Single Embedding (Mocked)
# ---------------------------------------------------------------------------

def test_gemini_single_embedding(monkeypatch):
    """Test Gemini single text embedding generates verified 1536-dimensional vector."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    expected_vector = _make_dummy_vector(1536, seed=0.05)

    def mock_post(url, *args, **kwargs):
        headers = kwargs.get("headers", {})
        json_data = kwargs.get("json", {})
        assert "gemini-embedding-001:embedContent" in url
        assert headers.get("x-goog-api-key") == "test-gemini-key"
        assert json_data["outputDimensionality"] == 1536
        assert json_data["content"]["parts"][0]["text"] == "Hello DocuMind"

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embedding": {
                "values": expected_vector
            }
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        res = get_embedding("Hello DocuMind")

    assert len(res) == 1536
    assert res == expected_vector


# ---------------------------------------------------------------------------
# Test 2: Gemini Batch Embedding & Order Preservation (Mocked)
# ---------------------------------------------------------------------------

def test_gemini_batch_embedding_order_preserved(monkeypatch):
    """Test Gemini batch embedding preserves exact input-to-output ordering."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    inputs = ["Passage Alpha", "Passage Beta", "Passage Gamma"]
    v_alpha = _make_dummy_vector(1536, seed=0.1)
    v_beta = _make_dummy_vector(1536, seed=0.2)
    v_gamma = _make_dummy_vector(1536, seed=0.3)

    def mock_post(url, *args, **kwargs):
        headers = kwargs.get("headers", {})
        json_data = kwargs.get("json", {})
        assert "gemini-embedding-001:batchEmbedContents" in url
        assert headers.get("x-goog-api-key") == "test-gemini-key"
        reqs = json_data["requests"]
        assert len(reqs) == 3
        assert reqs[0]["content"]["parts"][0]["text"] == "Passage Alpha"
        assert reqs[1]["content"]["parts"][0]["text"] == "Passage Beta"
        assert reqs[2]["content"]["parts"][0]["text"] == "Passage Gamma"

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embeddings": [
                {"values": v_alpha},
                {"values": v_beta},
                {"values": v_gamma},
            ]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        results = get_embeddings_batch(inputs)

    assert len(results) == 3
    assert results[0] == v_alpha
    assert results[1] == v_beta
    assert results[2] == v_gamma


# ---------------------------------------------------------------------------
# Test 3: Correct 1536-Dimensional Output Validation
# ---------------------------------------------------------------------------

def test_validate_embedding_dimension_exact():
    """Validates that vectors with exactly 1536 floats pass without modification."""
    v = _make_dummy_vector(1536)
    validated = validate_embedding_dimension(v, 1536)
    assert validated == v
    assert len(validated) == 1536


# ---------------------------------------------------------------------------
# Test 4: Incorrect Dimension Rejection
# ---------------------------------------------------------------------------

def test_incorrect_dimension_rejection(monkeypatch):
    """Vectors that deviate from 1536 dimensions (e.g., 768) must be rejected with EmbeddingDimensionError."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    short_vector = _make_dummy_vector(768)

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": {"values": short_vector}}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            get_embedding("Test short vector")

    assert "Embedding dimensionality mismatch: expected 1536, got 768" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 5: Malformed API Response
# ---------------------------------------------------------------------------

def test_gemini_malformed_api_response(monkeypatch):
    """Malformed or missing response structures raise EmbeddingProviderError."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    # Case A: Missing 'embedding' field
    def mock_missing_field(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unrelated_field": {}}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_missing_field):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            get_embedding("Test malformed")
        assert "missing 'embedding' field" in str(exc_info.value)

    # Case B: 'values' is not a list
    def mock_non_list_values(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": {"values": "not-a-list"}}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_non_list_values):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            get_embedding("Test malformed values")
        assert "'values' is not a list" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 6: HTTP 401/403 Authentication Errors
# ---------------------------------------------------------------------------

def test_gemini_authentication_error(monkeypatch):
    """HTTP 401/403 errors raise sanitized EmbeddingProviderError without exposing the API key."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    secret_key = "super-sensitive-secret-gemini-api-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)

    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent")
    res_401 = httpx.Response(401, request=req, json={"error": {"code": 401, "message": "API key not valid"}})

    def mock_auth_fail(*args, **kwargs):
        raise httpx.HTTPStatusError("Unauthorized", request=req, response=res_401)

    with patch("httpx.Client.post", side_effect=mock_auth_fail):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            get_embedding("Test auth error")

    err_msg = str(exc_info.value)
    assert "Gemini authentication failed (HTTP 401)" in err_msg
    # Ensure sensitive key is never leaked
    assert secret_key not in err_msg


# ---------------------------------------------------------------------------
# Test 7: HTTP 429 Rate-Limit Error
# ---------------------------------------------------------------------------

def test_gemini_rate_limit_error(monkeypatch):
    """HTTP 429 raises clear, sanitized rate limit error."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent")
    res_429 = httpx.Response(429, request=req, json={"error": {"code": 429, "message": "Resource exhausted"}})

    def mock_rate_limit(*args, **kwargs):
        raise httpx.HTTPStatusError("Rate limited", request=req, response=res_429)

    with patch("httpx.Client.post", side_effect=mock_rate_limit):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            get_embedding("Test rate limit")

    assert "rate limit exceeded (HTTP 429)" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 8: Network Timeout
# ---------------------------------------------------------------------------

def test_gemini_timeout_error(monkeypatch):
    """Network timeouts raise sanitized EmbeddingProviderError."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    def mock_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out after 30s")

    with patch("httpx.Client.post", side_effect=mock_timeout):
        with pytest.raises(EmbeddingProviderError) as exc_info:
            get_embedding("Test timeout")

    assert "timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 9: No Silent Deterministic Fallback on Live Provider Failure
# ---------------------------------------------------------------------------

def test_no_silent_fallback_on_live_failure(monkeypatch):
    """CRITICAL: Live provider failure must NEVER silently return pseudo-vectors."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    def mock_crash(*args, **kwargs):
        raise httpx.ConnectError("Network unreachable")

    with patch("httpx.Client.post", side_effect=mock_crash):
        # Must raise EmbeddingProviderError rather than returning a 1536-dim vector
        with pytest.raises(EmbeddingProviderError):
            get_embedding("Security policy document")

        with pytest.raises(EmbeddingProviderError):
            get_embeddings_batch(["Security policy chunk 1", "Security policy chunk 2"])


# ---------------------------------------------------------------------------
# Test 10: OpenAI Regression (Mocked)
# ---------------------------------------------------------------------------

def test_openai_regression_single_and_batch(monkeypatch):
    """Verify OpenAI provider functions identically with dimension validation and strict error handling."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    v1 = _make_dummy_vector(1536, seed=0.4)
    v2 = _make_dummy_vector(1536, seed=0.5)

    # 1. Single test
    def mock_single(url, *args, **kwargs):
        headers = kwargs.get("headers", {})
        json_data = kwargs.get("json", {})
        assert "api.openai.com" in url
        assert headers.get("Authorization") == "Bearer test-openai-key"
        assert json_data["model"] == "text-embedding-3-small"
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": v1, "index": 0}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_single):
        res_single = get_embedding("OpenAI single test")
    assert res_single == v1
    assert len(res_single) == 1536

    # 2. Batch test with unsorted index return (verifies sorting by index)
    def mock_batch(url, *args, **kwargs):
        assert "api.openai.com" in url
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        # Return out of order: index 1 then index 0
        mock_resp.json.return_value = {
            "data": [
                {"embedding": v2, "index": 1},
                {"embedding": v1, "index": 0},
            ]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_batch):
        res_batch = get_embeddings_batch(["item 0", "item 1"])
    assert res_batch == [v1, v2]

    # 3. OpenAI failure does not silently fall back
    def mock_openai_fail(*args, **kwargs):
        raise httpx.ConnectError("OpenAI unreachable")

    with patch("httpx.Client.post", side_effect=mock_openai_fail):
        with pytest.raises(EmbeddingProviderError):
            get_embedding("Fail test")


# ---------------------------------------------------------------------------
# Test 11: Explicit Offline Provider Behavior
# ---------------------------------------------------------------------------

def test_explicit_offline_provider_behavior(monkeypatch):
    """When EMBEDDING_PROVIDER=offline is explicitly set, deterministic unit vectors are returned without network calls."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    # Even if httpx.Client.post is configured to raise an error, offline provider should NOT call it
    def forbidden_network_call(*args, **kwargs):
        pytest.fail("Network call attempted while EMBEDDING_PROVIDER=offline!")

    with patch("httpx.Client.post", side_effect=forbidden_network_call):
        vec_single = get_embedding("Enterprise Security Baseline")
        vec_batch = get_embeddings_batch(["Passage 1", "Passage 2"])

    assert len(vec_single) == 1536
    assert len(vec_batch) == 2
    assert len(vec_batch[0]) == 1536
    assert len(vec_batch[1]) == 1536

    # Verify deterministic reproducibility
    expected_single = generate_deterministic_embedding("Enterprise Security Baseline", 1536)
    assert vec_single == expected_single


# ---------------------------------------------------------------------------
# Test 12: Empty and Invalid Input Handling
# ---------------------------------------------------------------------------

def test_empty_and_invalid_input_handling(monkeypatch):
    """Empty strings and edge inputs are handled safely without unhandled exceptions."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    # Empty text maps to 'empty'
    vec_empty = get_embedding("")
    assert len(vec_empty) == 1536

    # Newlines only
    vec_newlines = get_embedding("\n\n   \n")
    assert len(vec_newlines) == 1536

    # Empty batch returns empty list immediately
    batch_empty = get_embeddings_batch([])
    assert batch_empty == []

    # Batch with mixed empty strings
    batch_mixed = get_embeddings_batch(["", "valid text", "  \n  "])
    assert len(batch_mixed) == 3
    for v in batch_mixed:
        assert len(v) == 1536


# ---------------------------------------------------------------------------
# Test 13: Missing API Key in Live Mode
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_error(monkeypatch):
    """In live mode, missing or empty API keys immediately raise EmbeddingProviderError."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(EmbeddingProviderError) as exc_info:
        get_embedding("Test missing key")
    assert "GEMINI_API_KEY environment variable is missing" in str(exc_info.value)

    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(EmbeddingProviderError) as exc_info:
        get_embedding("Test missing key")
    assert "OPENAI_API_KEY environment variable is missing" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 14: Provider Configuration Validation
# ---------------------------------------------------------------------------

def test_invalid_provider_configuration(monkeypatch):
    """Unsupported provider names raise a clear ValueError."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "anthropic_unsupported")
    with pytest.raises(ValueError) as exc_info:
        get_embedding("Test invalid provider")
    assert "Unsupported EMBEDDING_PROVIDER" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 15–19: Provider-Aware Hybrid Retrieval Alpha Selection
# ---------------------------------------------------------------------------

def test_hybrid_alpha_offline_with_openai_key_present(monkeypatch):
    """EMBEDDING_PROVIDER=offline selects alpha 0.2 even when an OpenAI key exists."""
    from backend.retrieval import get_hybrid_alpha, DEFAULT_OFFLINE_ALPHA
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-dummy-openai-key-that-exists")
    monkeypatch.delenv("HYBRID_RETRIEVAL_ALPHA", raising=False)

    alpha = get_hybrid_alpha()
    assert alpha == DEFAULT_OFFLINE_ALPHA
    assert alpha == 0.2


def test_hybrid_alpha_gemini_provider(monkeypatch):
    """EMBEDDING_PROVIDER=gemini selects alpha 0.5."""
    from backend.retrieval import get_hybrid_alpha, DEFAULT_LIVE_ALPHA
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.delenv("HYBRID_RETRIEVAL_ALPHA", raising=False)

    alpha = get_hybrid_alpha()
    assert alpha == DEFAULT_LIVE_ALPHA
    assert alpha == 0.5


def test_hybrid_alpha_openai_provider(monkeypatch):
    """EMBEDDING_PROVIDER=openai selects alpha 0.5."""
    from backend.retrieval import get_hybrid_alpha, DEFAULT_LIVE_ALPHA
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("HYBRID_RETRIEVAL_ALPHA", raising=False)

    alpha = get_hybrid_alpha()
    assert alpha == DEFAULT_LIVE_ALPHA
    assert alpha == 0.5


def test_hybrid_alpha_explicit_env_override(monkeypatch):
    """Explicit HYBRID_RETRIEVAL_ALPHA override takes precedence over provider defaults."""
    from backend.retrieval import get_hybrid_alpha
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("HYBRID_RETRIEVAL_ALPHA", "0.75")

    alpha = get_hybrid_alpha()
    assert alpha == 0.75

    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("HYBRID_RETRIEVAL_ALPHA", "0.33")

    alpha_gemini = get_hybrid_alpha()
    assert alpha_gemini == 0.33


def test_is_live_embedding_provider_helper(monkeypatch):
    """is_live_embedding_provider helper accurately distinguishes live from offline."""
    from backend.embeddings import is_live_embedding_provider
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    assert is_live_embedding_provider() is True

    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    assert is_live_embedding_provider() is True

    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    assert is_live_embedding_provider() is False
