import os
from unittest.mock import MagicMock
import pytest
import httpx

from backend.models import SearchPassageResponse, Citation, RAGQueryResponse
from backend.rag import (
    generate_rag_answer,
    get_generation_provider,
    _call_gemini_generation,
    _call_openai_generation,
    _get_gemini_api_key,
    _get_openai_api_key,
    RAG_SYSTEM_PROMPT,
)


def _make_dummy_passage(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    document_title: str = "Corporate Travel Policy",
    page_number: int = 2,
    text: str = "The maximum allowable domestic per diem for meals is capped at $75 per day.",
    score: float = 0.85,
) -> SearchPassageResponse:
    return SearchPassageResponse(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        page_number=page_number,
        chunk_index=0,
        text=text,
        score=score,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Test 1: Provider Selection Logic
# ---------------------------------------------------------------------------

def test_generation_provider_selection_offline(monkeypatch):
    """GENERATION_PROVIDER=offline selects offline mode even if API keys are present."""
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")
    monkeypatch.setenv("OPENAI_API_KEY", "real-looking-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "real-looking-gemini-key")

    assert get_generation_provider() == "offline"


def test_generation_provider_selection_gemini(monkeypatch):
    """GENERATION_PROVIDER=gemini selects gemini provider."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    assert get_generation_provider() == "gemini"


def test_generation_provider_selection_openai(monkeypatch):
    """GENERATION_PROVIDER=openai selects openai provider."""
    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    assert get_generation_provider() == "openai"


def test_generation_provider_invalid_raises_value_error(monkeypatch):
    """Invalid GENERATION_PROVIDER raises ValueError."""
    monkeypatch.setenv("GENERATION_PROVIDER", "invalid_provider")
    with pytest.raises(ValueError, match="Unsupported GENERATION_PROVIDER"):
        get_generation_provider()


# ---------------------------------------------------------------------------
# Test 2: Gemini Successful Generation and Citation Extraction (Mocked)
# ---------------------------------------------------------------------------

def test_gemini_generation_success(monkeypatch):
    """Verifies Gemini generateContent request format, header, and response parsing."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-generation-key")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

    passage = _make_dummy_passage()

    captured_url = None
    captured_headers = None
    captured_payload = None

    def mock_post(*args, **kwargs):
        nonlocal captured_url, captured_headers, captured_payload
        captured_url = args[1] if len(args) > 1 else kwargs.get("url", "")
        captured_headers = kwargs.get("headers", {})
        captured_payload = kwargs.get("json", {})

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "The maximum domestic per diem for meals is capped at $75 per day [source: chunk-1]."
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 25,
                "totalTokenCount": 145,
            },
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the daily meal per diem cap?",
        passages=[passage],
    )

    assert "gemini-2.5-flash:generateContent" in captured_url
    assert captured_headers.get("x-goog-api-key") == "test-gemini-generation-key"
    assert captured_payload["systemInstruction"]["parts"][0]["text"] == RAG_SYSTEM_PROMPT
    assert "<passage id=\"chunk-1\"" in captured_payload["contents"][0]["parts"][0]["text"]

    assert response.insufficient_evidence is False
    assert "$75 per day" in response.answer
    assert response.model_used == "gemini-2.5-flash"
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "chunk-1"
    assert response.citations[0].document_title == "Corporate Travel Policy"
    assert response.token_usage == {
        "prompt_tokens": 120,
        "completion_tokens": 25,
        "total_tokens": 145,
    }
    assert response.estimated_cost_usd is not None


# ---------------------------------------------------------------------------
# Test 3: Gemini Abstention and Refusal Handling
# ---------------------------------------------------------------------------

def test_gemini_abstention_handling(monkeypatch):
    """When Gemini returns an abstention / insufficient evidence statement, citations are cleared."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    passage = _make_dummy_passage()

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "I do not have sufficient information in the provided documents to answer this question."
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 15,
                "totalTokenCount": 115,
            },
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the travel per diem?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert "sufficient information" in response.answer.lower()
    assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Test 4: Defense-in-depth: Hallucination Override on Insufficient Evidence
# ---------------------------------------------------------------------------

def test_gemini_defense_in_depth_hallucination_override(monkeypatch):
    """If evidence sufficiency check determines evidence is absent, but model hallucinates, system enforces abstention."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    # Passage is about travel per diem, question asks about gym membership
    passage = _make_dummy_passage()

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Gym memberships are reimbursed up to $50 per month [source: chunk-1]."
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "totalTokenCount": 120},
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the reimbursement limit for personal gym memberships and yoga classes?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert len(response.citations) == 0
    assert "insufficient" in response.answer.lower()


# ---------------------------------------------------------------------------
# Test 5: Citation Validation Rejects Hallucinated Chunk IDs
# ---------------------------------------------------------------------------

def test_gemini_citation_validation_rejects_hallucinated_ids(monkeypatch):
    """Gemini citations referencing fake chunk IDs are filtered out."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    passage = _make_dummy_passage(chunk_id="real-chunk-1")

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Meals are capped at $75 [source: real-chunk-1] and flights must be economy [source: fake-chunk-999]."
                            }
                        ]
                    }
                }
            ],
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the meal allowance and flight policy?",
        passages=[passage],
    )

    assert response.insufficient_evidence is False
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "real-chunk-1"
    assert all(c.chunk_id != "fake-chunk-999" for c in response.citations)


# ---------------------------------------------------------------------------
# Test 6: Missing Gemini API Key Handling
# ---------------------------------------------------------------------------

def test_gemini_missing_api_key_returns_sanitized_error(monkeypatch):
    """When GEMINI_API_KEY is missing, returns sanitized error without crashing."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr("backend.rag.GEMINI_API_KEY", "")

    passage = _make_dummy_passage()

    response = generate_rag_answer(
        question="What is the meal allowance?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert "missing or empty" in response.answer.lower()
    assert response.model_used == "gemini-2.5-flash"
    assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Test 7: HTTP 429 Rate Limit Handling
# ---------------------------------------------------------------------------

def test_gemini_http_429_rate_limit(monkeypatch):
    """HTTP 429 rate limit is caught and returns sanitized user message."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    passage = _make_dummy_passage()

    def mock_post(*args, **kwargs):
        req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
        resp = httpx.Response(status_code=429, request=req)
        raise httpx.HTTPStatusError("Resource exhausted", request=req, response=resp)

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the meal allowance?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert "rate limit exceeded" in response.answer.lower()
    assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Test 8: HTTP 401/403 Authentication Error Handling
# ---------------------------------------------------------------------------

def test_gemini_http_auth_error_sanitization(monkeypatch):
    """HTTP 401/403 error is sanitized and does not leak API keys."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-gemini-key-12345")

    passage = _make_dummy_passage()

    def mock_post(*args, **kwargs):
        req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
        resp = httpx.Response(status_code=403, request=req)
        raise httpx.HTTPStatusError("API key not valid", request=req, response=resp)

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the meal allowance?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert "403" in response.answer
    assert "test-secret-gemini-key" not in response.answer
    assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Test 9: HTTP Timeout Handling
# ---------------------------------------------------------------------------

def test_gemini_timeout_handling(monkeypatch):
    """Timeout exception is caught and returns a clear, sanitized response."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    passage = _make_dummy_passage()

    def mock_post(*args, **kwargs):
        raise httpx.ReadTimeout("Request timed out")

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the meal allowance?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert "timed out" in response.answer.lower()
    assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Test 10: Malformed API Response Handling
# ---------------------------------------------------------------------------

def test_gemini_malformed_response(monkeypatch):
    """Missing candidates list in response raises error and returns safe error message."""
    monkeypatch.setenv("GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    passage = _make_dummy_passage()

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"something_unexpected": True}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the meal allowance?",
        passages=[passage],
    )

    assert response.insufficient_evidence is True
    assert "connection error" in response.answer.lower()
    assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Test 11: Offline Mode Generates Zero Network Calls
# ---------------------------------------------------------------------------

def test_offline_mode_zero_network_calls(monkeypatch):
    """When GENERATION_PROVIDER=offline, zero HTTP network calls are made."""
    monkeypatch.setenv("GENERATION_PROVIDER", "offline")
    monkeypatch.setenv("OPENAI_API_KEY", "real-key-1")
    monkeypatch.setenv("GEMINI_API_KEY", "real-key-2")

    passage = _make_dummy_passage()

    network_call_made = False

    def mock_post(*args, **kwargs):
        nonlocal network_call_made
        network_call_made = True
        raise AssertionError("Network call was made in offline mode!")

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the maximum allowable domestic per diem for meals?",
        passages=[passage],
    )

    assert network_call_made is False
    assert response.insufficient_evidence is False
    assert "$75" in response.answer
    assert "(local-grounded)" in response.model_used


# ---------------------------------------------------------------------------
# Test 12: OpenAI Provider Regression (Mocked)
# ---------------------------------------------------------------------------

def test_openai_provider_generation_regression(monkeypatch):
    """Verifies existing OpenAI provider path continues to work properly."""
    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("backend.rag.OPENAI_API_KEY", "test-openai-key")

    passage = _make_dummy_passage()

    captured_url = None
    captured_headers = None
    captured_payload = None

    def mock_post(*args, **kwargs):
        nonlocal captured_url, captured_headers, captured_payload
        captured_url = args[1] if len(args) > 1 else kwargs.get("url", "")
        captured_headers = kwargs.get("headers", {})
        captured_payload = kwargs.get("json", {})

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "The daily meal cap is $75 per day [source: chunk-1]."
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 18,
                "total_tokens": 98,
            },
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    response = generate_rag_answer(
        question="What is the meal allowance?",
        passages=[passage],
    )

    assert "api.openai.com" in captured_url
    assert "Bearer test-openai-key" in captured_headers.get("Authorization")
    assert response.insufficient_evidence is False
    assert "$75 per day" in response.answer
    assert len(response.citations) == 1
    assert response.token_usage == {
        "prompt_tokens": 80,
        "completion_tokens": 18,
        "total_tokens": 98,
    }
