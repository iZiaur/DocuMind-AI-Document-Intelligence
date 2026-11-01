"""
Lightweight Production-Style Observability for DocuMind AI.
Provides:
1. Versioned pricing catalog and estimated API cost calculation.
2. Error categorization and sensitive data redaction.
3. Structured logging for request lifecycle tracking.
4. Database persistence helper for RequestMetric records.
5. In-app dashboard metrics aggregation (avg latency, P95, error rate, token totals, cost).
"""

import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

logger = logging.getLogger("documind.observability")

# ---------------------------------------------------------------------------
# Feature 3: Versioned Pricing Configuration
# ---------------------------------------------------------------------------
# Pricing catalog is versioned and models standard per-1M-token pricing rates.
# To update prices, add or modify entries here and bump PRICING_CONFIG_VERSION.
PRICING_CONFIG_VERSION = "2024-08-01"

MODEL_PRICING_CATALOG: Dict[str, Dict[str, float]] = {
    # Rates expressed in USD per 1,000,000 tokens
    "gpt-4o-mini": {
        "input_price_per_1m": 0.150,    # $0.150 per 1M prompt tokens
        "output_price_per_1m": 0.600,   # $0.600 per 1M completion tokens
    },
    "gpt-4o": {
        "input_price_per_1m": 2.500,    # $2.500 per 1M prompt tokens
        "output_price_per_1m": 10.000,  # $10.000 per 1M completion tokens
    },
    "text-embedding-3-small": {
        "input_price_per_1m": 0.020,    # $0.020 per 1M tokens
        "output_price_per_1m": 0.0,
    },
    "gemini-2.5-flash": {
        "input_price_per_1m": 0.150,    # $0.150 per 1M prompt tokens
        "output_price_per_1m": 0.600,   # $0.600 per 1M completion tokens
    },
    "gemini-embedding-001": {
        "input_price_per_1m": 0.020,    # $0.020 per 1M tokens
        "output_price_per_1m": 0.0,
    },
}


def calculate_estimated_cost(
    model_name: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    pricing_catalog: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Optional[float], str]:
    """
    Calculates estimated API cost based on versioned model pricing.
    Formula:
        estimated_cost = prompt_tokens * (input_price / 1_000_000)
                       + completion_tokens * (output_price / 1_000_000)

    When prompt_tokens or completion_tokens are unavailable (None):
    Returns (None, PRICING_CONFIG_VERSION).
    Never silently fabricates numbers or replaces missing tokens with 0.
    """
    if prompt_tokens is None or completion_tokens is None:
        return None, PRICING_CONFIG_VERSION

    catalog = pricing_catalog or MODEL_PRICING_CATALOG

    # Match model family by prefix or substring
    matched_pricing = None
    for key, spec in catalog.items():
        if key in model_name:
            matched_pricing = spec
            break

    if not matched_pricing:
        matched_pricing = catalog.get("gpt-4o-mini", {"input_price_per_1m": 0.150, "output_price_per_1m": 0.600})

    input_cost = prompt_tokens * (matched_pricing["input_price_per_1m"] / 1_000_000.0)
    output_cost = completion_tokens * (matched_pricing["output_price_per_1m"] / 1_000_000.0)
    total_cost = round(input_cost + output_cost, 7)

    return total_cost, PRICING_CONFIG_VERSION


# ---------------------------------------------------------------------------
# Feature 4: Failure Tracking & Redaction
# ---------------------------------------------------------------------------

def categorize_error(exc: Exception) -> Tuple[str, str]:
    """
    Categorizes exceptions into standardized failure categories and returns safe, user-friendly messages.
    STRICT SECURITY RULE: Never exposes API keys, filesystem paths, document text, or internal stack traces.
    """
    import httpx
    from fastapi import HTTPException

    # 1. HTTP error responses from LLM provider
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code if exc.response is not None else 500
        if code == 401:
            return "invalid_api_key", "Authentication with the LLM provider failed. Check API key."
        elif code == 429:
            return "rate_limit_exceeded", "LLM provider rate limit exceeded. Please retry shortly."
        elif code in (502, 503, 504):
            return "provider_unavailable", "The LLM provider service is temporarily unavailable."
        else:
            return "provider_error", f"LLM provider error (status {code})."

    # 2. Timeouts and connectivity
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "provider_timeout", "The request to the LLM provider timed out. Please retry."

    if isinstance(exc, httpx.ConnectError):
        return "provider_unavailable", "Unable to establish connection to the LLM provider."

    # 3. Application HTTP exceptions (e.g. document not found)
    if isinstance(exc, HTTPException):
        if exc.status_code == 404:
            return "document_not_found", "The requested document was not found or access is denied."
        elif exc.status_code == 403:
            return "forbidden", "Access denied for this resource."
        elif exc.status_code == 401:
            return "unauthorized", "Authentication credentials required."
        elif exc.status_code in (400, 422):
            safe_detail = exc.detail if isinstance(exc.detail, str) else "Invalid input parameter."
            if "C:" in safe_detail or "/" in safe_detail or "sk-" in safe_detail:
                safe_detail = "Invalid request parameter."
            return "invalid_input", safe_detail
        safe_detail = exc.detail if isinstance(exc.detail, str) else "Request error."
        # Sanitize against accidental path or key leakage
        if "C:" in safe_detail or "/" in safe_detail or "sk-" in safe_detail:
            safe_detail = "Invalid request parameter."
        return "request_error", safe_detail

    # 4. Ingestion / embedding failures
    exc_type_name = type(exc).__name__.lower()
    if "embedding" in exc_type_name or "vector" in exc_type_name:
        return "embedding_failure", "Failed to generate embeddings for query."

    if "retrieval" in exc_type_name or "database" in exc_type_name:
        return "retrieval_failure", "Failed to retrieve documents from vector database."

    # Default fallback
    return "internal_error", "An unexpected error occurred while processing the question."


# ---------------------------------------------------------------------------
# Feature 6: Structured Logging
# ---------------------------------------------------------------------------

def log_request_event(
    request_id: str,
    event: str,
    duration_ms: float,
    status: str,
    user_id: Optional[int] = None,
    error_category: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    estimated_cost_usd: Optional[float] = None,
) -> None:
    """
    Emits structured key-value log entries for monitoring and audit ingestion.
    Example:
    [OBSERVABILITY] request_id=req-123 event=rag_query status=success duration_ms=124.5 user_id=1 prompt_tokens=250 completion_tokens=40 estimated_cost_usd=0.00006
    """
    parts = [
        "[OBSERVABILITY]",
        f"request_id={request_id}",
        f"event={event}",
        f"status={status}",
        f"duration_ms={duration_ms:.2f}",
    ]
    if user_id is not None:
        parts.append(f"user_id={user_id}")
    if error_category:
        parts.append(f"error_category={error_category}")
    if prompt_tokens is not None:
        parts.append(f"prompt_tokens={prompt_tokens}")
    if completion_tokens is not None:
        parts.append(f"completion_tokens={completion_tokens}")
    if estimated_cost_usd is not None:
        parts.append(f"estimated_cost_usd={estimated_cost_usd:.7f}")

    log_line = " ".join(parts)
    if status == "success":
        logger.info(log_line)
    else:
        logger.warning(log_line)


# ---------------------------------------------------------------------------
# Feature 1 & 5: Record Metric & Compute Dashboard Aggregates
# ---------------------------------------------------------------------------

def record_request_metric(
    db: Session,
    request_id: str,
    user_id: Optional[int],
    document_count: int,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    status: str,
    error_category: Optional[str] = None,
    safe_message: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    estimated_cost_usd: Optional[float] = None,
    pricing_version: Optional[str] = None,
) -> Any:
    """
    Persists a RequestMetric record into the database.
    Does NOT store raw question, document text, or API secrets.
    """
    from backend.models import RequestMetric

    metric = RequestMetric(
        request_id=request_id,
        user_id=user_id,
        document_count=document_count,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        status=status,
        error_category=error_category,
        safe_message=safe_message,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        pricing_version=pricing_version or PRICING_CONFIG_VERSION,
    )
    try:
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric
    except Exception as err:
        db.rollback()
        logger.error("Failed to persist request metric: %s", type(err).__name__)
        return None


def calculate_p95_latency(latencies: List[float]) -> float:
    """
    Computes the 95th percentile latency from a list of measurements.
    Uses nearest-rank method without external dependencies.
    """
    if not latencies:
        return 0.0
    sorted_lats = sorted(latencies)
    idx = max(0, int(math.ceil(0.95 * len(sorted_lats))) - 1)
    return round(sorted_lats[idx], 2)


def get_monitoring_dashboard_summary(db: Session, limit_recent: int = 25) -> Dict[str, Any]:
    """
    Aggregates in-app observability statistics from stored RequestMetric rows.
    Computes total requests, success/fail counts, error rate, average and P95 latency,
    token totals, unknown token counts, and total estimated cost.
    Strictly preserves unknown / null states for unmetered requests and distinguishes
    them from genuine measured zero values.
    """
    from backend.models import RequestMetric

    records: List[RequestMetric] = db.query(RequestMetric).order_by(desc(RequestMetric.timestamp)).all()
    total_requests = len(records)

    if total_requests == 0:
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "error_rate": 0.0,
            "avg_total_latency_ms": 0.0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_generation_latency_ms": 0.0,
            "p95_total_latency_ms": 0.0,
            "token_usage": {
                "total_prompt_tokens": None,
                "total_completion_tokens": None,
                "total_tokens": None,
                "requests_with_unknown_tokens": 0,
                "known_token_requests": 0,
                "unknown_token_requests": 0,
                "is_partial": False,
            },
            "total_estimated_cost_usd": None,
            "cost_is_partial": False,
            "known_cost_requests": 0,
            "unknown_cost_requests": 0,
            "pricing_version": PRICING_CONFIG_VERSION,
            "recent_requests": [],
        }

    successful_count = sum(1 for r in records if r.status == "success")
    failed_count = sum(1 for r in records if r.status == "failed")
    error_rate = round(failed_count / total_requests, 4)

    total_lats = [r.total_latency_ms for r in records]
    ret_lats = [r.retrieval_latency_ms for r in records]
    gen_lats = [r.generation_latency_ms for r in records]

    avg_total_latency = round(sum(total_lats) / len(total_lats), 2)
    avg_ret_latency = round(sum(ret_lats) / len(ret_lats), 2)
    avg_gen_latency = round(sum(gen_lats) / len(gen_lats), 2)
    p95_latency = calculate_p95_latency(total_lats)

    # Token usage aggregation: preserve null when usage is unknown; never replace with zero
    known_prompt_records = [r.prompt_tokens for r in records if r.prompt_tokens is not None]
    known_comp_records = [r.completion_tokens for r in records if r.completion_tokens is not None]
    known_token_records = [r.total_tokens for r in records if r.total_tokens is not None]
    unknown_token_count = sum(1 for r in records if r.total_tokens is None)
    known_token_count = len(known_token_records)

    if known_token_count == 0:
        total_prompt = None
        total_comp = None
        total_tok = None
        token_is_partial = False
    else:
        total_prompt = sum(known_prompt_records)
        total_comp = sum(known_comp_records)
        total_tok = sum(known_token_records)
        token_is_partial = unknown_token_count > 0

    # Total estimated cost: preserve null when cost is unknown; never replace with zero
    known_cost_records = [r.estimated_cost_usd for r in records if r.estimated_cost_usd is not None]
    unknown_cost_count = sum(1 for r in records if r.estimated_cost_usd is None)
    known_cost_count = len(known_cost_records)

    if known_cost_count == 0:
        total_cost = None
        cost_is_partial = False
    else:
        total_cost = round(sum(known_cost_records), 6)
        cost_is_partial = unknown_cost_count > 0

    recent_records = records[:limit_recent]
    recent_list = [
        {
            "request_id": r.request_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else datetime.now(timezone.utc).isoformat(),
            "user_id": r.user_id,
            "document_count": r.document_count,
            "retrieval_latency_ms": r.retrieval_latency_ms,
            "generation_latency_ms": r.generation_latency_ms,
            "total_latency_ms": r.total_latency_ms,
            "status": r.status,
            "error_category": r.error_category,
            "safe_message": r.safe_message,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "estimated_cost_usd": r.estimated_cost_usd,
            "pricing_version": r.pricing_version or PRICING_CONFIG_VERSION,
        }
        for r in recent_records
    ]

    return {
        "total_requests": total_requests,
        "successful_requests": successful_count,
        "failed_requests": failed_count,
        "error_rate": error_rate,
        "avg_total_latency_ms": avg_total_latency,
        "avg_retrieval_latency_ms": avg_ret_latency,
        "avg_generation_latency_ms": avg_gen_latency,
        "p95_total_latency_ms": p95_latency,
        "token_usage": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
            "total_tokens": total_tok,
            "requests_with_unknown_tokens": unknown_token_count,
            "known_token_requests": known_token_count,
            "unknown_token_requests": unknown_token_count,
            "is_partial": token_is_partial,
        },
        "total_estimated_cost_usd": total_cost,
        "cost_is_partial": cost_is_partial,
        "known_cost_requests": known_cost_count,
        "unknown_cost_requests": unknown_cost_count,
        "pricing_version": PRICING_CONFIG_VERSION,
        "recent_requests": recent_list,
    }
