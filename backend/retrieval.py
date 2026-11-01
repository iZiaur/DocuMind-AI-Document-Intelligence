"""
DocuMind AI - Hybrid Retrieval Engine.

Combines dense vector similarity with normalized lexical relevance (term overlap & frequency)
to achieve high-precision retrieval across both semantic queries and exact-term queries.

Key Architecture:
- In production with live OpenAI embeddings: Vector similarity (semantic abstraction) and
  lexical matching (exact acronyms, identifiers, numbers) are balanced (alpha ~ 0.5).
- In offline development / evaluation mode: Embeddings are deterministic SHA-256 projections
  which exhibit cryptographic avalanche effects and lack semantic proximity. Lexical relevance
  is therefore prioritized (alpha ~ 0.2) to accurately retrieve relevant passages based on
  exact keywords (e.g. 'Severity-1', '15 minutes', 'SOC 2', 'per diem') rather than hash collisions.
"""

import os
import re
from typing import List, Dict, Any, Optional

from backend.embeddings import OPENAI_API_KEY, is_live_embedding_provider

# Standard English stopwords (omits domain-significant terms like 'per', numbers, acronyms)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "and", "or", "that", "this", "it",
    "as", "be", "what", "how", "when", "where", "which", "who", "does",
    "do", "did", "must", "can", "could", "should", "would", "about",
}

# Default alpha weights (vector weight vs lexical weight)
DEFAULT_LIVE_ALPHA = 0.5     # 50% vector, 50% lexical when live embeddings active
DEFAULT_OFFLINE_ALPHA = 0.2  # 20% vector, 80% lexical in offline mode (SHA-256 pseudo-vectors)


def tokenize_lexical(text: str) -> List[str]:
    """
    Tokenizes text while preserving domain terms, numbers, acronyms, and hyphenated terms.
    Examples:
    - 'Severity-1' -> ['severity-1', 'severity', '1']
    - 'SOC 2' -> ['soc', '2']
    - '$75' -> ['75']
    - '15 minutes' -> ['15', 'minutes']
    """
    if not text:
        return []

    # Find alphanumeric words, hyphenated compounds, and underscores
    raw_tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b", text.lower())
    tokens: List[str] = []

    for token in raw_tokens:
        tokens.append(token)
        # Also include sub-components of hyphenated tokens to support partial matching
        if "-" in token or "_" in token:
            parts = re.split(r"[-_]", token)
            for part in parts:
                if part and part not in tokens:
                    tokens.append(part)

    return tokens


def compute_lexical_score(query: str, document_text: str) -> float:
    """
    Calculates a bounded lexical relevance score in [0.0, 1.0].

    Uses sublinear term frequency saturation:
        sat(term) = tf / (tf + 1.2)
    Applies higher weights to numeric values, hyphenated identifiers, and acronyms.
    Awards bigram phrase bonuses for consecutive query term pairs.
    Handles empty queries and empty document text safely.
    """
    if not query or not document_text:
        return 0.0

    q_tokens = tokenize_lexical(query)
    # Filter stopwords unless query consists entirely of stopwords
    meaningful = [t for t in q_tokens if t not in STOPWORDS]
    if not meaningful:
        meaningful = q_tokens
    if not meaningful:
        return 0.0

    doc_lower = document_text.lower()
    doc_tokens = tokenize_lexical(document_text)

    # Count term frequencies in document
    doc_freqs: Dict[str, int] = {}
    for dt in doc_tokens:
        doc_freqs[dt] = doc_freqs.get(dt, 0) + 1

    total_weight = 0.0
    weighted_score = 0.0

    # 1. Unigram scoring with saturation
    for term in meaningful:
        # Boost weight for terms with digits or hyphens (e.g. 'severity-1', '15', '75')
        if any(c.isdigit() for c in term) or "-" in term:
            weight = 1.5
        elif len(term) <= 4 and term.isupper():
            weight = 1.5
        else:
            weight = 1.0

        count = doc_freqs.get(term, 0)
        sat = count / (count + 1.2) if count > 0 else 0.0

        weighted_score += weight * sat
        total_weight += weight

    # 2. Bigram phrase bonus (e.g. 'per diem', '15 minutes', 'root cause')
    query_words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS and len(w) > 1]
    if len(query_words) >= 2:
        for i in range(len(query_words) - 1):
            phrase = f"{query_words[i]} {query_words[i+1]}"
            total_weight += 1.5
            if phrase in doc_lower:
                weighted_score += 1.5

    if total_weight <= 0:
        return 0.0

    raw_score = weighted_score / total_weight
    return round(max(0.0, min(1.0, float(raw_score))), 4)


def get_hybrid_alpha() -> float:
    """
    Resolves the hybrid weighting parameter alpha (vector weight).
    Can be explicitly configured via HYBRID_RETRIEVAL_ALPHA environment variable.
    Defaults to 0.5 when a live embedding provider is active ('gemini' or 'openai'),
    and 0.2 in offline mode (deterministic SHA-256 projections).
    """
    env_alpha = os.getenv("HYBRID_RETRIEVAL_ALPHA", "").strip()
    if env_alpha:
        try:
            val = float(env_alpha)
            return max(0.0, min(1.0, val))
        except ValueError:
            pass

    # Use live vs offline default based on active embedding provider
    return DEFAULT_LIVE_ALPHA if is_live_embedding_provider() else DEFAULT_OFFLINE_ALPHA


def compute_hybrid_score(
    vector_similarity: float,
    lexical_similarity: float,
    alpha: Optional[float] = None,
) -> float:
    """
    Combines vector similarity and lexical score using linear combination:
        hybrid_score = (alpha * vector_sim) + ((1 - alpha) * lexical_sim)
    Ensures result is clamped in [0.0, 1.0] and rounded to 4 decimal places.
    """
    effective_alpha = get_hybrid_alpha() if alpha is None else max(0.0, min(1.0, float(alpha)))
    combined = (effective_alpha * vector_similarity) + ((1.0 - effective_alpha) * lexical_similarity)
    return round(max(0.0, min(1.0, float(combined))), 4)
