import os
import math
import hashlib
from typing import List
import httpx
from dotenv import load_dotenv

load_dotenv()

# Embedding Model Configuration
# Standard model: OpenAI text-embedding-3-small (1536 dimensions)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


def generate_deterministic_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Generates a deterministic, normalized unit vector of fixed dimension from text.
    Used for local testing, offline development, and environments without an external API key.
    Ensures mathematical consistency without runtime dependencies on external APIs.
    """
    # Seed pseudo-random values using SHA-256 digests
    base_hash = hashlib.sha256(text.encode("utf-8")).digest()
    vector: List[float] = []

    for i in range(dimension):
        # Derive float from hashed bytes and position
        byte_idx = i % len(base_hash)
        val = (base_hash[byte_idx] + (i * 31)) % 256
        # Center around 0
        vector.append(float(val - 128))

    # L2 normalize so cosine distance behaves predictably
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0:
        vector = [x / norm for x in vector]
    else:
        vector[0] = 1.0

    return vector


def get_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for a single text passage.
    Calls OpenAI text-embedding-3-small if OPENAI_API_KEY is configured;
    otherwise falls back to deterministic local unit vectors.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        text = "empty"

    if OPENAI_API_KEY:
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": text,
                        "model": EMBEDDING_MODEL,
                    },
                )
                res.raise_for_status()
                data = res.json()
                embedding = data["data"][0]["embedding"]
                if len(embedding) == EMBEDDING_DIMENSION:
                    return embedding
        except Exception as err:
            print(f"[DocuMind AI Embeddings] OpenAI API call failed ({err}); falling back to local provider.")

    return generate_deterministic_embedding(text, EMBEDDING_DIMENSION)


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Batch embedding generation for multiple text chunks.
    """
    if not texts:
        return []

    # Filter/clean texts
    cleaned = [t.replace("\n", " ").strip() or "empty" for t in texts]

    if OPENAI_API_KEY:
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": cleaned,
                        "model": EMBEDDING_MODEL,
                    },
                )
                res.raise_for_status()
                data = res.json()
                # Sort by index to maintain ordering
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
        except Exception as err:
            print(f"[DocuMind AI Embeddings] Batch OpenAI call failed ({err}); using local generator.")

    return [generate_deterministic_embedding(t, EMBEDDING_DIMENSION) for t in cleaned]
