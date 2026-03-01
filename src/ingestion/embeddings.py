"""
Embedding module – generates dense vector embeddings for text chunks
using sentence-transformers (all-MiniLM-L6-v2).
"""

import hashlib
import json
import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from src.utils.config import (
    CHUNKS_JSON_PATH,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def compute_chunk_hash(text: str) -> str:
    """Return a deterministic SHA-256 hash of normalised chunk text."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def load_model() -> SentenceTransformer:
    """Load the embedding model once and return it."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
) -> list[dict[str, Any]]:
    """
    Add 'embedding' and 'hash' fields to each chunk dict.

    Args:
        chunks: List of chunk dicts (must have 'text' key).
        model:  Pre-loaded SentenceTransformer; loaded lazily if not supplied.

    Returns:
        Same list with 'embedding' (list[float]) and 'hash' (str) fields added.
    """
    if model is None:
        model = load_model()

    texts = [c["text"] for c in chunks]
    logger.info("Generating embeddings for %d chunks …", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        chunk["hash"] = compute_chunk_hash(chunk["text"])

    logger.info("Embeddings generated successfully.")
    return chunks


def load_and_embed_chunks(model: SentenceTransformer | None = None) -> list[dict[str, Any]]:
    """Convenience helper: load chunks from disk and embed them."""
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embed_chunks(chunks, model)
