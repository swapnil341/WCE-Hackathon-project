"""
Query handler – embed a user question for semantic retrieval.
"""

import logging

from sentence_transformers import SentenceTransformer

from src.utils.config import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once per process
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model for retrieval: %s",
                    EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    return _model


def embed_query(query: str) -> list[float]:
    """
    Convert a plain-text query into a dense embedding vector.

    Args:
        query: User's natural-language question.

    Returns:
        Embedding as a list of floats.
    """
    model = get_model()
    logger.info("Embedding query: %s", query[:80])
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()
