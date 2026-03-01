"""
Vector store module – persists chunk embeddings + metadata into ChromaDB.
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from src.utils.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)

logger = logging.getLogger(__name__)


def get_collection(reset: bool = False) -> chromadb.Collection:
    """
    Return (or create) the persistent ChromaDB collection.

    Args:
        reset: If True, delete the existing collection and start fresh.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("Existing collection deleted for fresh ingestion.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(
    chunks: list[dict[str, Any]],
    reset: bool = False,
) -> chromadb.Collection:
    """
    Store embedded chunks into ChromaDB.

    Each chunk must have:
        text, section, page_number, chunk_id, embedding, hash

    Args:
        chunks: Fully embedded chunk dicts.
        reset:  Whether to wipe the existing collection first.

    Returns:
        The ChromaDB Collection object.
    """
    collection = get_collection(reset=reset)

    # Check if already populated (skip re-ingestion)
    existing = collection.count()
    if existing > 0 and not reset:
        logger.info(
            "Collection already contains %d vectors – skipping ingestion.", existing
        )
        return collection

    ids, embeddings, documents, metadatas = [], [], [], []

    for chunk in chunks:
        ids.append(chunk["hash"])          # deterministic ID from hash
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(
            {
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "page_number": int(chunk["page_number"]),
                "hash": chunk["hash"],
            }
        )

    # ChromaDB add() in batches of 5000 to avoid memory issues
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i: i + batch_size],
            embeddings=embeddings[i: i + batch_size],
            documents=documents[i: i + batch_size],
            metadatas=metadatas[i: i + batch_size],
        )
        logger.info("Stored batch %d/%d", i // batch_size +
                    1, -(-len(ids) // batch_size))

    logger.info("Total vectors stored: %d", collection.count())
    return collection


def load_collection() -> chromadb.Collection:
    """Load the existing ChromaDB collection (must be already ingested)."""
    return get_collection(reset=False)
