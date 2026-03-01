"""
Hybrid search module – combines dense (ChromaDB) and sparse (BM25) retrieval,
merges scores, and returns the top-K most relevant chunks.
"""

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from src.ingestion.vector_store import load_collection
from src.retrieval.query_handler import embed_query
from src.utils.config import DENSE_WEIGHT, SPARSE_WEIGHT, TOP_K

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokeniser for BM25."""
    return text.lower().split()


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ── Public API ─────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> list[dict[str, Any]]:
    """
    Perform hybrid dense + sparse retrieval for a query.

    Args:
        query:        Natural-language question.
        top_k:        Number of results to return.
        dense_weight: Weight for the semantic similarity score.
        sparse_weight: Weight for the BM25 keyword score.

    Returns:
        List of result dicts sorted by combined score (descending):
        [
          {
            "text": str,
            "section": str,
            "page": int,
            "hash": str,
            "chunk_id": str,
            "score": float,
          },
          ...
        ]
    """
    collection = load_collection()

    # Fetch all stored documents and metadata for BM25
    # ChromaDB does not expose a simple "get all", so we use a large limit
    total = collection.count()
    if total == 0:
        raise RuntimeError(
            "Vector store is empty. Run the ingestion pipeline first."
        )

    all_docs = collection.get(include=["documents", "metadatas"])
    raw_texts: list[str] = all_docs["documents"]        # type: ignore
    raw_meta: list[dict] = all_docs["metadatas"]        # type: ignore
    doc_ids: list[str] = all_docs["ids"]                # type: ignore

    # ── Dense search ──────────────────────────────────────────────────────────
    query_vec = embed_query(query)
    dense_results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k * 3, total),    # over-fetch for reranking
        include=["documents", "metadatas", "distances"],
    )
    dense_docs: list[str] = dense_results["documents"][0]       # type: ignore
    dense_meta: list[dict] = dense_results["metadatas"][0]      # type: ignore
    # Chroma returns cosine *distance* (0=identical, 2=opposite)
    # Convert to similarity: sim = 1 - dist/2  → [0, 1]
    # type: ignore
    dense_distances: list[float] = dense_results["distances"][0]
    dense_sim = [1.0 - d / 2.0 for d in dense_distances]

    # Build a lookup: hash → dense score
    dense_score_map: dict[str, float] = {}
    for meta, sim in zip(dense_meta, dense_sim):
        dense_score_map[meta["hash"]] = sim

    # ── Sparse (BM25) search ──────────────────────────────────────────────────
    tokenised_corpus = [_tokenize(t) for t in raw_texts]
    bm25 = BM25Okapi(tokenised_corpus)
    bm25_raw_scores: list[float] = bm25.get_scores(_tokenize(query)).tolist()
    bm25_scores = _normalise(bm25_raw_scores)

    # Build a lookup: hash → BM25 score
    bm25_score_map: dict[str, float] = {}
    for meta, score in zip(raw_meta, bm25_scores):
        bm25_score_map[meta["hash"]] = score

    # ── Hybrid scoring ────────────────────────────────────────────────────────
    # Gather candidate hashes: all from dense + BM25 top-N
    bm25_top_indices = sorted(
        range(len(bm25_raw_scores)), key=lambda i: bm25_raw_scores[i], reverse=True
    )[: top_k * 3]
    candidate_hashes: set[str] = set(dense_score_map.keys())
    candidate_hashes.update(raw_meta[i]["hash"] for i in bm25_top_indices)

    # Build final scored candidates
    results: list[dict[str, Any]] = []
    hash_to_text = {m["hash"]: raw_texts[i] for i, m in enumerate(raw_meta)}
    hash_to_meta = {m["hash"]: m for m in raw_meta}

    for h in candidate_hashes:
        d_score = dense_score_map.get(h, 0.0)
        b_score = bm25_score_map.get(h, 0.0)
        combined = dense_weight * d_score + sparse_weight * b_score
        meta = hash_to_meta[h]
        results.append(
            {
                "text": hash_to_text[h],
                "section": meta.get("section", "unknown"),
                "page": int(meta.get("page_number", 0)),
                "hash": h,
                "chunk_id": meta.get("chunk_id", ""),
                "score": round(combined, 4),
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    top_results = results[:top_k]
    logger.info("Retrieved %d chunks for query: %s",
                len(top_results), query[:60])
    return top_results
