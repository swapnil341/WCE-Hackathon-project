"""
Central configuration for the RAG pipeline.
All constants and settings live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vectorstore"

PDF_PATH = str(RAW_DATA_DIR / "Psychology2e_WEB.pdf")
CHUNKS_JSON_PATH = str(PROCESSED_DATA_DIR / "chunks.json")

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # change to "cuda" if GPU is available

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_psychology_textbook"
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR)

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K = 5                   # number of chunks to retrieve
DENSE_WEIGHT = 0.6          # weight for semantic / dense search score
SPARSE_WEIGHT = 0.4         # weight for BM25 / keyword search score

# ── NVIDIA LLM ─────────────────────────────────────────────────────────────────
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
