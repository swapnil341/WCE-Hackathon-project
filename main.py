"""
main.py – Full RAG pipeline orchestrator.

Usage:
    # Step 1 – Ingest the PDF (only run once, or after PDF update)
    python main.py ingest

    # Step 2 – Ask a question
    python main.py query "What is classical conditioning?"

    # Step 3 – Both at once (useful for first-time setup)
    python main.py ingest-and-query "What is classical conditioning?"
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv

from src.generation.formatter import format_output, pretty_print
from src.generation.llm_client import generate_answer
from src.generation.prompt_builder import build_prompt
from src.ingestion.book_processor import parse_and_chunk_openstax
from src.ingestion.embeddings import embed_chunks, load_model
from src.ingestion.vector_store import store_chunks
from src.retrieval.search import hybrid_search
from src.utils.config import (
    CHUNKS_JSON_PATH,
    PDF_PATH,
    PROCESSED_DATA_DIR,
    TOP_K,
)

# ── Logging ────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Pipeline Stages ────────────────────────────────────────────────────────────

def run_ingestion(pdf_path: str = PDF_PATH, reset: bool = False) -> None:
    """
    Full ingestion pipeline:
      1. Parse and chunk the PDF
      2. Generate embeddings + hashes
      3. Store in ChromaDB
    """
    logger.info("=== INGESTION PIPELINE STARTED ===")

    # 1. Parse + chunk
    logger.info("Step 1/3 – Parsing PDF: %s", pdf_path)
    chunks = parse_and_chunk_openstax(pdf_file_path=pdf_path)
    logger.info("  Chunks created: %d", len(chunks))

    # Save raw chunks to disk
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        # Don't serialize embeddings here – save them after embedding step
        json.dump([{k: v for k, v in c.items() if k != "embedding"}
                  for c in chunks], f, indent=2)

    # 2. Generate embeddings
    logger.info("Step 2/3 – Generating embeddings …")
    model = load_model()
    chunks = embed_chunks(chunks, model)

    # 3. Store in ChromaDB
    logger.info("Step 3/3 – Storing vectors in ChromaDB …")
    store_chunks(chunks, reset=reset)

    logger.info("=== INGESTION COMPLETE ===")


def run_query(question: str, top_k: int = TOP_K) -> dict:
    """
    Full retrieval + generation pipeline:
      1. Embed the query
      2. Hybrid search (dense + BM25)
      3. Build strict context prompt
      4. Call NVIDIA LLM
      5. Format + return output
    """
    logger.info("=== RETRIEVAL + GENERATION PIPELINE STARTED ===")

    # 1+2. Retrieve relevant chunks
    logger.info("Retrieving top-%d chunks for question: %s",
                top_k, question[:80])
    retrieved = hybrid_search(question, top_k=top_k)

    # 3. Build prompt
    prompt = build_prompt(question, retrieved)

    # 4. Generate answer
    answer = generate_answer(prompt)

    # 5. Format output
    output = format_output(answer, retrieved)
    pretty_print(output)

    logger.info("=== PIPELINE COMPLETE ===")
    return output


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    command = args[0].lower()

    if command == "ingest":
        run_ingestion()

    elif command == "query":
        if len(args) < 2:
            print("Usage: python main.py query \"<your question>\"")
            sys.exit(1)
        question = " ".join(args[1:])
        run_query(question)

    elif command == "ingest-and-query":
        if len(args) < 2:
            print("Usage: python main.py ingest-and-query \"<your question>\"")
            sys.exit(1)
        run_ingestion()
        question = " ".join(args[1:])
        run_query(question)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
