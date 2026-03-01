"""
Prompt builder – constructs a strict context-grounded prompt from
retrieved chunks and the user's question.
"""

from typing import Any


SYSTEM_INSTRUCTIONS = (
    "You are a strictly grounded assistant answering questions about a psychology textbook.\n"
    "Answer ONLY using the provided context below.\n"
    "If the answer is not present in the context, respond with:\n"
    "\"The answer is not available in the provided document.\"\n"
    "Do NOT use external knowledge, assumptions, or educated guesses.\n"
    "Reference the section name and page number in your answer where applicable."
)


def build_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Build a structured prompt for the LLM.

    Args:
        question:         User's question.
        retrieved_chunks: Output from hybrid_search().

    Returns:
        A single formatted prompt string.
    """
    # Format each context chunk with metadata header
    context_blocks: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        header = (
            f"[Chunk {i} | Section: {chunk['section']} | "
            f"Page: {chunk['page']} | Hash: {chunk['hash']}]"
        )
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "Context:\n"
        "---------\n"
        f"{context_str}\n\n"
        "Question:\n"
        "---------\n"
        f"{question}\n\n"
        "Answer:"
    )
    return prompt
