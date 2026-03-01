"""
Output formatter – packages the LLM answer together with
structured references (section, page, hash) for final submission.
"""

from typing import Any


def format_output(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the final output dict containing the answer and structured references.

    Args:
        answer:           Raw text answer from the LLM.
        retrieved_chunks: Chunks returned by hybrid_search().

    Returns:
        Dict with 'answer' and 'references' keys.

    Example output:
        {
          "answer": "Classical conditioning is ...",
          "references": [
            {"section": "2.1_classical_conditioning", "page": 47, "hash": "abc123"}
          ]
        }
    """
    references = [
        {
            "section": chunk["section"],
            "page": chunk["page"],
            "hash": chunk["hash"],
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "references": references,
    }


def pretty_print(output: dict[str, Any]) -> None:
    """Print the formatted output to stdout in a human-readable way."""
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(output["answer"])
    print("\n" + "-" * 60)
    print("REFERENCES")
    print("-" * 60)
    for ref in output["references"]:
        print(
            f"  Section: {ref['section']:<40} "
            f"Page: {ref['page']:<5} "
            f"Hash: {ref['hash']}"
        )
    print("=" * 60)
