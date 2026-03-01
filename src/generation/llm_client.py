"""
NVIDIA LLM client – calls the NVIDIA NIM API (OpenAI-compatible endpoint)
with a structured prompt and returns the model's response.
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    NVIDIA_API_BASE,
    NVIDIA_MODEL,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    """Create an OpenAI-compatible client pointing at the NVIDIA NIM endpoint."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(base_url=NVIDIA_API_BASE, api_key=api_key)


def generate_answer(prompt: str) -> str:
    """
    Send a prompt to the NVIDIA-hosted LLM and return the answer text.

    Args:
        prompt: Fully formatted context + question prompt.

    Returns:
        The model's response as a string.
    """
    client = _get_client()
    logger.info("Calling NVIDIA LLM: %s", NVIDIA_MODEL)

    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    answer = response.choices[0].message.content or ""
    logger.info("LLM response received (%d chars).", len(answer))
    return answer.strip()
