"""
Streamlit Web Interface for the Psychology RAG Pipeline
"""

from src.utils.config import (
    DENSE_WEIGHT,
    EMBEDDING_MODEL_NAME,
    NVIDIA_MODEL,
    SPARSE_WEIGHT,
    TOP_K,
)
from src.retrieval.search import hybrid_search
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate_answer
from src.generation.formatter import format_output
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Psychology RAG Assistant",
    page_icon="",
    layout="centered",
)

# Import our pipeline after setting page config

# ── Header & Sidebar ──────────────────────────────────────────────────────────

st.title("Psychology RAG Assistant")
st.markdown(
    "Ask questions about the OpenStax Psychology textbook. "
    "Answers are strictly grounded in retrieved context."
)

with st.sidebar:
    st.header("Pipeline Configuration")
    st.markdown(f"**LLM:** `{NVIDIA_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Top K Chunks:** `{TOP_K}`")
    st.markdown(f"**Dense Weight:** `{DENSE_WEIGHT}`")
    st.markdown(f"**Sparse Weight:** `{SPARSE_WEIGHT}`")

    st.divider()
    st.markdown("Built with ChromaDB, rank-bm25, and NVIDIA NIMs.")

# ── Chat State ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you study today?", "refs": None}
    ]

# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the assistant message has references, render them in an expander
        if msg.get("refs"):
            with st.expander("View Source References"):
                st.table(msg["refs"])

# ── User Input & Pipeline Execution ───────────────────────────────────────────

if prompt := st.chat_input("What is classical conditioning?"):
    # 1. Add user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "refs": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        with st.spinner("Retrieving context & generating answer..."):
            try:
                # Run the exact same logic as main.py's run_query
                retrieved = hybrid_search(prompt, top_k=TOP_K)
                llm_prompt = build_prompt(prompt, retrieved)
                answer_text = generate_answer(llm_prompt)

                output = format_output(answer_text, retrieved)
                final_answer = output["answer"]
                refs = output["references"]

                # Show answer
                answer_placeholder.markdown(final_answer)

                # Show references
                with st.expander("View Source References"):
                    st.table(refs)

                # Save to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "refs": refs
                })

            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
