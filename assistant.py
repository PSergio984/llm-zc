"""
assistant.py — Factory for the course Q&A assistant (monitoring module).

Builds the RAG pipeline that answers questions about DataTalks courses.
This is the "something to monitor" from lesson 02 of the monitoring
module: a search → prompt → answer loop backed by the course FAQ.

The factory:
  1. loads the OpenAI/Groq API credentials from .env
  2. downloads the FAQ dataset and builds an in-memory minsearch index
  3. wraps the index in RAGWithMetrics (from metrics.py) so every
     LLM call records response time, token usage and cost

Using RAGWithMetrics instead of plain RAGBase means the assistant is
fully instrumented from the start — the chat app (app.py) can read the
last call's metrics off `assistant.last_call`.

Follows the llm-zoomcamp lesson "Assistant":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/02-assistant-setup.md

Usage:
  python assistant.py ["your question"]
  python assistant.py "How do I join the course?"
"""

import sys

from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_faq_data, build_index
from metrics import RAGWithMetrics
from db_save import save_conversation


def create_assistant():
    """Build a ready-to-use RAG assistant with metrics capture."""
    # Load OPENAI_API_KEY / OPENAI_BASE_URL from .env (Groq-compatible endpoint)
    load_dotenv()

    # Step 1: download the course FAQ dataset
    documents = load_faq_data()
    # Step 2: build an in-memory minsearch index over it
    index = build_index(documents)

    # Step 3: wrap index + LLM client in the metrics-instrumented RAG
    return RAGWithMetrics(
        index=index,
        llm_client=OpenAI(),
    )


if __name__ == "__main__":
    assistant = create_assistant()

    # Default question; override with the first CLI argument
    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)

    # Persist this call to the monitoring Postgres (lesson 05)
    save_conversation(assistant.last_call, query, "llm-zoomcamp")
