"""
rag_vector.py — RAG with persistent vector search (sqlitesearch).

Subclasses RAGBase to swap keyword search for vector search.
Uses sqlitesearch's VectorSearchIndex for persistent on-disk
ANN search, avoiding re-embedding on every startup.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import VectorSearchIndex
from embedder import Embedder
from rag_helper import RAGBase


class RAGVector(RAGBase):
    """
    RAG pipeline using vector search instead of keyword search.

    Takes an extra 'embedder' argument for encoding the query into a
    vector, then delegates the vector search to the VectorSearchIndex.
    """

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )


def main():
    DB_PATH = "faq_vectors.db"

    load_dotenv()
    openai_client = OpenAI()

    if os.path.exists(DB_PATH):
        print(f"Opening persistent vector index from {DB_PATH}...")
        vs_index = VectorSearchIndex(
            keyword_fields=["course"],
            mode="ivf",
            db_path=DB_PATH,
        )
    else:
        print(f"{DB_PATH} not found. Run ingest_vector.py first to build the index.")
        return

    model = Embedder()

    vector_assistant = RAGVector(
        embedder=model,
        index=vs_index,
        llm_client=openai_client,
    )

    queries = [
        "the program has already begun, can I still sign up?",
        "I just found out about the program, can I still sign up?",
        "How do I run Ollama? I made a typo",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        answer = vector_assistant.rag(q)
        print(f"Answer: {answer}")

    vs_index.close()


if __name__ == "__main__":
    main()