"""
rag_vector.py — RAG with vector search over FAQ embeddings.

Subclasses RAGBase to swap keyword search for vector search.
Only the search method is overridden; build_prompt and llm are inherited.
"""

import numpy as np
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from minsearch import VectorSearch
from embedder import Embedder
from ingest import load_faq_data, build_index
from rag_helper import RAGBase


class RAGVector(RAGBase):
    """
    RAG pipeline using vector search instead of keyword search.

    Takes an extra 'embedder' argument for encoding the query into a
    vector, then delegates the vector search to the VectorSearch index.
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
    load_dotenv()
    openai_client = OpenAI()

    # --- Load pre-computed embeddings and documents ---
    print("Loading embedding matrix and documents...")
    X = np.load("X.npy")
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} documents, matrix shape {X.shape}")

    # --- Create vector index ---
    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, documents)

    # --- Create embedder ---
    model = Embedder()

    # --- Create vector RAG assistant ---
    vector_assistant = RAGVector(
        embedder=model,
        index=vindex,
        llm_client=openai_client,
    )

    # --- Query ---
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


if __name__ == "__main__":
    main()