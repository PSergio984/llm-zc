"""
minsearch_vector.py — Vector search using the minsearch VectorSearch library.

Loads the pre-computed embedding matrix (X.npy) and documents (documents.pkl),
indexes them into a minsearch VectorSearch, then demonstrates searching with
and without keyword filtering by course.

Usage:
    python minsearch_vector.py
"""

import numpy as np
import pickle
from minsearch import VectorSearch
from embedder import Embedder


def load_data():
    X = np.load("X.npy")
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return X, documents


def main():
    print("Loading embedding matrix and documents...")
    X, documents = load_data()
    print(f"Loaded matrix: {X.shape[0]} documents x {X.shape[1]} dimensions\n")

    # Create a VectorSearch index with 'course' as a keyword field for filtering
    vindex = VectorSearch(keyword_fields=["course"])
    vindex.fit(X, documents)

    embedder = Embedder()

    # --- Search without filtering ---
    query = "I just discovered the course. Can I still join it?"
    print(f"Query: {query}\n")
    query_vector = embedder.encode(query)

    results = vindex.search(query_vector, num_results=5)

    print("Top 5 results (unfiltered):")
    for i, doc in enumerate(results):
        print(f"  {i+1}. [{doc['course']}] {doc['question']}")
        print(f"     {doc['answer'][:120]}...")
        print()

    # --- Search with course filter ---
    results_filtered = vindex.search(
        query_vector,
        filter_dict={"course": "llm-zoomcamp"},
        num_results=5,
    )

    print("Top 5 results (filtered to llm-zoomcamp):")
    for i, doc in enumerate(results_filtered):
        print(f"  {i+1}. [{doc['course']}] {doc['question']}")
        print(f"     {doc['answer'][:120]}...")
        print()


if __name__ == "__main__":
    main()