"""
vector_search.py — Vector search over FAQ document embeddings.

Loads the embedding matrix (X.npy) and documents (documents.pkl) produced
by embed_dataset.py, then performs cosine-similarity search via dot product.

Usage:
    python vector_search.py
"""

import numpy as np
import pickle
from embedder import Embedder


def load_data():
    # Load the pre-computed embedding matrix: rows = documents, cols = 384-d vectors
    X = np.load("X.npy")
    # Load the original document dicts in the same row order as X
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return X, documents


def main():
    print("Loading embedding matrix and documents...")
    X, documents = load_data()
    print(f"Loaded matrix: {X.shape[0]} documents x {X.shape[1]} dimensions\n")

    embedder = Embedder()

    # Step 1: embed the query into the same vector space
    query = "Can I still join the course after the start date?"
    print(f"Query: {query}\n")
    v_query = embedder.encode(query)

    # Step 2: compute cosine similarity via dot product
    # all-MiniLM-L6-v2 produces unit vectors, so dot product ≈ cosine similarity
    scores = X.dot(v_query)

    # Step 3: find the single best-matching document
    best_idx = np.argmax(scores)
    print(f"Best match — index {best_idx}, score {scores[best_idx]:.4f}")
    print(f"Document: {documents[best_idx]}\n")

    # Step 4: retrieve top 5 results using argsort with negation trick
    # Negating turns argsort's ascending sort into a descending one
    top5 = np.argsort(-scores)[:5]
    print("Top 5 results:")
    for idx in top5:
        print(f"  score {scores[idx]:.4f} | {documents[idx]['question']}")
        print(f"           {documents[idx]['answer'][:100]}...")
        print()


if __name__ == "__main__":
    main()