"""
embed_dataset.py — Embed the full FAQ dataset into a vector matrix.

Loads every document from the course FAQ, builds a combined text
per document (question + answer), then encodes them in batches
using sentence-transformers. The result is a numpy matrix of shape
(n_documents, 384) ready for vector search.

Usage:
    python embed_dataset.py
"""

# Import the FAQ data loader from our ingest module
from ingest import load_faq_data
# Import our embedder wrapper that wraps all-MiniLM-L6-v2
from embedder import Embedder
# tqdm gives us a progress bar so we can watch batch encoding
from tqdm.auto import tqdm
import numpy as np


def build_texts(documents):
    """
    Combine each document's question and answer into a single text string.
    This way a query can match against either the question or the answer
    when we search the embedding space later.
    """
    texts = []
    for doc in documents:
        # Simple concatenation with a space separator
        text = doc["question"] + " " + doc["answer"]
        texts.append(text)
    return texts


def encode_in_batches(model, texts, batch_size=50):
    """
    Encode texts in batches to avoid running out of memory and
    to let us see progress. Returns a list of numpy vectors.
    """
    vectors = []
    # Step through the texts in chunks of batch_size
    for i in tqdm(range(0, len(texts), batch_size)):
        # Slice out this batch
        batch = texts[i:i + batch_size]
        # Encode the batch — sentence-transformers handles batching internally
        batch_vectors = model.encode(batch)
        # Accumulate into our results list
        vectors.extend(batch_vectors)
    return vectors


def main():
    """
    Full pipeline: load data → build texts → encode → numpy array.
    """
    print("Loading FAQ documents...")
    documents = load_faq_data()
    print(f"Loaded {len(documents)} documents.")

    # Step 1: Build one combined text per document
    print("Building combined texts (question + answer)...")
    texts = build_texts(documents)

    # Step 2: Load the sentence-transformers model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = Embedder()

    # Step 3: Encode everything in batches
    print("Encoding texts in batches...")
    vectors = encode_in_batches(embedder.model, texts, batch_size=50)

    # Step 4: Convert the list of vectors to a 2D numpy array (matrix).
    # Rows = documents, columns = embedding dimensions (384).
    X = np.array(vectors)
    print(f"\nEmbedding matrix shape: {X.shape}")
    print(f"  => {X.shape[0]} documents")
    print(f"  => {X.shape[1]} dimensions per vector")

    # Sanity check: first vector preview
    print(f"\nFirst vector (first 5 values): {X[0][:5]}")


if __name__ == "__main__":
    main()
