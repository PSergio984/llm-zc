"""
ingest_vector.py — Build a persistent vector index with sqlitesearch.

Embeds the FAQ dataset using sentence-transformers and stores the
vectors + documents in a sqlitesearch VectorSearchIndex on disk.
This avoids re-embedding the dataset on every startup.

Usage:
    python ingest_vector.py
"""

from sqlitesearch import VectorSearchIndex
from ingest import load_faq_data
from embedder import Embedder
from tqdm.auto import tqdm
import numpy as np


def build_texts(documents):
    texts = []
    for doc in documents:
        text = doc["question"] + " " + doc["answer"]
        texts.append(text)
    return texts


def encode_in_batches(model, texts, batch_size=50):
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_vectors = model.encode(batch)
        vectors.extend(batch_vectors)
    return vectors


def main():
    DB_PATH = "faq_vectors.db"
    COURSE = "llm-zoomcamp"

    print("Loading FAQ documents...")
    documents = load_faq_data()
    print(f"Loaded {len(documents)} total documents.")

    docs_filtered = [doc for doc in documents if doc["course"] == COURSE]
    print(f"{COURSE}: {len(docs_filtered)} documents to index.")

    print("Building combined texts (question + answer)...")
    texts = build_texts(docs_filtered)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = Embedder()

    print("Encoding texts in batches...")
    vectors = encode_in_batches(embedder.model, texts, batch_size=50)

    X = np.array(vectors)
    print(f"\nEmbedding matrix shape: {X.shape}")

    print(f"\nCreating persistent vector index at {DB_PATH}...")
    vs_index = VectorSearchIndex(
        keyword_fields=["course"],
        mode="ivf",
        db_path=DB_PATH,
    )

    vs_index.fit(X, docs_filtered)
    vs_index.close()

    print(f"\nDone. Index saved to {DB_PATH} with {len(docs_filtered)} documents.")


if __name__ == "__main__":
    main()
