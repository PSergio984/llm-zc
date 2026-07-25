"""
ingest_pgvector.py — Embed FAQ data and insert into PostgreSQL with pgvector.

Connects to a pgvector-enabled Postgres instance (run docker-compose up first),
creates a documents table with a vector(384) column, and inserts every FAQ
document alongside its embedding.
"""

from tqdm.auto import tqdm
import psycopg
from ingest import load_faq_data
from embedder import Embedder


def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"


def main():
    DSN = "postgresql://postgres:pswd@localhost:5433/faq"

    print("Loading FAQ documents...")
    documents = load_faq_data()
    print(f"Loaded {len(documents)} total documents.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = Embedder()

    texts = [doc["question"] + " " + doc["answer"] for doc in documents]

    print("Encoding texts...")
    batch_size = 50
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_vectors = embedder.model.encode(batch)
        vectors.extend(batch_vectors)

    print(f"Produced {len(vectors)} vectors of dimension {vectors[0].shape[0]}")

    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(DSN)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute("""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            course TEXT,
            section TEXT,
            question TEXT,
            answer TEXT,
            embedding vector(384)
        )
    """)

    print("Inserting documents with embeddings...")
    for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
        conn.execute(
            """
            INSERT INTO documents (course, section, question, answer, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (doc["course"], doc["section"], doc["question"], doc["answer"],
             vec_to_str(vec))
        )

    conn.commit()

    print("Creating HNSW index for faster search...")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding
        ON documents USING hnsw (embedding vector_cosine_ops)
    """)

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"Done. {count} documents inserted into pgvector.")

    conn.close()


if __name__ == "__main__":
    main()
