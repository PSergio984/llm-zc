"""
rag_pgvector.py — RAG with vector search via PostgreSQL + pgvector.

Subclasses RAGBase to use pgvector (cosine distance via <=> operator)
for document retrieval instead of keyword search.
"""

from dotenv import load_dotenv
from openai import OpenAI
import psycopg
from embedder import Embedder
from rag_helper import RAGBase


def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"


class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]


def main():
    DSN = "postgresql://postgres:pswd@localhost:5433/faq"

    load_dotenv()
    openai_client = OpenAI()

    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(DSN)

    model = Embedder()

    vector_assistant = RAGPgVector(
        embedder=model,
        conn=conn,
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

    conn.close()


if __name__ == "__main__":
    main()
