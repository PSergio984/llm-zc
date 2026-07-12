"""
Standalone data ingestion script.

Run this once to populate faq.db with the LLM Zoomcamp FAQ documents.
The query process (main.py) then opens the same file without re-fetching or re-indexing.

Usage:
    python ingest_sqlite.py
"""

import time
from sqlitesearch import TextSearchIndex
from ingest import load_faq_data

DB_PATH = "faq.db"
COURSE = "llm-zoomcamp"


def main():
    print("Fetching FAQ documents...")
    documents = load_faq_data()
    print(f"Loaded {len(documents)} total documents.")

    # Filter to just the target course
    docs_llm = [doc for doc in documents if doc["course"] == COURSE]
    print(f"{COURSE}: {len(docs_llm)} documents to index.")

    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=DB_PATH
    )

    for doc in docs_llm:
        index.add(doc)
        print(f"  Added: {doc['question'][:60]}...")
        time.sleep(0.5)  # simulate slow ingestion (remove in production)

    index.close()
    print(f"\nDone. Index saved to {DB_PATH}")


if __name__ == "__main__":
    main()
