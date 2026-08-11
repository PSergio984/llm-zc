"""
db_init.py — Create the conversations table in the monitoring Postgres.

Lesson 05 of the monitoring module: the metrics captured by metrics.py
vanish when the app closes, so every conversation gets persisted to a
dedicated PostgreSQL database (run only for monitoring — nothing else
in the system touches it).

This module provides:
  - DB_TIMEZONE       → the local timezone, used for timestamps so
                        Grafana lines data up on its time axis later
  - get_db_connection → connect to Postgres, credentials from env vars
                        with defaults matching the docker-compose
                        `postgres` service (course-assistant-pg)
  - init_db(drop)     → create the `conversations` table (one row per
                        LLM call: tokens, response time, cost, ...)

The table is created once; data survives container restarts because the
compose service uses a named volume. Re-run init_db only when changing
the schema — with drop=True it wipes existing data first.

Follows the llm-zoomcamp lesson "Storing Data in PostgreSQL":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/05-database.md

Usage:
  docker compose up -d postgres
  uv run python db_init.py
"""

import os
import psycopg
from datetime import datetime

# Local timezone, attached to every saved record (TIMESTAMPTZ column)
DB_TIMEZONE = datetime.now().astimezone().tzinfo
print(f"Using timezone: {DB_TIMEZONE}")


def get_db_connection():
    """Open a connection to the monitoring Postgres."""
    # Env vars override the defaults, which match the compose service
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "course_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def init_db(drop=False):
    """Create the conversations table; with drop=True, drop it first."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                # Wipes all existing data — only for schema changes
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    course TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized")
