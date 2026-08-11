"""
db_feedback.py — Save user (and later judge) feedback on answers.

Lesson 08 of the monitoring module: execution metrics (time, tokens,
cost) don't tell us whether an answer was good. The people using the
system know, so the chat app records thumbs-up / thumbs-down and stores
them here.

save_feedback() inserts one row into the feedback table:

  - conversation_id → which answer the feedback refers to (FK to the
                      conversations table)
  - source          → "user" for human clicks, "judge" for the LLM
                      evaluator added in a later lesson
  - relevance / explanation → reserved for the built-in judge later
  - score           → +1 thumbs up, -1 thumbs down

The signal is noisy (mis-clicks, generous raters), but it feeds the
evaluation dataset and gives the dashboard a clear wave-of-thumbs-down
sign when something breaks.

Follows the llm-zoomcamp lesson "User Feedback":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/08-user-feedback.md

Usage (from app.py):
  save_feedback(conversation_id, "user", score=1)
"""

from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE


def save_feedback(conversation_id, source, relevance=None,
                  explanation=None, score=None):
    """Insert one feedback row for the given conversation."""
    # Timestamp this save with the same local timezone as the table column
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (conversation_id, source, relevance,
                 explanation, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()
