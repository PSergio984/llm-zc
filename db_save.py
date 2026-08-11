"""
db_save.py — Persist an LLMCallRecord into the conversations table.

Lesson 05 of the monitoring module: every assistant call (CLI or chat
app) is saved to Postgres so usage can be tracked over time.

save_conversation(record, question, course) inserts one row built from
a live LLMCallRecord (metrics.py) plus the raw user question and the
course the assistant serves. The question is passed separately from the
record's prompt: the prompt is the full text sent to the model, while
the question is what the user actually typed — keeping them apart means
we always know what was asked.

The id column is SERIAL, so Postgres assigns it; RETURNING id hands it
back so feedback can be attached to the right conversation later.

Follows the llm-zoomcamp lesson "Storing Data in PostgreSQL":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/05-database.md

Usage (from assistant.py / app.py):
  conversation_id = save_conversation(assistant.last_call, user_input, "llm-zoomcamp")
"""

from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE


def save_conversation(record, question, course):
    """Insert one LLMCallRecord and return its new conversation id."""
    # Timestamp this save with the same local timezone as the table column
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, course, model, instructions, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    course,
                    record.model,
                    record.instructions,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            # SERIAL id assigned by Postgres, returned for later feedback
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id
