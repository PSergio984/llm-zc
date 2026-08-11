"""
db_query.py — Read saved conversations back out of Postgres.

Lesson 06 of the monitoring module: the dashboard runs on this data.
Instead of working with raw row tuples (where you must remember that
column 4 is the model), each row is converted back into the same
LLMCallRecord dataclass used for live calls.

  - row_to_record(row)     → map a DB row tuple onto LLMCallRecord fields
  - get_conversations(limit) → most recent conversations, newest first

Ordering: by timestamp DESC. There is no index on timestamp (only on
id); ids increase over time anyway, so ordering by id would be faster
at scale — with a handful of rows it does not matter.

Follows the llm-zoomcamp lesson "Querying Data":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/06-querying.md

Usage:
  uv run python db_query.py
"""

from dataclasses import dataclass

from db_init import get_db_connection
from metrics import LLMCallRecord


def row_to_record(row):
    """Convert one database row (tuple) into an LLMCallRecord."""
    # Column order matches the SELECT in get_conversations:
    # 0=id, 1=question, 2=answer, 3=course, 4=model, 5=instructions,
    # 6=prompt, 7-9=tokens, 10=response_time, 11=cost, 12=timestamp
    return LLMCallRecord(
        model=row[4],
        prompt=row[6],
        instructions=row[5],
        answer=row[2],
        prompt_tokens=row[7],
        completion_tokens=row[8],
        total_tokens=row[9],
        response_time=row[10],
        cost=row[11],
        timestamp=row[12],
    )


def get_conversations(limit=10):
    """Return the `limit` most recent conversations as LLMCallRecords."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, course, model,
                       instructions, prompt,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # Convert raw tuples into the familiar dataclass shape
    return [row_to_record(row) for row in rows]


if __name__ == "__main__":
    records = get_conversations()
    for record in records:
        print(record)
