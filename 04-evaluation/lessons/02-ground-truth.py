"""
02-ground-truth.py — Generate ground truth data for evaluating search.

Follows the llm-zoomcamp lesson "Generating Ground Truth Data":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/02-ground-truth.md

For each FAQ document, the LLM generates 5 questions that the document
would answer.  The result is a ground-truth dataset where each question
is paired with the ID of the document that contains the correct answer.

Usage:
  python 04-evaluation/lessons/02-ground-truth.py

Requires OPENAI_API_KEY in .env.
"""

import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ingest import load_faq_data
from evaluation_utils import llm_structured, calc_price

# ── 1. Load the FAQ documents ────────────────────────────────────────────
# We need a set of documents where we know the correct answer for each query.
# The FAQ data has built-in IDs — each record becomes a ground-truth label.

load_dotenv()
openai_client = OpenAI()

documents = load_faq_data()

# Keep only the LLM Zoomcamp FAQ (118 docs) to limit cost and time
documents = [doc for doc in documents if doc["course"] == "llm-zoomcamp"]

print(f"Number of LLM Zoomcamp documents: {len(documents)}")

# Inspect the first document — note the id field used as ground-truth label
doc = documents[0]
print(f"\nDocument ID: {doc['id']}")
print(f"Question: {doc['question']}")
print(f"Answer: {doc['answer'][:200]}...")

# ── 2. Define the structured output model ────────────────────────────────
# Using Pydantic ensures the LLM returns a consistent structure we can
# process programmatically — no regex parsing of free-form text.

class Questions(BaseModel):
    """Schema for the LLM's structured output: a list of generated questions."""
    questions: list[str]

data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()

# ── 3. Generate questions for one document ───────────────────────────────
# We pass the document as JSON so the LLM sees the full Q&A record.
# The API returns a parsed Questions instance (not raw JSON).

user_prompt = json.dumps(doc)

result, usage = llm_structured(
    openai_client,
    data_gen_instructions,
    user_prompt,
    Questions,
)

print(f"\nGenerated questions for doc {doc['id']}:")
for q in result.questions:
    print(f"  • {q}")

# ── 4. Track cost ───────────────────────────────────────────────────────
# Each API call has a cost.  Tracking it helps budget before scaling up
# to the full dataset (lesson 03).

cost = calc_price(usage)

print(f"\nInput tokens:  {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Cost:          ${cost['total_cost']:.6f}")

# ── 5. Build ground truth records ───────────────────────────────────────
# Each record pairs a generated question with the document ID.
# Later, evaluation checks whether search retrieves this document ID.

records = [
    {"question": q, "document": doc["id"]}
    for q in result.questions
]

print(f"\nGround truth records for one document:")
for r in records:
    print(f"  Q: {r['question']}")
    print(f"  → doc: {r['document']}\n")
