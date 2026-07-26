"""
03-ground-truth-batch.py — Generate ground truth for all FAQ documents.

Follows the llm-zoomcamp lesson "Generating Ground Truth for All Documents":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/03-ground-truth-batch.md

For each document, the LLM generates 5 questions.  Results are saved as
a CSV file under data/ground_truth-new.csv.

Usage:
  python 04-evaluation/lessons/03-ground-truth-batch.py

Requires OPENAI_API_KEY in .env.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ingest import load_faq_data
from evaluation_utils import (
    llm_structured_retry,
    calc_total_price,
    map_progress,
)

# ── 1. Load the FAQ documents ────────────────────────────────────────────
# Each document already has a stable 'id' field that will serve as the
# ground-truth label.  We filter to llm-zoomcamp only to keep cost low.

load_dotenv()
openai_client = OpenAI()

documents = load_faq_data()
documents = [doc for doc in documents if doc["course"] == "llm-zoomcamp"]

print(f"Number of LLM Zoomcamp documents: {len(documents)}")

# ── 2. Define the structured output model and instructions ────────────────
# Using Pydantic structured output means the LLM returns a consistent
# Python object rather than free text we'd need to parse.

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


# ── 3. Processing function for a single document ─────────────────────────
# Returns (list_of_records, usage_object) so we can track both the ground
# truth data and the cost for each API call.

def generate_ground_truth(doc):
    """Generate ground truth records for one FAQ document.

    Converts the document to JSON for the LLM, calls the API with retry
    logic, then creates one record per generated question pairing it with
    the document's ID.

    Returns:
        Tuple of (list[dict], usage_object).
    """
    # Serialize the document so the LLM sees the full Q&A record
    user_prompt = json.dumps(doc)

    # Use the retry variant so a transient API error doesn't fail the batch
    out, usage = llm_structured_retry(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions,
    )

    # Each generated question maps to this document's ID as the correct answer
    results = [
        {"question": q, "document": doc["id"]}
        for q in out.questions
    ]

    return results, usage


# ── 4. Sequential: first 5 documents (demonstration) ─────────────────────
# Runs one LLM call after another — simple to understand but wastes time
# waiting on the network.  Good for a quick smoke test.

from tqdm.auto import tqdm

ground_truth = []
usages = []

print("\nGenerating ground truth for first 5 documents (sequential)...")
for doc in tqdm(documents[:5]):
    records, usage = generate_ground_truth(doc)
    ground_truth.extend(records)
    usages.append(usage)

print(f"Sequential run: {len(ground_truth)} records generated")

# ── 5. Parallel: all documents ───────────────────────────────────────────
# ThreadPoolExecutor lets requests overlap — each call spends most of its
# time waiting for OpenAI's response, so we can run several concurrently.
# 6 workers is safe; more would risk hitting rate limits.

print(f"\nGenerating ground truth for all {len(documents)} documents (parallel)...")

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, documents, generate_ground_truth)

# Split each (records, usage) tuple back into two flat lists
ground_truth = []
usages = []

for records, usage in results:
    ground_truth.extend(records)
    usages.append(usage)

print(f"Total records generated: {len(ground_truth)}")

# ── 6. Calculate total cost ──────────────────────────────────────────────
# Summing all usage objects gives us the full price for the run.
# Expected: ~$0.06 for 118 documents at gpt-5.4-mini pricing.

total_cost = calc_total_price(usages)
print(f"Total cost: ${total_cost:.6f}")

# ── 7. Save to CSV ───────────────────────────────────────────────────────
# The CSV has two columns: 'question' (the generated query) and 'document'
# (the ID of the FAQ record that contains the correct answer).
# This file is the ground-truth dataset used by later evaluation lessons.

df_ground_truth = pd.DataFrame(ground_truth)
print(f"\nDataFrame shape: {df_ground_truth.shape}")
print(df_ground_truth.head())

import os

os.makedirs("data", exist_ok=True)
df_ground_truth.to_csv("data/ground_truth-new.csv", index=False)
print("Saved to data/ground_truth-new.csv")
