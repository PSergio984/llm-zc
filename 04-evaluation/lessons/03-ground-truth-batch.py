"""
03-ground-truth-batch.py — Generate ground truth for all FAQ documents.

What is ground truth?
  In search evaluation, "ground truth" means a set of queries where we already
  know which document has the correct answer.  Once we have that, we can send
  each query to our search engine and check whether the right document appears
  in the top-N results (hit-rate / recall).  Without ground truth we cannot
  measure whether our search works.

  This script uses the LLM itself to generate questions from each FAQ record.
  The record becomes the "correct answer" for those questions — that's our
  ground-truth label.

Pipeline:
  1. Load and filter FAQ data (llm-zoomcamp only to keep cost low)
  2. For each document, ask the LLM to produce 5 natural questions
  3. Pair each question with the document ID → ground-truth record
  4. Save all records as CSV for downstream evaluation

Sequential → Parallel progression:
  We first run a sequential loop on 5 documents (quick smoke test).
  Then we switch to ThreadPoolExecutor for all documents.  Each LLM call is
  I/O-bound (waiting on the network), so parallel workers let us overlap requests
  and finish ~6x faster than sequential.

Cost awareness:
  Every API call costs money.  We track token usage per call and sum the total
  so we know the budget consumed before scaling to a larger dataset.

Follows the llm-zoomcamp lesson "Generating Ground Truth for All Documents":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/03-ground-truth-batch.md

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
# Each FAQ record has a stable 'id' field.  When we generate a question from a
# record, that record's ID becomes the ground-truth label — the "correct answer"
# that the search engine should retrieve.
#
# Why filter to llm-zoomcamp only?
#   The full FAQ dataset has ~1380 documents across multiple courses (ml-zoomcamp,
#   data-engineering, etc.).  Generating 5 questions per doc would cost ~$0.60.
#   Filtering to 79 llm-zoomcamp docs costs ~$0.06 — enough to demonstrate the
#   evaluation pipeline without burning through API credits.

load_dotenv()
openai_client = OpenAI()

documents = load_faq_data()
documents = [doc for doc in documents if doc["course"] == "llm-zoomcamp"]

print(f"Number of LLM Zoomcamp documents: {len(documents)}")

# ── 2. Define the structured output model and instructions ────────────────
# Why structured output (Pydantic) instead of free-text + regex?
#   Free-text responses vary in format (bullet points, numbering, prefixes).
#   Parsing them is fragile and error-prone.  Structured output tells the API
#   to return a JSON object matching our Questions schema — we get a parsed
#   Python object with zero parsing overhead.
#
# The Questions model declares that the LLM must return a JSON object with
# a single key "questions" whose value is a list of strings.  The API enforces
# this schema at the model level, so we never have to handle malformed output.

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
# Each call returns two things: the generated records AND the token usage.
# We need usage for cost tracking — the ground-truth records are useless if
# we don't know what they cost to produce (budget planning for production).
#
# Why use llm_structured_retry instead of llm_structured?
#   In a batch of 79+ calls, a single transient failure (network blip, 5xx,
#   rate-limit hiccup) would crash the whole run.  The retry wrapper sleeps
#   1s, 2s, 4s between attempts (exponential backoff) and only raises after
#   exhausting 3 retries.  This makes the batch resilient to random failures
#   without complex error handling at the call site.

def generate_ground_truth(doc):
    """Generate ground truth records for one FAQ document.

    Converts the document to JSON for the LLM, calls the API with retry
    logic, then creates one record per generated question pairing it with
    the document's ID.

    Args:
        doc: FAQ record dict with 'id', 'question', 'answer', etc.

    Returns:
        Tuple of (list[dict], usage_object).
        Each dict: {"question": str, "document": str}
    """
    user_prompt = json.dumps(doc)

    out, usage = llm_structured_retry(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions,
    )

    results = [
        {"question": q, "document": doc["id"]}
        for q in out.questions
    ]

    return results, usage


# ── 4. Sequential: first 5 documents (demonstration) ─────────────────────
# Why do 5 docs sequentially first?
#   1. Quick smoke test — catches API errors, auth issues, or schema mismatches
#      before committing to the full batch.
#   2. Educational progression — sequential is easier to understand, then we
#      level up to parallel once the sequential version works.
#   3. Cost check — we can estimate the per-doc cost from 5 docs and extrapolate
#      to the full dataset before spending money.
#
# Trade-off: sequential is simple but slow.  Each call blocks until the LLM
# responds.  For 79 docs at ~2s/response that's ~2.5 minutes wasted on I/O wait.

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
# Why ThreadPoolExecutor instead of asyncio or multiprocessing?
#   The bottleneck is network I/O (waiting for OpenAI's API response), not CPU.
#   ThreadPoolExecutor is the simplest way to overlap I/O-bound work in Python.
#   asyncio would work but requires rewriting everything with async/await.
#   Multiprocessing adds overhead for no benefit here since we're not CPU-bound.
#
# Why 6 workers?
#   Each worker holds one TCP connection to the OpenAI API.  More workers means
#   more concurrent requests, which risks hitting rate limits (429 errors) or
#   overwhelming your network.  6 is a safe default for most providers.
#
# How map_progress works:
#   It submits every document to the pool immediately, then collects results
#   in input order while updating a tqdm progress bar.  This is the most
#   readable pattern for batch-processing with progress feedback.

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
# Typical cost: ~$0.06 for 79 llm-zoomcamp docs at gpt-5.4-mini pricing
# (5 questions each, ~300 input tokens + ~100 output tokens per doc).
#
# Why track cost explicitly?
#   Evaluation pipelines can run many LLM calls (this script = ~400 calls).
#   Without cost tracking, you can accidentally burn through API budgets.
#   calc_total_price gives a single number you can log, alert on, or compare
#   across runs to detect cost regressions.

total_cost = calc_total_price(usages)
print(f"Total cost: ${total_cost:.6f}")

# ── 7. Save to CSV ───────────────────────────────────────────────────────
# The CSV has two columns:
#   question  — the natural-language query (what a student would ask)
#   document  — the ID of the FAQ record that contains the correct answer
#
# This file is consumed by later evaluation lessons to compute:
#   - Hit-rate (is the correct doc in the top-N search results?)
#   - Recall (what fraction of correct docs are retrieved?)
#   - Mean Reciprocal Rank (how high is the correct doc ranked?)
#
# By generating ground truth from the FAQ data itself, we create a labeled
# evaluation set without manual annotation — a technique called "LLM-as-judge"
# data generation.  The quality depends on the prompt and model, but for
# search evaluation it produces high-quality labels at low cost.

df_ground_truth = pd.DataFrame(ground_truth)
print(f"\nDataFrame shape: {df_ground_truth.shape}")
print(df_ground_truth.head())

import os

os.makedirs("data", exist_ok=True)
df_ground_truth.to_csv("data/ground_truth-new.csv", index=False)
print(f"Saved {len(df_ground_truth)} records to data/ground_truth-new.csv")

# ── 8. Verify ──────────────────────────────────────────────────────────────

print(f"\nSummary:")
print(f"  Documents processed: {len(documents)}")
print(f"  Total ground truth records: {len(df_ground_truth)}")
print(f"  Total cost: ${total_cost:.6f}")
print(f"  Cost per record: ${total_cost / len(df_ground_truth):.6f}")
