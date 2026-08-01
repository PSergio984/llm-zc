"""
12-rag-answers.py - Generate RAG answers for every ground-truth question.

What this script does:
  Lessons 04-06 evaluated search quality in isolation: does the right
  document appear in the top-5 results?  Now we evaluate the full RAG
  pipeline - retrieval plus LLM generation together.

  This is the A -> Q -> A' setup:
    A  = the original answer in the FAQ (written by the course authors)
    Q  = a question generated from that answer (ground truth, lesson 03)
    A' = the answer produced by OUR RAG system for that question

  If A' is close to A, the RAG system is doing a good job.  The next
  lesson (13) formalizes that comparison with an LLM judge.

  This is still offline evaluation: we know the correct original answer
  for every question, so no human labelling is required.

Pipeline:
  1. Load ground truth questions and the FAQ documents
  2. Build the search index and a document-ID lookup table
  3. Run RAG (search + LLM) on every question, tracking token usage
  4. Pair each generated answer with the original FAQ answer
  5. Save everything to data/rag-answers-new.csv

Cost awareness:
  This lesson makes one real LLM call per question.  The course runs all
  395, but Groq's free tier caps llama models at 100k tokens/day (~40 calls
  with 5-doc context), so GT_LIMIT defaults to 40.  Set GT_LIMIT to
  len(ground_truth) when running against a paid tier or after quota resets.
  We track usage after every call and print the total cost at the end.

Follows the llm-zoomcamp lesson "Generating RAG Answers":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/12-rag-answers.md

Usage:
  python 04-evaluation/lessons/12-rag-answers.py

  Run from the repo root with both source dirs on the path, e.g.:
    $env:PYTHONPATH = ".;04-evaluation\code"
    python 04-evaluation/lessons/12-rag-answers.py

Requires an API key in .env (Groq-compatible endpoint for this run).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_faq_data, build_index
from evaluation_utils import RAGWithUsage, map_progress

# Number of ground-truth questions to process.  The full dataset is 395,
# but the free tier's daily token budget limits a single sitting to ~40.
GT_LIMIT = 40

# Model for this run.  llama-3.1-8b-instant is cheaper and faster than
# llama-3.3-70b and has its own fresh daily token budget on the free tier.
MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# 1. Load ground truth questions
# ---------------------------------------------------------------------------
# Same CSV as lessons 04-06: one row per question, with the ID of the
# document that contains the correct answer.  If the file is missing we
# download it from the course repo, exactly like lesson 04 does.

ground_truth_path = "data/ground_truth-new.csv"
if not os.path.exists(ground_truth_path):
    import urllib.request
    print("Ground truth file not found locally. Downloading from course repo...")
    os.makedirs("data", exist_ok=True)
    url = ("https://raw.githubusercontent.com/DataTalksClub/"
           "llm-zoomcamp/main/04-evaluation/data/ground_truth-new.csv")
    urllib.request.urlretrieve(url, ground_truth_path)
    print(f"Downloaded to {ground_truth_path}")

df_ground_truth = pd.read_csv(ground_truth_path)
ground_truth = df_ground_truth.to_dict(orient="records")

print(f"Loaded {len(ground_truth)} ground truth records")


# ---------------------------------------------------------------------------
# 2. Load FAQ documents and build the search index
# ---------------------------------------------------------------------------
# Only llm-zoomcamp documents are indexed, matching the course filter used
# when the ground truth was generated.

documents = load_faq_data()
documents = [doc for doc in documents if doc["course"] == "llm-zoomcamp"]

index = build_index(documents)

print(f"Loaded {len(documents)} llm-zoomcamp documents")


# ---------------------------------------------------------------------------
# 3. Document lookup table
# ---------------------------------------------------------------------------
# Each ground-truth record points at a document ID.  To compare the RAG
# answer (A') with the original answer (A) we need to fetch the original
# FAQ record by its ID - hence the dict.

doc_idx = {}
for doc in documents:
    doc_idx[doc["id"]] = doc

missing_ids = [rec["document"] for rec in ground_truth
               if rec["document"] not in doc_idx]
if missing_ids:
    print(f"WARNING: {len(missing_ids)} ground-truth document IDs are missing "
          f"from the current FAQ data (the dataset changed since the ground "
          f"truth was generated). They will get an empty original answer.")

# ---------------------------------------------------------------------------
# 4. Set up the RAG assistant
# ---------------------------------------------------------------------------
# RAGWithUsage (evaluation_utils.py) extends RAGBase from module 01 with
# token-usage tracking.  It uses the search boosts chosen during search
# tuning: question=1.0, answer=2.0, section=0.1.
#
# rag(question) runs the whole pipeline: search the FAQ, build a prompt
# with the retrieved context, and ask the LLM for a grounded answer.

load_dotenv()
openai_client = OpenAI()

assistant = RAGWithUsage(
    index=index,
    llm_client=openai_client,
    model=MODEL,
)

print(f"RAG assistant ready (model: {assistant.model})")


# ---------------------------------------------------------------------------
# 5. Demo: one question end to end
# ---------------------------------------------------------------------------
# Before looping over everything, run a single question and inspect:
#   - the generated answer (A')
#   - the cost of that one call
#   - the original FAQ answer (A)

if __name__ == "__main__":
    rec = ground_truth[0]
    question = rec["question"]
    doc_id = rec["document"]

    answer_llm = assistant.rag(question)
    original_doc = doc_idx.get(doc_id)
    answer_orig = original_doc["answer"] if original_doc else ""

    print("\n--- 5. Demo on the first question ---")
    print(f"  Question:   {question}")
    print(f"  RAG answer: {answer_llm}")
    print(f"  Cost of this call: ${assistant.total_cost():.6f}")
    print(f"  Original FAQ answer: {answer_orig[:200]}...")

    # -----------------------------------------------------------------------
    # 6. Processing function for one record
    # -----------------------------------------------------------------------
    # Takes one ground-truth record, runs RAG on its question, looks up the
    # original answer by document ID, and returns the comparison record
    # consumed by lesson 13.
    #
    # Rate limiting: Groq's free tier allows 6k tokens/minute and 100k
    # tokens/day for llama-3.1-8b-instant.  Each RAG call sends ~2.3k
    # tokens of context, so at most ~2.5 calls per minute are allowed.
    # A single worker pausing 24 seconds between calls stays just under
    # the per-minute cap and comfortably under the daily cap for GT_LIMIT
    # questions.  ThreadPoolExecutor is kept so paid tiers can simply
    # raise max_workers and shrink the pause.

    RATE_LIMIT_PAUSE_SECONDS = 24.0

    def generate_rag_answer(rec):
        """Run RAG on one ground-truth record.

        Args:
            rec: Dict with 'question' and 'document' keys.

        Returns:
            Dict with 'question', 'answer_llm', 'answer_orig', 'document'.
        """
        question = rec["question"]
        doc_id = rec["document"]

        # A' - the answer produced by our RAG system
        answer_llm = assistant.rag(question)

        # Respect the provider's token-per-minute limit
        time.sleep(RATE_LIMIT_PAUSE_SECONDS)

        # A - the original FAQ answer this question was generated from
        original_doc = doc_idx.get(doc_id)
        answer_orig = original_doc["answer"] if original_doc else ""

        return {
            "question": question,
            "answer_llm": answer_llm,
            "answer_orig": answer_orig,
            "document": doc_id,
        }

    # -----------------------------------------------------------------------
    # 7. Smoke test: first 5 questions, sequentially
    # -----------------------------------------------------------------------
    # Quick sanity check (prompt, retrieval, API key, cost) before the
    # full parallel run.  ~5 LLM calls, a few cents.

    print("\n--- 7. Smoke test on the first 5 questions ---")
    smoke = [generate_rag_answer(rec) for rec in ground_truth[:5]]
    print(f"  Generated {len(smoke)} answers, "
          f"running total cost: ${assistant.total_cost():.6f}")

    # -----------------------------------------------------------------------
    # 8. Run over the (sampled) set of questions
    # -----------------------------------------------------------------------
    # Each call is I/O-bound (network latency), so parallel workers overlap
    # the requests.  On the free tier a single worker with a 24s pause
    # between calls stays under the 6k-token/minute cap (see section 6);
    # bump max_workers and lower the pause for paid tiers.
    # map_progress keeps results in input order and updates a progress bar
    # as jobs finish.  Usage objects still accumulate in the assistant, so
    # total_cost() is accurate across threads.

    work_set = ground_truth[:GT_LIMIT]
    print(f"\n--- 8. Generating answers for {len(work_set)} questions "
          f"(GT_LIMIT={GT_LIMIT}) ---")
    with ThreadPoolExecutor(max_workers=1) as pool:
        results = map_progress(pool, work_set, generate_rag_answer)

    # -----------------------------------------------------------------------
    # 9. Save results
    # -----------------------------------------------------------------------
    # One row per question: the question, both answers (A and A'), and the
    # document ID.  Lesson 13 reads this CSV to judge answer quality.

    df_answers = pd.DataFrame(results)
    df_answers.to_csv("data/rag-answers-new.csv", index=False)

    print(f"\n--- 9. Saved {len(df_answers)} records to data/rag-answers-new.csv ---")

    # -----------------------------------------------------------------------
    # 10. Cost summary
    # -----------------------------------------------------------------------
    # total_cost() sums the tracked usage from every LLM call in this run.

    total_cost = assistant.total_cost()
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average cost per answer: ${total_cost / len(df_answers):.6f}")

    # -----------------------------------------------------------------------
    # 11. Interpretation
    # -----------------------------------------------------------------------
    print("\n--- 11. Interpretation ---")
    print("  Each record now pairs the RAG answer (A') with the original")
    print("  FAQ answer (A).  The next lesson judges how close A' is to A.")
    print("  If A' is close, the RAG system is doing a good job; if not,")
    print("  the fault can lie in retrieval (wrong documents) or in")
    print("  generation (right documents, weak answer) - lesson 13")
    print("  separates those two cases.")
