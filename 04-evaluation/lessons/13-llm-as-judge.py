"""
13-llm-as-judge.py - Use an LLM judge to score RAG answer quality.

What this script does:
  Lesson 12 generated a RAG answer (A') for every ground-truth question
  and paired it with the original FAQ answer (A).  Now we need to decide
  whether A' is good enough.  Exact text matching is too strict - a good
  answer can rephrase the FAQ while keeping the same key information.
  Instead, a second LLM call (the "judge") compares three pieces:

    1. the question generated from the FAQ record (Q)
    2. the original FAQ answer (A, ground truth)
    3. the answer produced by our RAG pipeline (A')

  The judge returns a score ("good" or "bad") plus a short reasoning
  explaining the verdict.  The score aggregates into one metric (fraction
  of answers judged good); the reasoning tells us where to investigate
  when an answer fails.

  This evaluates the full RAG flow in one pass:
    - search:  did we retrieve context that contains the answer?
    - prompt:  did we give the model enough context to answer?
    - LLM:     did the model produce a useful answer from that context?

  Important caveat: the judge can be wrong (too lenient or too strict).
  It points us at suspicious cases - it does not replace reading them.

Pipeline:
  1. Load the RAG answers CSV from lesson 12
  2. Define the structured judge output (score + reasoning)
  3. Run the judge on every record (one LLM call per record)
  4. Aggregate stats: how many answers are good?
  5. Print the bad cases with the judge's reasoning

Follows the llm-zoomcamp lesson "LLM as a Judge":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/13-llm-as-judge.md

Usage:
  python 04-evaluation/lessons/13-llm-as-judge.py

  Run from the repo root with both source dirs on the path, e.g.:
    $env:PYTHONPATH = ".;04-evaluation\code"
    python 04-evaluation/lessons/13-llm-as-judge.py

Requires an API key in .env (Groq-compatible endpoint for this run).
"""

import os
import sys
import time
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from evaluation_utils import llm_structured_retry

# Windows consoles default to cp1252 and crash on Unicode characters in
# judge reasoning (e.g. non-breaking hyphens).  Force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

# Judge model.  A strong 70b+-class model is preferable for judging, and
# its daily token budget is separate from the 8b model used in lesson 12.
# (The old llama-3.3-70b-specdec / llama-3.1-70b-versatile ids are
# decommissioned on Groq; gpt-oss-120b is the strongest model with a
# fresh free-tier daily budget.)
JUDGE_MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# 1. Load the RAG answers from lesson 12
# ---------------------------------------------------------------------------
# One row per question: the question, the RAG answer (A'), and the
# original FAQ answer (A).  Rows with an empty original answer come from
# ground-truth document IDs that no longer exist in the current FAQ data
# (the dataset changed since the ground truth was generated).  They are
# skipped: with no original answer there is nothing to compare against,
# and the judge would only mark them "bad" for the wrong reason.

df_answers = pd.read_csv("data/rag-answers-new.csv")
total = len(df_answers)

# Note: pandas reads empty cells as NaN, and bool(NaN) is True - so a
# naive `if rec["answer_orig"]` filter would NOT drop the empty rows.
# Filter on the DataFrame before converting to records.
df_answers = df_answers[
    df_answers["answer_orig"].notna()
    & (df_answers["answer_orig"].astype(str).str.strip() != "")
]

answers = df_answers.to_dict(orient="records")

skipped = total - len(answers)

print(f"Loaded {total} RAG answers, skipped {skipped} without an original "
      f"answer, judging {len(answers)}")


# ---------------------------------------------------------------------------
# 2. Define the judge's output format
# ---------------------------------------------------------------------------
# Structured output (Pydantic) instead of free text: the judge must return
# exactly two fields.  score is a strict "good"/"bad" choice we can count;
# reasoning is free text explaining the verdict.  The API enforces this
# schema, so we never have to parse fragile free-text output.

class AnswerEvaluation(BaseModel):
    """Schema for the judge's verdict on one RAG answer."""
    reasoning: str = Field(
        description="Reasoning about the quality of the answer."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the answer is correct and complete, 'bad' otherwise."
    )


# ---------------------------------------------------------------------------
# 3. Judge instructions and prompt template
# ---------------------------------------------------------------------------
# The instructions tell the judge what to compare and how to assign the
# score.  The prompt template is the per-record data we fill in: question,
# original answer, generated answer.

aqa_judge_instructions = """
You are an expert evaluator. You will be given:
1. A question from a student
2. The original answer from the FAQ (ground truth)
3. An answer generated by an AI assistant

Evaluate whether the answer contains all the information from the original answer.

If the answer contains all the information, score it as "good", otherwise score it as "bad".

Explain in a short text what you think is wrong with the answer.
""".strip()

aqa_judge_prompt = """
Question:
{question}

Original answer:
{answer_orig}

Generated answer:
{answer_llm}
""".strip()


# ---------------------------------------------------------------------------
# 4. Single-record judge call
# ---------------------------------------------------------------------------
# Wraps llm_structured_retry: one structured-output call with retries on
# temporary API/network errors, returning the parsed verdict and the token
# usage (so we can price the whole judging run).

def evaluate_aqa(question, answer_orig, answer_llm, model=JUDGE_MODEL):
    """Ask the LLM judge to score one RAG answer.

    Args:
        question: The question the RAG system was asked.
        answer_orig: The original FAQ answer (ground truth).
        answer_llm: The answer produced by the RAG pipeline.
        model: Model ID for the judge.

    Returns:
        Tuple of (AnswerEvaluation instance, usage object).
    """
    prompt = aqa_judge_prompt.format(
        question=question,
        answer_orig=answer_orig,
        answer_llm=answer_llm,
    )

    result, usage = llm_structured_retry(
        openai_client,
        aqa_judge_instructions,
        prompt,
        AnswerEvaluation,
        model=model,
    )

    return result, usage


if __name__ == "__main__":
    load_dotenv()
    openai_client = OpenAI()

    # -----------------------------------------------------------------------
    # 5. Run the judge on every record
    # -----------------------------------------------------------------------
    # One judge call per record.  Sequential with a small pause between
    # calls to stay under the free tier's token-per-minute limit; the
    # dataset is small (35 records), so this finishes in a few minutes.

    RATE_LIMIT_PAUSE_SECONDS = 10.0

    results = []
    usages = []

    print(f"\n--- 5. Judging {len(answers)} answers (model: {JUDGE_MODEL}) ---")

    from tqdm.auto import tqdm

    for rec in tqdm(answers):
        eval_result, usage = evaluate_aqa(
            question=rec["question"],
            answer_orig=rec["answer_orig"],
            answer_llm=rec["answer_llm"],
        )

        results.append({
            "question": rec["question"],
            "answer_orig": rec["answer_orig"],
            "answer_llm": rec["answer_llm"],
            "reasoning": eval_result.reasoning,
            "score": eval_result.score,
        })
        usages.append(usage)

        time.sleep(RATE_LIMIT_PAUSE_SECONDS)

    df_eval = pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # 6. Aggregate stats
    # -----------------------------------------------------------------------
    # The fraction of "good" answers is the single headline metric: it
    # tells us what share of ground-truth questions our RAG system answers
    # correctly and completely.

    good_count = (df_eval["score"] == "good").sum()
    total_count = len(df_eval)

    print("\n--- 6. Overall results ---")
    print(f"  Good: {good_count}/{total_count} = {good_count / total_count:.2%}")
    print(f"  Bad:  {total_count - good_count}/{total_count} = "
          f"{(total_count - good_count) / total_count:.2%}")

    # -----------------------------------------------------------------------
    # 7. Look at the bad cases
    # -----------------------------------------------------------------------
    # The metric alone is not actionable.  For every "bad" verdict the
    # judge explains what went wrong - wrong retrieved document, missing
    # key point, model saying it doesn't know, etc.  This is where the
    # evaluation tells us what to fix next.

    bad_cases = df_eval[df_eval["score"] == "bad"]

    print(f"\n--- 7. Bad cases ({len(bad_cases)}) ---")
    for i, row in bad_cases.iterrows():
        print(f"\n  [{i}] Q: {row['question']}")
        print(f"      A (original): {row['answer_orig'][:200]}")
        print(f"      A' (RAG):     {row['answer_llm'][:200]}")
        print(f"      Judge:        {row['reasoning']}")

    # -----------------------------------------------------------------------
    # 8. Save the full judged results for later analysis
    # -----------------------------------------------------------------------
    df_eval.to_csv("data/rag-judge-results.csv", index=False)
    print(f"\n--- 8. Saved {len(df_eval)} judged records to "
          f"data/rag-judge-results.csv ---")

    # -----------------------------------------------------------------------
    # 9. Interpretation
    # -----------------------------------------------------------------------
    print("\n--- 9. Interpretation ---")
    print("  The judge is a second LLM, not ground truth: it can rate an")
    print("  answer 'good' even when search retrieved the wrong document")
    print("  (too lenient), or 'bad' when the answer is fine but rephrased")
    print("  (too strict).  The verdicts point at suspicious cases - read")
    print("  them before changing the pipeline.")
    print("  In production there is usually no original answer for real")
    print("  user questions; the judge prompt then compares only the")
    print("  question and the generated answer.  Here we use the stronger")
    print("  offline setup with known ground truth.")
