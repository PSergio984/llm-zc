"""
judge.py — Automatic relevance evaluation for every answer (lesson 09).

Execution metrics (time, tokens, cost) don't say whether an answer was
any good. After each RAG response, an LLM judge grades how relevant the
answer is to the question — an automatic quality signal that doesn't
wait for a human to click thumbs up or down.

Unlike the offline judge from module 04, there is no ground truth to
compare against: the judge sees only the question and the answer, so the
instructions describe more carefully what a good answer looks like. The
verdict is a structured label:

  - RELEVANT        → the answer addresses the question
  - PARTLY_RELEVANT → the answer partially addresses the question
  - NON_RELEVANT    → the answer does not address the question

plus an `explanation` field. Asking for an explanation forces the judge
to reason before committing to a label, which makes the label better.

The call goes through llm_structured_retry from module 04's
evaluation_utils (found via sys.path — it lives in 04-evaluation/code),
which uses the OpenAI Responses API structured output. When the
provider rejects json_schema (e.g. Groq), evaluation_utils falls back
to JSON mode automatically.

Handle verdicts with care: the judge is deliberately basic and can
mislabel. Align it with real user labels before trusting it.

Follows the llm-zoomcamp lesson "Built-in Judge":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/09-built-in-judge.md

Usage:
  uv run python judge.py
"""

import json
import sys

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

# evaluation_utils lives in module 04; make it importable from the root
sys.path.insert(0, "04-evaluation/code")

from evaluation_utils import llm_structured_retry


class RelevanceVerdict(BaseModel):
    """Structured output of the judge: a label plus the reasoning."""

    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()


def evaluate_relevance(question, answer, client=None):
    """Judge how relevant `answer` is to `question`.

    Returns (relevance_label, explanation). Uses a fresh OpenAI()
    client by default, or the caller's client if one is passed.
    """
    if client is None:
        client = OpenAI()

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
    )

    return result.relevance, result.explanation


if __name__ == "__main__":
    load_dotenv()

    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)
