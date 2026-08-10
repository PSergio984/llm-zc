"""
14-agent-evaluation.py - Evaluate an agentic RAG pipeline with an LLM judge.

What this script does:
  Lessons 12-13 evaluated the fixed RAG pipeline: retrieve -> generate.
  This lesson evaluates the AGENT version, where an LLM agent decides
  when and what to search (tool calls) before producing the final answer.

  We reuse the A -> Q -> A' setup:
    A  = the original answer in the FAQ (ground truth)
    Q  = a question generated from that answer (ground truth, lesson 03)
    A' = the answer produced by the agent

  New in this lesson: we also record and evaluate the TRAJECTORY - the
  tool calls the agent made before answering.  The judge scores two
  things per record:

    - answer_score:     is the final answer correct and complete?
    - trajectory_score: were the tool calls reasonable?  (relevant
      queries, no duplicates, 1 call usually enough, 2-3 okay, more
      than 3 needs a clear reason)

  This separates "did the agent find the right info" from "did the
  agent behave sensibly while looking".

Pipeline:
  1. Load ground truth questions and the FAQ documents
  2. Run the agent (toyaikit, module 01) on every question
  3. Extract the trajectory (tool calls) from the message history
  4. Ask an LLM judge for answer_score + trajectory_score per record
  5. Aggregate stats and inspect the bad cases

Cost awareness:
  Free-tier daily token budgets cap the run size (GT_LIMIT, default 16):
  the agent phase runs on openai/gpt-oss-20b and the judge phase on
  openai/gpt-oss-120b, each with its own budget.  Raise GT_LIMIT when
  running against a paid tier.

Follows the llm-zoomcamp lesson "Agent Evaluation":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/14-agent-evaluation.md

Usage:
  python 04-evaluation/lessons/14-agent-evaluation.py

  Run from the repo root with both source dirs on the path, e.g.:
    $env:PYTHONPATH = ".;04-evaluation/code"
    python 04-evaluation/lessons/14-agent-evaluation.py

Requires an API key in .env (Groq-compatible endpoint for this run).
"""

import json
import os
import sys
import time
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from ingest import load_faq_data, build_index
from evaluation_utils import llm_structured_retry

# Windows consoles default to cp1252 and crash on Unicode characters in
# judge reasoning.  Force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

# Agent and judge use different models so their daily token budgets do
# not compete.  gpt-oss-20b runs the agent loop (tool calling), the
# stronger gpt-oss-120b acts as the judge.
AGENT_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"

# Number of ground-truth questions to process (see Cost awareness).
GT_LIMIT = 16


# ---------------------------------------------------------------------------
# 1. Load ground truth questions
# ---------------------------------------------------------------------------
# Same CSV as lessons 03-06: one row per question, with the ID of the
# document that contains the correct answer.

df_ground_truth = pd.read_csv("data/ground_truth-new.csv")
ground_truth_all = df_ground_truth.to_dict(orient="records")


# ---------------------------------------------------------------------------
# 2. Load FAQ documents and build the search index
# ---------------------------------------------------------------------------
# Only llm-zoomcamp documents are indexed, matching the ground-truth
# filter.  doc_idx maps a document ID to its original record, so we can
# fetch the original answer (A) for each question.

documents = load_faq_data()
documents = [doc for doc in documents if doc["course"] == "llm-zoomcamp"]

index = build_index(documents)

doc_idx = {}
for doc in documents:
    doc_idx[doc["id"]] = doc


# ---------------------------------------------------------------------------
# 3. Filter out records whose original document no longer exists
# ---------------------------------------------------------------------------
# The FAQ data changed since the ground truth was generated: some
# document IDs in the ground truth are missing from the current dataset.
# Those records have no original answer to compare against, so they are
# skipped (same treatment as in lesson 13).

ground_truth = [
    rec for rec in ground_truth_all if rec["document"] in doc_idx
]
skipped = len(ground_truth_all) - len(ground_truth)

print(f"Loaded {len(ground_truth_all)} ground truth records, "
      f"skipped {skipped} with missing documents, "
      f"using {len(ground_truth)}")


# ---------------------------------------------------------------------------
# 4. Define the search tool for the agent
# ---------------------------------------------------------------------------
# The agent's only tool: query the FAQ index.  Boosts match the values
# used in lesson 12 (question=1.0, answer=2.0, section=0.1).

def search(query: str) -> list[dict]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 1.0, "answer": 2.0, "section": 0.1},
        filter_dict={"course": "llm-zoomcamp"}
    )


# ---------------------------------------------------------------------------
# 5. Set up the agent
# ---------------------------------------------------------------------------
# ToyAIKit (module 01) runs the agent loop: it lets the model call tools,
# feeds the tool results back, and repeats until the model produces a
# final answer.  The runner stores the full message history so we can
# extract the trajectory afterwards.

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat.runners import OpenAIResponsesRunner

load_dotenv()
openai_client = OpenAI()

agent_tools = Tools()
agent_tools.add_tool(search)

agent_instructions = """
You're a course teaching assistant. Answer student questions based on
the FAQ search results. Use the search tool before answering.
""".strip()

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=agent_instructions,
    llm_client=OpenAIClient(model=AGENT_MODEL),
)

print(f"Agent ready (model: {AGENT_MODEL})")


# ---------------------------------------------------------------------------
# 6. Extract the trajectory from the message history
# ---------------------------------------------------------------------------
# The trajectory is the list of tool calls the agent made.  Message
# history may contain plain dicts (tool results) and typed message
# objects; we keep only function_call messages and record name +
# arguments.

def extract_tool_calls(messages):
    """Extract tool calls from the agent's message history.

    Args:
        messages: Full message history from the runner result.

    Returns:
        List of dicts with 'name' and 'arguments' keys.
    """
    tool_calls = []

    for message in messages:
        if isinstance(message, dict):
            continue

        if message.type == "function_call":
            try:
                arguments = json.loads(message.arguments)
            except (ValueError, TypeError):
                arguments = message.arguments

            tool_calls.append({
                "name": message.name,
                "arguments": arguments,
            })

    return tool_calls


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 7. Demo: run the agent on one question
    # -----------------------------------------------------------------------
    # Before the full run, check that the agent loop works: the final
    # answer, the trajectory it produced, and the cost of the run.

    print("\n--- 7. Demo: one question ---")
    demo_rec = ground_truth[0]
    demo_result = runner.loop(prompt=demo_rec["question"])

    print(f"  Q:   {demo_rec['question']}")
    print(f"  A':  {demo_result.last_message}")
    print(f"  Trajectory: {extract_tool_calls(demo_result.all_messages)}")
    demo_cost = demo_result.cost.total_cost if demo_result.cost else 0.0
    print(f"  Cost: ${demo_cost:.6f}")

    # -----------------------------------------------------------------------
    # 8. Run the agent over the sample of questions
    # -----------------------------------------------------------------------
    # One agent run per question.  Sequential with a short pause between
    # records to stay under the free tier's rate limits; the sample is
    # small, so total runtime is a few minutes.  Each record stores the
    # final answer, the original answer, the trajectory (as JSON), the
    # cost, and the document ID - ready for the judge phase.

    RATE_LIMIT_PAUSE_SECONDS = 10.0

    work_set = ground_truth[:GT_LIMIT]

    print(f"\n--- 8. Running the agent on {len(work_set)} questions ---")

    from tqdm.auto import tqdm

    agent_results = []
    for rec in tqdm(work_set):
        doc_id = rec["document"]
        result = runner.loop(prompt=rec["question"])
        tool_calls = extract_tool_calls(result.all_messages)

        agent_results.append({
            "question": rec["question"],
            "answer_agent": result.last_message,
            "answer_orig": doc_idx[doc_id]["answer"],
            "tool_calls": json.dumps(tool_calls),
            "cost": result.cost.total_cost if result.cost else 0.0,
            "document": doc_id,
        })

        time.sleep(RATE_LIMIT_PAUSE_SECONDS)

    df_agent = pd.DataFrame(agent_results)
    df_agent.to_csv("data/agent-answers-new.csv", index=False)

    total_cost = df_agent["cost"].sum()
    print(f"  Saved {len(df_agent)} records to data/agent-answers-new.csv")
    print(f"  Agent phase cost: ${total_cost:.4f}")

    # -----------------------------------------------------------------------
    # 9. Judge: score the answer AND the trajectory
    # -----------------------------------------------------------------------
    # The judge gets the question, the original answer, the agent's final
    # answer, and the JSON-serialized tool calls.  It returns two scores
    # with reasoning, so a bad verdict tells us whether the problem is
    # "bad answer" or "bad behavior" (or both).

    class AgentEvaluation(BaseModel):
        """Schema for the judge's verdict on one agent run."""
        answer_reasoning: str = Field(
            description="Reasoning about whether the final answer is correct."
        )
        answer_score: Literal["good", "bad"] = Field(
            description="'good' if the final answer matches the original answer."
        )
        trajectory_reasoning: str = Field(
            description="Reasoning about whether the tool calls were useful."
        )
        trajectory_score: Literal["good", "bad"] = Field(
            description="'good' if the tool calls were reasonable for the question."
        )

    agent_judge_instructions = """
You are an expert evaluator. You will be given:
1. A question from a student
2. The original answer from the FAQ (ground truth)
3. An answer generated by an AI agent
4. The tool calls made by the agent

Evaluate the answer quality and the trajectory quality.

Answer quality:
- Does the answer contain all the information from the original answer?
- Score it "good" if it does, "bad" otherwise.

Trajectory quality:
- Were the search queries relevant to the question?
- Did the queries include important keywords from the question?
- Did the agent avoid duplicate or unnecessary tool calls?
- If it made multiple searches, did the later searches refine the query?
- Was the number of search calls reasonable? Usually 1 is enough, 2-3
  can be okay, and more than 3 needs a clear reason.
- Did the tool calls support the final answer?

Explain both scores briefly.
""".strip()

    agent_judge_prompt = """
Question:
{question}

Original answer:
{answer_orig}

Generated answer:
{answer_agent}

Tool calls:
{tool_calls}
""".strip()

    def evaluate_agent(question, answer_orig, answer_agent, tool_calls,
                       model=JUDGE_MODEL):
        """Ask the LLM judge to score one agent run.

        Args:
            question: The question the agent was asked.
            answer_orig: The original FAQ answer (ground truth).
            answer_agent: The agent's final answer.
            tool_calls: List of dicts with 'name' and 'arguments'.
            model: Model ID for the judge.

        Returns:
            Tuple of (AgentEvaluation instance, usage object).
        """
        prompt = agent_judge_prompt.format(
            question=question,
            answer_orig=answer_orig,
            answer_agent=answer_agent,
            tool_calls=json.dumps(tool_calls, indent=2),
        )

        return llm_structured_retry(
            openai_client,
            agent_judge_instructions,
            prompt,
            AgentEvaluation,
            model=model,
        )

    print(f"\n--- 9. Judging {len(df_agent)} records (model: {JUDGE_MODEL}) ---")

    agent_evaluations = []
    usages = []

    for rec in tqdm(df_agent.to_dict(orient="records")):
        agent_eval, usage = evaluate_agent(
            question=rec["question"],
            answer_orig=rec["answer_orig"],
            answer_agent=rec["answer_agent"],
            tool_calls=json.loads(rec["tool_calls"]),
        )

        agent_evaluations.append({
            "question": rec["question"],
            "document": rec["document"],
            "answer_score": agent_eval.answer_score,
            "answer_reasoning": agent_eval.answer_reasoning,
            "trajectory_score": agent_eval.trajectory_score,
            "trajectory_reasoning": agent_eval.trajectory_reasoning,
        })
        usages.append(usage)

        time.sleep(RATE_LIMIT_PAUSE_SECONDS)

    df_eval = pd.DataFrame(agent_evaluations)

    # -----------------------------------------------------------------------
    # 10. Aggregate stats
    # -----------------------------------------------------------------------
    # Two independent metrics: how often is the ANSWER right, and how
    # often is the BEHAVIOR (tool usage) right.  The cross-tab shows
    # combinations like "good answer but bad trajectory" (lucky answer
    # after sloppy searching).

    print("\n--- 10. Results ---")
    print("\n  Answer score:")
    print(df_eval["answer_score"].value_counts().to_string())
    answer_good = (df_eval["answer_score"] == "good").sum()
    print(f"  Good answers: {answer_good}/{len(df_eval)} "
          f"= {answer_good / len(df_eval):.2%}")

    print("\n  Trajectory score:")
    print(df_eval["trajectory_score"].value_counts().to_string())
    traj_good = (df_eval["trajectory_score"] == "good").sum()
    print(f"  Good trajectories: {traj_good}/{len(df_eval)} "
          f"= {traj_good / len(df_eval):.2%}")

    print("\n  Answer x trajectory cross-tab:")
    print(pd.crosstab(df_eval["answer_score"], df_eval["trajectory_score"])
          .to_string())

    # -----------------------------------------------------------------------
    # 11. Look at the bad cases
    # -----------------------------------------------------------------------
    # The judge's reasoning explains each bad verdict: wrong retrieval,
    # missing key point, irrelevant or duplicate searches, too many
    # calls, etc.

    bad_answer = df_eval[df_eval["answer_score"] == "bad"]
    bad_trajectory = df_eval[df_eval["trajectory_score"] == "bad"]

    print(f"\n--- 11. Bad cases (bad answer: {len(bad_answer)}, "
          f"bad trajectory: {len(bad_trajectory)}) ---")

    for i, row in df_eval.iterrows():
        if row["answer_score"] == "bad" or row["trajectory_score"] == "bad":
            print(f"\n  [{i}] Q: {row['question']}")
            if row["answer_score"] == "bad":
                print(f"      ANSWER ({row['answer_score']}): "
                      f"{row['answer_reasoning']}")
            if row["trajectory_score"] == "bad":
                print(f"      TRAJECTORY ({row['trajectory_score']}): "
                      f"{row['trajectory_reasoning']}")

    # -----------------------------------------------------------------------
    # 12. Save and interpret
    # -----------------------------------------------------------------------
    df_eval.to_csv("data/agent-eval-results.csv", index=False)
    print(f"\n--- 12. Saved {len(df_eval)} judged records to "
          f"data/agent-eval-results.csv ---")

    print("\n--- Interpretation ---")
    print("  answer_score measures what the user actually gets;")
    print("  trajectory_score measures how the agent looked for it.")
    print("  A good answer with a bad trajectory means the agent got")
    print("  lucky after sloppy searching; a bad answer with a good")
    print("  trajectory points at the generation step instead.")
    print("  The judge is another LLM - verify suspicious verdicts by")
    print("  reading the reasoning, and keep the trajectory in the")
    print("  evaluation data so bad behavior is debuggable.")
