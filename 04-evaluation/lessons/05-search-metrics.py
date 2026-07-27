"""
05-search-metrics.py — Compute hit-rate and MRR from relevance lists.

What this script does:
  Takes the relevance lists from lesson 04 (each list is [0/1, 0/1, ...] per
  query, one entry per rank position) and computes two standard IR metrics:
    - Hit Rate (Recall@k): fraction of queries where the correct document
      appears anywhere in the top-k results.
    - Mean Reciprocal Rank (MRR): average of 1/rank for the first correct
      document.  Rewards systems that put the right document near the top.

Why both metrics?
  Hit Rate tells us if the document was found at all.  MRR tells us where.
  A system with high hit rate but low MRR retrieves the right document but
  buries it under irrelevant results.  Both are needed to understand search
  quality.

Pipeline:
  1. Import ground truth data and search pipeline from lesson 04
  2. Define hit_rate() and mrr() metric functions
  3. Define evaluate() that combines relevance computation + metrics
  4. Run evaluation and print results

Follows the llm-zoomcamp lesson "Search Evaluation Metrics":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/05-search-metrics.md

Usage:
  python 04-evaluation/lessons/05-search-metrics.py
"""

import importlib.util
import os

# The lesson 04 file has a hyphen in its name, so we can't import it with a
# regular import statement.  Use importlib to load it from its file path.
_lesson04_path = os.path.join(
    os.path.dirname(__file__), "04-search-evaluation.py"
)
_spec = importlib.util.spec_from_file_location(
    "lesson04", _lesson04_path, submodule_search_locations=[]
)
lesson04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson04)

ground_truth = lesson04.ground_truth
text_search = lesson04.text_search
compute_relevance_total = lesson04.compute_relevance_total

# ── 1. Hit Rate (Recall@k) ───────────────────────────────────────────────
# Hit rate measures the fraction of queries where the correct document appears
# anywhere in the top-k search results.  In our setup, each query has exactly
# one correct document, so hit rate and recall are equivalent.
#
# Formula:
#   hit_rate = (queries with at least one relevant result) / (total queries)

def hit_rate(relevance):
    """Compute hit rate from a list of relevance lists.

    Args:
        relevance: List of lists, each inner list is binary relevance per rank.

    Returns:
        Float in [0, 1]: fraction of queries with at least one relevant doc.
    """
    cnt = 0
    for line in relevance:
        if 1 in line:
            cnt += 1
    return cnt / len(relevance)


# ── 2. Mean Reciprocal Rank (MRR) ────────────────────────────────────────
# MRR considers the position of the first relevant document.  If the correct
# doc is at rank 1, score = 1/1 = 1.0.  At rank 2, score = 1/2 = 0.5.
# At rank 3, score = 1/3 ≈ 0.333.  Not found = 0.
#
# Why use rank + 1?
#   Python lists are 0-indexed.  Rank 0 in the list = position 1 to the user.
#   We add 1 so position 1 scores 1/1 instead of 1/0 (which would crash).

def mrr(relevance):
    """Compute Mean Reciprocal Rank from a list of relevance lists.

    Args:
        relevance: List of lists, each inner list is binary relevance per rank.

    Returns:
        Float in [0, 1]: average of reciprocal ranks of first relevant doc.
    """
    total_score = 0.0
    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance)


# ── 3. Combined evaluation function ──────────────────────────────────────
# Wraps relevance computation and metric calculation into a single call.
# Takes any search_function so we can evaluate text, vector, or hybrid search
# with the same interface.

def evaluate(ground_truth, search_function):
    """Compute hit rate and MRR for a given search function.

    Args:
        ground_truth: List of dicts with 'question' and 'document'.
        search_function: Callable(query) returning ranked result dicts.

    Returns:
        Dict with 'hit_rate' and 'mrr' keys.
    """
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


# ── 4. Run evaluation ────────────────────────────────────────────────────
# The example relevance list from the course lesson for demonstration.

if __name__ == "__main__":
    # Example from the lesson (15 queries, 5 results each)
    example = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]

    print("--- Example (15 queries from course lesson) ---")
    print(f"  Hit Rate: {hit_rate(example):.3f}  (expected: 0.933)")
    print(f"  MRR:      {mrr(example):.3f}  (expected: 0.822)")

    # Run on the full ground truth dataset with text search
    print(f"\n--- Full evaluation ({len(ground_truth)} queries) ---")
    metrics = evaluate(ground_truth, text_search)
    print(f"  Hit Rate: {metrics['hit_rate']:.3f}")
    print(f"  MRR:      {metrics['mrr']:.3f}")

    # ── 5. Interpretation ─────────────────────────────────────────────────
    print(f"\n--- Interpretation ---")
    print(f"  Hit Rate ({metrics['hit_rate']:.1%}): "
          f"the correct doc appears in top-5 for "
          f"{metrics['hit_rate']:.1%} of queries.")
    print(f"  MRR ({metrics['mrr']:.3f}): "
          f"the first correct doc is, on average, at rank "
          f"{1 / metrics['mrr']:.1f} (reciprocal).")
    print()
    print("  Note: with synthetic ground truth, questions may be too close")
    print("  to the original FAQ text, which can inflate these numbers.")
    print("  Good thresholds depend on your use case and how much the")
    print("  downstream LLM can compensate for imperfect retrieval.")
