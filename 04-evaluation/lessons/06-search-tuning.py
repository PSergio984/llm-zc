"""
06-search-tuning.py - Tune search parameters against the ground truth.

What this script does:
  Lesson 04 built a keyword search index and lesson 05 defined the metrics
  (hit rate, MRR).  Until now the search boosts were a guess: we boosted
  'question' to 3.0 on intuition.  This script replaces guessing with data.
  For each parameter combination we re-run the same ground-truth queries
  through the search index and watch how hit rate and MRR move.

Why offline tuning matters:
  The ground truth dataset is fixed, so every evaluation uses the exact
  same queries and correct answers.  When we change one boost and the
  metrics change, the difference is attributable to that boost - not to
  random variation in the questions.  This is the core benefit of offline
  evaluation: measure first, trust intuition second.

Pipeline:
  1. Reuse ground_truth, index and evaluate() from lessons 04/05
  2. Sweep the question boost alone:  [0.5, 1.0, 3.0, 5.0, 10.0]
  3. Grid search all three boosts:     question x answer x section
  4. Pick the best combination and freeze it into a tuned search function
  5. Compare tuned search against the original guess (question=3.0)

Follows the llm-zoomcamp lesson "Search Parameter Tuning":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/06-search-tuning.md

Usage:
  python 04-evaluation/lessons/06-search-tuning.py

Requires network access on first run: lesson 04 downloads the ground-truth
CSV and the FAQ data if they are not cached locally.
"""

import importlib.util
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Load lessons 04 and 05 as modules.
#
# The lesson files are not a package, so a plain `import` will not work.
# We load them by path with importlib - the same trick lesson 05 already
# uses.  Lesson 05 itself loads lesson 04, so the FAQ download and index
# build happen exactly once.  We reach into it for the shared objects:
#
#   evaluate()   from lesson 05  - computes hit rate + MRR for a search fn
#   ground_truth from lesson 05  - the (question, correct-doc) dataset
#   index        from lesson 04  - the minsearch index we are tuning
# ---------------------------------------------------------------------------

_lesson05_path = os.path.join(
    os.path.dirname(__file__), "05-search-metrics.py"
)
_spec05 = importlib.util.spec_from_file_location(
    "lesson05", _lesson05_path, submodule_search_locations=[]
)
lesson05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(lesson05)

evaluate = lesson05.evaluate
ground_truth = lesson05.ground_truth
index = lesson05.lesson04.index

print(f"Loaded {len(ground_truth)} ground truth records, index ready.")


# ---------------------------------------------------------------------------
# 1. Configurable question boost
# ---------------------------------------------------------------------------
# The current search function (lesson 04) hard-codes question=3.0.  To tune
# it we need a version where the question boost is an argument, so we can
# sweep it without editing code.  The section boost stays at 0.5 - section
# is a weak signal (the course name), so it should never outrank content.

def search_boost(query, question_boost):
    """Search with a configurable question field boost.

    Args:
        query: Natural-language query string.
        question_boost: Weight for matches in the 'question' field.

    Returns:
        List of result dicts with 'id', 'question', 'answer', etc.
    """
    boost_dict = {"question": question_boost, "section": 0.5}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )


# Sweep one parameter at a time.  The other boost stays fixed, so any
# change in the metrics is caused by this parameter alone.
# The lesson's data shows that boosting 'question' actually hurts: the
# best hit rate lands at boost=1.0 (no boost at all).  Intuition said 3.0
# was better - the data says otherwise, which is exactly the point.

question_boosts = [0.5, 1.0, 3.0, 5.0, 10.0]

sweep_rows = []
for boost in question_boosts:
    metrics = evaluate(
        ground_truth,
        lambda query, boost=boost: search_boost(query, boost),
    )
    sweep_rows.append({
        "question_boost": boost,
        "hit_rate": metrics["hit_rate"],
        "mrr": metrics["mrr"],
    })

df_sweep = pd.DataFrame(sweep_rows)

print("\n--- 1. Question boost sweep (section fixed at 0.5) ---")
print(df_sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_sweep = df_sweep.loc[df_sweep["hit_rate"].idxmax()]
print(
    f"\nBest from sweep: question_boost={best_sweep['question_boost']:.1f} "
    f"(hit_rate={best_sweep['hit_rate']:.4f}, mrr={best_sweep['mrr']:.4f})"
)


# ---------------------------------------------------------------------------
# 2. Grid search over all three boosts
# ---------------------------------------------------------------------------
# Tuning one parameter in isolation can miss interactions between them
# (e.g. a high answer boost only helps when the question boost is low).
# A grid search evaluates every combination of the candidate values:

def search_boosts(query, question_boost, answer_boost, section_boost):
    """Search with all three field boosts configurable.

    Args:
        query: Natural-language query string.
        question_boost: Weight for matches in the 'question' field.
        answer_boost: Weight for matches in the 'answer' field.
        section_boost: Weight for matches in the 'section' field.

    Returns:
        List of result dicts with 'id', 'question', 'answer', etc.
    """
    boost_dict = {
        "question": question_boost,
        "section": section_boost,
        "answer": answer_boost,
    }

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )


# 3 x 4 x 3 = 36 combinations.  Each one re-runs all ~395 queries through
# the in-memory index, so the whole grid takes only a couple of minutes
# and costs nothing (no LLM calls involved).

grid_rows = []
for question_boost in [1.0, 2.0, 5.0]:
    for answer_boost in [1.0, 2.0, 4.0, 10.0]:
        for section_boost in [0.1, 0.2, 0.5]:
            metrics = evaluate(
                ground_truth,
                lambda query, q=question_boost, a=answer_boost, s=section_boost:
                    search_boosts(query, q, a, s),
            )
            grid_rows.append({
                "question_boost": question_boost,
                "answer_boost": answer_boost,
                "section_boost": section_boost,
                "hit_rate": metrics["hit_rate"],
                "mrr": metrics["mrr"],
            })

df_grid = pd.DataFrame(grid_rows)
df_grid_sorted = df_grid.sort_values("hit_rate", ascending=False)

print("\n--- 2. Grid search (36 combinations, sorted by hit rate) ---")
print(df_grid_sorted.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_grid = df_grid_sorted.iloc[0]
print(
    f"\nBest from grid: question={best_grid['question_boost']:.1f}, "
    f"answer={best_grid['answer_boost']:.1f}, section={best_grid['section_boost']:.1f} "
    f"(hit_rate={best_grid['hit_rate']:.4f}, mrr={best_grid['mrr']:.4f})"
)


# ---------------------------------------------------------------------------
# 3. Freeze the best parameters into the tuned search function
# ---------------------------------------------------------------------------
# The winning combination becomes the new default.  Keeping it in one named
# function means the rest of the code (and future lessons) can call
# text_search_tuned() and get the tuned behavior without knowing the details.

BEST_QUESTION_BOOST = float(best_grid["question_boost"])
BEST_ANSWER_BOOST = float(best_grid["answer_boost"])
BEST_SECTION_BOOST = float(best_grid["section_boost"])


def text_search_tuned(query):
    """Search with the best boosts found by the grid search.

    Args:
        query: Natural-language query string.

    Returns:
        List of result dicts with 'id', 'question', 'answer', etc.
    """
    boost_dict = {
        "question": BEST_QUESTION_BOOST,
        "answer": BEST_ANSWER_BOOST,
        "section": BEST_SECTION_BOOST,
    }

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )


# ---------------------------------------------------------------------------
# 4. Compare tuned vs. the original guess
# ---------------------------------------------------------------------------
# The original text_search from lesson 04 used question=3.0, section=0.5 -
# an intuition-based guess.  The tuned version comes from the grid search.
# The delta is the measurable value of offline evaluation.

if __name__ == "__main__":
    text_search_original = lesson05.text_search

    print("\n--- 4. Tuned vs. original search ---")

    metrics_original = evaluate(ground_truth, text_search_original)
    metrics_tuned = evaluate(ground_truth, text_search_tuned)

    print(f"  Original (question=3.0, section=0.5): "
          f"hit_rate={metrics_original['hit_rate']:.4f}, "
          f"mrr={metrics_original['mrr']:.4f}")
    print(f"  Tuned    (question={BEST_QUESTION_BOOST:.1f}, "
          f"answer={BEST_ANSWER_BOOST:.1f}, section={BEST_SECTION_BOOST:.1f}): "
          f"hit_rate={metrics_tuned['hit_rate']:.4f}, "
          f"mrr={metrics_tuned['mrr']:.4f}")

    print(
        f"\n  Improvement: "
        f"hit_rate {metrics_tuned['hit_rate'] - metrics_original['hit_rate']:+.4f}, "
        f"mrr {metrics_tuned['mrr'] - metrics_original['mrr']:+.4f}"
    )

    # -----------------------------------------------------------------------
    # 5. Interpretation and tuning workflow notes
    # -----------------------------------------------------------------------
    print("\n--- 5. Interpretation ---")
    print("  Boosting the 'question' field beyond 1.0 hurt both metrics on")
    print("  this dataset - matching the answer text matters more than the")
    print("  lesson's initial intuition suggested.")
    print("  Grid search is the right tool when the parameter space is small")
    print("  (here: 36 combinations).  For larger spaces, sample randomly,")
    print("  use Bayesian optimization, or hold out a validation split so the")
    print("  chosen parameters do not overfit the evaluation set.")
    print()
    print(f"  Final tuned function: text_search_tuned() - "
          f"question={BEST_QUESTION_BOOST:.1f}, answer={BEST_ANSWER_BOOST:.1f}, "
          f"section={BEST_SECTION_BOOST:.1f}")
