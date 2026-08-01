"""
evaluation_utils.py — Reusable helpers for LLM-based evaluation.

Provides functions for:
  - Calling the OpenAI Responses API with structured (Pydantic) output
  - Retrying structured-output calls when requests fail
  - Calculating API costs from token usage
  - Running functions in parallel with a progress bar
  - RAGBase subclass that tracks token usage across calls

Intended for use in the lesson workflows under 04-evaluation/lessons/.
"""

import time

from tqdm.auto import tqdm

from rag_helper import RAGBase


# ── Pricing helpers ──────────────────────────────────────────────────────
# OpenAI charges per token, split into input (prompt) and output (completion).
# The rates below are for llama-3.3-70b-versatile on Groq, used in this course.
# We return a dict so callers can inspect the breakdown if needed.


def calc_price(usage):
    """
    Calculate USD cost from a single API usage object.

    Args:
        usage: Object with .input_tokens and .output_tokens int attributes.

    Returns:
        dict with keys 'input_cost', 'output_cost', 'total_cost'.
    """
    input_price_per_million = 0.59
    output_price_per_million = 0.79

    # Convert per-token rates to actual cost based on usage
    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    """
    Sum the total cost of multiple API usage objects.

    Args:
        usages: Iterable of usage objects with .input_tokens, .output_tokens.

    Returns:
        float total cost in USD.
    """
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


# ── Structured-output helpers ────────────────────────────────────────────
# The OpenAI Responses API supports structured output via the `text_format`
# parameter.  Instead of getting free text back, we pass a Pydantic model
# and receive a fully parsed instance — no JSON parsing required.


def llm_structured(client, instructions, user_prompt, output_type, model="llama-3.3-70b-versatile"):
    """
    Call the OpenAI Responses API with structured output.

    Uses client.responses.parse() with text_format=output_type so the
    return value is a parsed Pydantic model instead of raw text.

    Args:
        client: OpenAI client instance.
        instructions: Developer/system instruction for the LLM.
        user_prompt: User message content (string).
        output_type: A Pydantic BaseModel subclass — the API returns a
                     parsed instance of this model.
        model: Model ID string (default gpt-5.4-mini).

    Returns:
        Tuple of (parsed_model_instance, usage_object).
    """
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type
    )

    return response.output_parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="llama-3.3-70b-versatile",
    max_retries=3,
):
    """
    Call llm_structured with exponential-backoff retry logic.

    Useful in batch processing: a single flaky network call shouldn't
    fail the entire ground-truth generation run.  Retries with 2^attempt
    seconds of sleep between attempts (1s, 2s, 4s).

    Args:
        Same as llm_structured, plus:
        max_retries: Number of attempts before giving up (default 3).
    """
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


# ── RAG with usage tracking ──────────────────────────────────────────────


class RAGWithUsage(RAGBase):
    """
    RAGBase subclass that tracks token usage for every LLM call.

    Overrides search() with different boost weights and llm() to capture
    usage from each response.  Accumulates usage objects in self.usages
    and provides total_cost() to calculate the combined price.

    Use this when you need cost-awareness during evaluation runs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        """Clear accumulated usage tracking (e.g. between eval runs)."""
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        # Boost answer matching over question/section for evaluation queries
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        # Capture usage before returning so we can price the run later
        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        """Return total cost in USD for all tracked usage objects."""
        return calc_total_price(self.usages)


# ── Parallel processing ──────────────────────────────────────────────────
# ThreadPoolExecutor + tqdm: submits all jobs at once, then updates the
# progress bar as each one finishes.  Keeps results in input order.


def map_progress(pool, seq, f):
    """
    Apply function *f* to every element in *seq* using an existing
    ThreadPoolExecutor, and show a tqdm progress bar while waiting.

    Submits all tasks immediately, then collects results in order as they
    complete.  This is more efficient than iterating sequentially when each
    call is I/O-bound (e.g., waiting on an HTTP response).

    Args:
        pool: A concurrent.futures.ThreadPoolExecutor instance.
        seq: Iterable of inputs.
        f: Callable taking a single item.

    Returns:
        List of results in the same order as *seq*.
    """
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            # Callback fires when the future completes, updating the bar
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        # Collect in submission order so results match seq order
        for future in futures:
            result = future.result()
            results.append(result)

    return results
