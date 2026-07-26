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


def calc_price(usage):
    """
    Calculate USD cost from a single API usage object.

    Args:
        usage: Object with .input_tokens and .output_tokens int attributes.

    Returns:
        dict with keys 'input_cost', 'output_cost', 'total_cost'.
    """
    input_price_per_million = 0.75
    output_price_per_million = 4.50

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


def llm_structured(client, instructions, user_prompt, output_type, model="gpt-5.4-mini"):
    """
    Call the OpenAI Responses API with structured output.

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
    model="gpt-5.4-mini",
    max_retries=3,
):
    """
    Call llm_structured with exponential-backoff retry logic.

    Retries on any exception up to `max_retries` times, sleeping
    2^attempt seconds between retries.
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

    Accumulates usage objects in self.usages and provides total_cost()
    to calculate the combined price.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        """Clear accumulated usage tracking."""
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
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

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        """Return total cost in USD for all tracked usage."""
        return calc_total_price(self.usages)


# ── Parallel processing ──────────────────────────────────────────────────


def map_progress(pool, seq, f):
    """
    Apply function *f* to every element in *seq* using an existing
    ThreadPoolExecutor, and show a tqdm progress bar while waiting.

    Results are returned in the same order as *seq*.

    Args:
        pool: A concurrent.futures.ThreadPoolExecutor instance.
        seq: Iterable of inputs.
        f: Callable taking a single item.

    Returns:
        List of results in input order.
    """
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
