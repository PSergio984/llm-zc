"""
evaluation_utils.py — Reusable helpers for LLM-based evaluation.

Provides functions for:
  - Calling the OpenAI Responses API with structured (Pydantic) output
  - Retrying structured-output calls when requests fail
  - Calculating API costs from token usage
  - Running functions in parallel with a progress bar

Intended for use in the lesson workflows under 04-evaluation/lessons/.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

# OpenAI pricing per 1M tokens used in this course's lessons
PRICING = {
    "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def llm_structured(client, system_prompt, user_prompt, response_model, model="gpt-5.4-mini"):
    """
    Call the OpenAI Responses API with structured output.

    Args:
        client: OpenAI client instance.
        system_prompt: Developer/system instruction for the LLM.
        user_prompt: User message content (string).
        response_model: A Pydantic BaseModel subclass — the API returns a
                        parsed instance of this model.
        model: Model ID string (default gpt-5.4-mini).

    Returns:
        Tuple of (parsed_response_model_instance, usage_object).
    """
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=response_model,
    )
    return response.output_parsed, response.usage


def llm_structured_retry(
    client, system_prompt, user_prompt, response_model,
    model="gpt-5.4-mini", max_retries=3, delay=1
):
    """
    Call llm_structured with exponential-backoff retry logic.

    Retries on any exception up to `max_retries` times, sleeping
    `delay * (attempt + 1)` seconds between attempts.
    """
    for attempt in range(max_retries):
        try:
            return llm_structured(client, system_prompt, user_prompt, response_model, model)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def calc_price(usage, model="gpt-5.4-mini"):
    """
    Calculate USD cost for a single API usage object.

    Args:
        usage: Object with .input_tokens and .output_tokens int attributes.
        model: Model ID used for pricing lookup.

    Returns:
        float cost in USD.
    """
    pricing = PRICING.get(model, PRICING["gpt-5.4-mini"])
    input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def calc_total_price(usage_list, model="gpt-5.4-mini"):
    """
    Sum the cost of multiple API usage objects.

    Args:
        usage_list: Iterable of usage objects with .input_tokens, .output_tokens.
        model: Model ID used for pricing lookup.

    Returns:
        float total cost in USD.
    """
    return sum(calc_price(u, model) for u in usage_list)


def map_progress(func, items, max_workers=4, desc="Processing"):
    """
    Apply a function to every item in parallel with a progress bar.

    Preserves input order in the returned results list.

    Args:
        func: Callable taking a single item from `items`.
        items: List of inputs to process.
        max_workers: Number of parallel threads.
        desc: Label shown on the tqdm progress bar.

    Returns:
        List of results in the same order as `items`.
    """
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_map = {executor.submit(func, item): i for i, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(fut_map):
                idx = fut_map[future]
                results[idx] = future.result()
                pbar.update(1)
    return results
