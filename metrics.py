"""
metrics.py — Instrument the RAG pipeline to capture LLM call metrics.

Every RAG call is a black box unless we record what happened. This module
adds that bookkeeping without touching the working RAGBase code:

  - LLMCallRecord  → a dataclass holding everything we know about one call
                     (model, prompt, tokens, response time, cost, timestamp)
  - calculate_cost → price a call from its token usage
  - RAGWithMetrics → a RAGBase subclass whose llm() times the request and
                     stores a fresh LLMCallRecord on self.last_call

Design note: we subclass RAGBase instead of modifying it so the plain
pipeline stays untouched. The metrics are stashed on the object rather
than returned, so the existing rag() return type (a string) is unchanged.
This is fine for a single-user app; a shared concurrent service would need
a thread-safe store.

Follows the llm-zoomcamp lesson "Capturing Metrics":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/04-metrics.md

Pricing note: calculate_cost only prices gpt-5.4-mini (kept verbatim from
the lesson). This repo runs llama-3.3-70b-versatile via Groq, so cost
currently records as 0.0 — extend the function with your provider's rates
to get real cost figures.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase


@dataclass
class LLMCallRecord:
    """Everything we want to keep about a single LLM call."""

    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    # Automatically set to the moment the record is created
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_cost(model, usage):
    """
    Compute the dollar cost of a call from its token usage.

    The provider charges per million tokens, so we multiply each count
    by its rate and divide by a million. `usage` is the token-count
    object that comes straight from the LLM response.
    """
    cost = 0
    # Only gpt-5.4-mini is priced here (lesson keeps it self-contained)
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost


class RAGWithMetrics(RAGBase):
    """
    RAGBase subclass that records metrics for every LLM call.

    Only llm() is overridden; search, prompt building and rag() stay
    inherited, so instrumenting takes effect everywhere the LLM is called.
    After each call, the latest metrics are on self.last_call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Last captured call metrics (None until the first LLM call)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        """Time the LLM call, log its metrics, and return the text answer."""
        # Time the request while we still have the client call in scope
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        # Record tokens, cost, timings on self.last_call
        self._log_response(prompt, response, response_time)
        return response.output_text

    def _call_llm(self, prompt):
        """Send the prompt to the LLM and return the raw API response."""
        # Message payload: system behaviour comes from self.instructions
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        # Call the LLM client creation API
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response

    def _log_response(self, prompt, response, response_time):
        """Build an LLMCallRecord from the raw response and stash it."""
        # Token counts and cost for this specific call
        usage = response.usage
        cost = calculate_cost(self.model, usage)

        # Bundle every measured value into one readable record
        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost,
        )

        # Print so we can see what we're capturing
        print(call_record)
        self.last_call = call_record
