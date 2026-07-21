"""
rag_helper.py — Retrieval-Augmented Generation pipeline components.

Contains the system prompt, the prompt template, and the RAGBase class
that orchestrates search → context-building → LLM call.  Also includes
an agentic loop (rag_agent) that uses OpenAI's Responses API to let the
LLM decide when and what to search.
"""

# Instructions defining how the LLM should behave as an agent
INSTRUCTIONS = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searches. Try to expand your search by using new keywords
or corrected spellings based on the results you get from the search.

The question has to be about the course or its logistics, offtopic questions 
shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
If you can't answer the question using FAQ, don't do it yourself. Only use the 
facts from the FAQ database.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

# Template used to format the final query submitted to the LLM
PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    """
    Core RAG pipeline: retrieve relevant FAQ entries, build a prompt,
    and ask an LLM for a grounded answer.

    Supports two modes:
      - rag()       → simple retrieve-then-generate (one-shot)
      - rag_agent() → agentic loop with function-calling search
    """

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        """
        Store the search index, the LLM client, and the behavioural
        configuration (instructions, prompt template, course filter).
        """
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        """
        Query the index, boosting question-field matches and filtering
        to the configured course.
        """
        # Define boosts for indexing fields (boost question matching over sections)
        boost_dict = {"question": 3.0, "section": 0.5}
        # Filter documents based on specific course ID
        filter_dict = {"course": self.course}

        # Query the search index with boosting and filtering
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        """
        Format a list of result dicts into a readable Q&A text block
        that the LLM can consume as context.
        """
        lines = []

        # Iterate over results to build formatted reference text block
        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        """
        Combine the user's question with the retrieved context inside
        the prompt template.
        """
        # Format retrieval context block
        context = self.build_context(search_results)
        # Apply structured formatting template
        return self.prompt_template.format(question=question, context=context)

    def llm(self, prompt):
        """
        Send a single prompt (instructions + user message) to the LLM
        and return the text response.
        """
        # Construct message payload with role-based segregation
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        # Call LLM client creation API
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        """
        Simple one-shot RAG pipeline:
          1. Search the FAQ index for relevant documents.
          2. Build a context block from the results.
          3. Inject the question + context into the prompt template.
          4. Send the full prompt to the LLM.
        """
        # Step 1: Retrieve context documents from the index
        search_results = self.search(query)

        # Step 2 & 3: Format the context and build the final user prompt
        prompt = self.build_prompt(query, search_results)

        # Step 4: Call the LLM to get the final answer
        answer = self.llm(prompt)

        return answer

    def rag_agent(self, query):
        """
        Agentic RAG loop using the OpenAI Responses API.

        The LLM receives a search tool it can call multiple times.
        The loop continues until the LLM produces a plain-text message
        (no more function calls).
        """
        import json

        # Tool definition the LLM can invoke
        search_tool = {
            "type": "function",
            "name": "search",
            "description": "Search the FAQ database for entries matching the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to look up in the course FAQ."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }

        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": query}
        ]

        iteration = 1
        while True:
            print(f"\n[Agent] Iteration #{iteration}...")
            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                tools=[search_tool]
            )

            messages.extend(response.output)
            has_function_calls = False

            for item in response.output:
                if item.type == "function_call":
                    args = json.loads(item.arguments)
                    search_query = args.get("query")
                    print(f"[Agent] Searching: '{search_query}'")
                    results = self.search(search_query)
                    print(f"[Agent] Got {len(results)} results.")
                    messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(results, indent=2),
                    })
                    has_function_calls = True

                elif item.type == "message":
                    last_answer = item.content[0].text

            iteration += 1
            if not has_function_calls:
                break

        return last_answer
