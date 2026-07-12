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

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        # Store index instance and client for querying and LLM calls
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
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
        lines = []

        # Iterate over results to build formatted reference text block
        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        # Format retrieval context block
        context = self.build_context(search_results)
        # Apply structured formatting template
        return self.prompt_template.format(question=question, context=context)

    def llm(self, prompt):
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
        Executes the full Retrieval-Augmented Generation (RAG) pipeline:
        1. Search: Queries the search index to find relevant documents.
        2. Context Building: Formats the raw search results into a Q&A context block.
        3. Prompt Construction: Embeds the query and context into the prompt template.
        4. LLM Generation: Sends the prompt and system instructions to the LLM.
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
        Executes the agentic RAG pipeline using function calling:
        1. The LLM is provided with the search tool.
        2. The LLM decides whether to call the search tool.
        3. If called, search results are retrieved and passed back to the LLM.
        4. The LLM synthesizes the final answer.
        """
        import json

        # Detect if we should use standard chat completions (e.g., Groq)
        # or the new responses endpoint.
        use_chat_completions = "groq" in str(self.llm_client.base_url).lower()

        # Define the search tool schema for function calling
        if use_chat_completions:
            # Standard chat completions requires the function details nested under "function"
            search_tool = {
                "type": "function",
                "function": {
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
            }
        else:
            # Responses API (from the lesson) specifies parameters at the top level of the tool object
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

        # Initialize conversational history
        if use_chat_completions:
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": query}
            ]
        else:
            messages = [
                {"role": "user", "content": query}
            ]

        # Execute agent loop (max 5 turns to prevent infinite execution loops)
        for turn in range(5):
            print(f"\n[Agent Turn {turn + 1}] Calling LLM...")
            if use_chat_completions:
                # Use standard chat completions format
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[search_tool]
                )
                assistant_message = response.choices[0].message
                messages.append(assistant_message)

                # If no tool calls are generated, we are done
                if not assistant_message.tool_calls:
                    print("[Agent] LLM returned a text response (no tool calls).")
                    return assistant_message.content

                # Process all requested tool calls
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "search":
                        args = json.loads(tool_call.function.arguments)
                        search_query = args.get("query")
                        print(f"[Agent] LLM requested search with query: '{search_query}'")
                        # Execute search against local index
                        results = self.search(search_query)
                        print(f"[Agent] Search returned {len(results)} results.")
                        result_json = json.dumps(results, indent=2)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result_json
                        })
            else:
                # Use OpenAI responses API format (from the lesson)
                response = self.llm_client.responses.create(
                    model=self.model,
                    input=messages,
                    tools=[search_tool],
                    instructions=self.instructions
                )
                
                # If no output/tool calls are generated, we are done
                if not response.output:
                    print("[Agent] LLM returned a text response (no tool calls).")
                    return response.output_text

                # Append assistant's output to history
                messages.extend(response.output)
                
                # Process the tool calls
                has_tool_call = False
                for call in response.output:
                    if hasattr(call, "arguments") and call.arguments:
                        has_tool_call = True
                        args = json.loads(call.arguments)
                        search_query = args.get("query")
                        print(f"[Agent] LLM requested search with query: '{search_query}'")
                        # Execute search against local index
                        results = self.search(search_query)
                        print(f"[Agent] Search returned {len(results)} results.")
                        result_json = json.dumps(results, indent=2)
                        
                        messages.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": result_json,
                        })
                
                if not has_tool_call:
                    print("[Agent] LLM returned a text response.")
                    return response.output_text

        return "Agent execution exceeded max turns."



