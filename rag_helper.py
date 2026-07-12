# Instructions defining how the LLM should behave
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

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


