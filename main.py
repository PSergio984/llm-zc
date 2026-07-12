import os
from dotenv import load_dotenv
from openai import OpenAI

# Import modular helper functions from ingest and helper
from ingest import load_faq_data, build_index, open_sqlite_index
from rag_helper import RAGBase

# Load configurations from .env
load_dotenv()

# --- Client Setup ---

api_key = os.getenv("OPENAI_API_KEY")
is_groq = api_key and api_key.startswith("gsk_")

if is_groq:
    # Setup OpenAI client routed to Groq's gateway
    openai_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    MODEL = "llama-3.1-8b-instant"
else:
    # Standard OpenAI client setup
    openai_client = OpenAI(api_key=api_key)
    MODEL = "gpt-5.4-mini"

# --- Setup ---

DB_PATH = "faq.db"

if os.path.exists(DB_PATH):
    # Persistent index exists — open it directly, no fetching or re-indexing
    print(f"Opening persistent index from {DB_PATH}...")
    index = open_sqlite_index(DB_PATH)
else:
    # No persistent index — fall back to in-memory minsearch
    print("faq.db not found. Falling back to minsearch (fetching data)...")
    print("Run ingest_sqlite.py first to build a persistent index.")
    documents = load_faq_data()
    print(f"Loaded {len(documents)} documents.")
    index = build_index(documents)

# Create an assistant using the modular class
assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    model=MODEL,
    course="llm-zoomcamp"
)

# --- Groq compatibility patch ---
# Groq doesn't support the responses.create() endpoint or "developer" role.
# We patch the llm method to use chat.completions instead.
if is_groq:
    def _llm_groq(self, prompt):
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    import types
    # Dynamically bind the patched method to the assistant instance
    assistant.llm = types.MethodType(_llm_groq, assistant)

# --- Main ---

def main():
    # Use a question with a typo to showcase agentic recovery via function calling
    question = "How do I run Olama?"

    print(f"Sending request to agentic LLM (Model: {MODEL})...")
    try:
        # Run the agentic RAG pipeline (function calling loop)
        answer = assistant.rag_agent(question)
        print("\n--- LLM Response ---")
        print(answer)
    except Exception as e:
        print(f"\nError calling LLM API: {e}")

if __name__ == "__main__":
    main()

