import os
import requests
import minsearch
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- LLM Client Setup ---

api_key = os.getenv("OPENAI_API_KEY")
is_groq = api_key and api_key.startswith("gsk_")

if is_groq:
    openai_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    DEFAULT_MODEL = "llama-3.1-8b-instant"
else:
    openai_client = OpenAI(api_key=api_key)
    DEFAULT_MODEL = "gpt-4o-mini"

# --- Data Ingestion ---

def fetch_documents():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    courses_raw = requests.get(docs_url).json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"{url_prefix}{course['path']}"
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        documents.extend(course_response.json())

    return documents

# --- Search Index ---

def build_index(documents):
    index = minsearch.Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    index.fit(documents)
    return index

def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )

# --- LLM ---

def build_prompt(question, search_results):
    context = "\n\n".join(
        f"Section: {doc.get('section', 'N/A')}\n"
        f"Question: {doc.get('question')}\n"
        f"Answer: {doc.get('text', doc.get('answer', ''))}"
        for doc in search_results
    )
    return f"""
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."

Question:
{question}

Context:
{context}
""".strip()

def llm(prompt):
    response = openai_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Main ---

print("Fetching documents...")
documents = fetch_documents()
print(f"Loaded {len(documents)} documents.")

print("Building search index...")
index = build_index(documents)

def main():
    question = "I just discovered the course. Can I join now?"

    results = search(question)
    prompt = build_prompt(question, results)

    print(f"Sending request to LLM (Model: {DEFAULT_MODEL})...")
    try:
        answer = llm(prompt)
        print("\n--- LLM Response ---")
        print(answer)
    except Exception as e:
        print(f"\nError calling LLM API: {e}")

if __name__ == "__main__":
    main()
