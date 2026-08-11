"""
app.py — Streamlit chat interface for the course assistant (lessons 03 + 04).

A deliberately plain web front end that talks to the same RAG assistant
as the command line. The assistant is built once at startup via
create_assistant(); each "Ask" click runs a RAG query and displays the
answer. Because the assistant is a RAGWithMetrics (see metrics.py), we
also render the last call's response time, token usage and cost below
the answer — the visibility that the monitoring module is about.

Follows the llm-zoomcamp lessons "Chat App" and "Capturing Metrics":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/03-chat-app.md
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/04-metrics.md

Usage:
  uv run streamlit run app.py
"""

import streamlit as st
from assistant import create_assistant
from db_save import save_conversation

# Build the instrumented RAG assistant once, when the app starts
assistant = create_assistant()

st.title("Course Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        # Run the RAG pipeline (search → prompt → LLM)
        answer = assistant.rag(user_input)
        st.success("Completed!")
        st.write(answer)

        # Show the metrics captured for this call by RAGWithMetrics
        record = assistant.last_call
        st.write(f"Response time: {record.response_time:.2f}s")
        st.write(f"Prompt tokens: {record.prompt_tokens}")
        st.write(f"Completion tokens: {record.completion_tokens}")
        st.write(f"Cost: ${record.cost:.4f}")

        # Persist this call to the monitoring Postgres (lesson 05);
        # keep the id so feedback can be attached to the conversation
        conversation_id = save_conversation(record, user_input, "llm-zoomcamp")
        st.session_state.conversation_id = conversation_id
