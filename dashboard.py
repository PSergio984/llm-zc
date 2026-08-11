"""
dashboard.py — Streamlit dashboard over the saved conversation metrics.

Lesson 07 of the monitoring module: before reaching for Grafana, a
Streamlit dashboard gives real visibility for free — latency, cost and
recent conversations in one place. For many projects this is all you
need (and Postgres could be swapped for SQLite here; we keep Postgres
only because Grafana connects to it more easily later).

Layout, top to bottom:
  1. Four summary metric cards fed by get_stats(): total conversations,
     average response time, total cost, average token count.
  2. Two line charts from the last 100 conversations: cost over time and
     response time over time. We fetch whole LLMCallRecords and chart
     two columns — not the leanest approach (a dedicated query would
     fetch only timestamp + value), but fine at this volume.
  3. The 20 most recent conversations rendered as plain text snippets
     with their response time and cost.

Follows the llm-zoomcamp lesson "Streamlit Dashboard":
  https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/07-streamlit-dashboard.md

Usage (port 8501 is the chat app, so use 8502):
  uv run streamlit run dashboard.py --server.port 8502
"""

import streamlit as st
from dataclasses import asdict
import pandas as pd
from db_query import get_conversations, get_stats

st.title("Course Assistant Dashboard")

# Aggregate numbers worth watching when getting started
stats = get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats.total)
col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
col3.metric("Total cost", f"${stats.total_cost:.4f}")
col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

# Last 100 calls as a DataFrame for the time charts
records = get_conversations(limit=100)
df = pd.DataFrame([asdict(r) for r in records])

st.subheader("Cost over time")
st.line_chart(df, x="timestamp", y="cost")

st.subheader("Response time over time")
st.line_chart(df, x="timestamp", y="response_time")

st.subheader("Recent conversations")
records = get_conversations(limit=20)

# Plain text is enough to make the point; no table needed
for record in records:
    st.write(f"**{record.prompt[:80]}...**")
    st.write(f"{record.answer[:200]}...")
    st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
    st.divider()
