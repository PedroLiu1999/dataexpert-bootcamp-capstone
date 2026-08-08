"""
Streamlit Web Application for Academic Research & Personalized Study Plan Assistant on Databricks Apps.
Features OpenAlex paper discovery, Lakebase pgvector persistence, PySpark batch pipeline,
collection management, sequenced study plan generator, and AI research agent chatbot.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure current working directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent.research_agent import ResearchAgent
from src.agent.tools import tool_search_openalex_papers
from src.db.repository import (
    create_collection,
    create_learning_goal,
    create_user,
    get_collection_papers,
    get_user_by_email,
    get_user_collections,
    get_user_learning_goals,
    get_user_notes,
    get_user_reading_progress,
    init_db,
    update_reading_progress,
    vector_search_papers,
)

# Page configuration
st.set_page_config(
    page_title="OpenAlex Research & Study Plan Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Brand Palette & Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #0b132b 100%);
        color: #f8fafc;
    }

    .header-card {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        padding: 1.8rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
        margin-bottom: 2rem;
    }

    .paper-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .badge {
        background-color: #3b82f6;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_headers=True)

# Initialize database schema
init_db()

# User session state setup
DEFAULT_EMAIL = "student@databricks.com"
user = get_user_by_email(DEFAULT_EMAIL)
if not user:
    user = create_user(email=DEFAULT_EMAIL, full_name="Databricks Research Student")

user_id = user["user_id"]
agent = ResearchAgent(user_id=user_id)

# Header Banner
st.markdown("""
<div class="header-card">
    <h1>🎓 Academic Research & Study Plan Assistant</h1>
    <p>Discover papers via OpenAlex API, build Lakebase collections, sequence study plans, and converse with an AI Agent.</p>
</div>
""", unsafe_allow_headers=True)

# Main Navigation Tabs
tab_discover, tab_collections, tab_agent = st.tabs([
    "🎯 Learning Goals & Paper Discovery",
    "📚 Collections & Sequenced Study Plan",
    "🤖 AI Research Assistant Chatbot"
])

# --- Tab 1: Discovery ---
with tab_discover:
    st.subheader("1. Create a Learning Objective")
    col1, col2 = st.columns([3, 1])
    with col1:
        goal_title = st.text_input("Learning Goal Title", placeholder="e.g. Graph Neural Networks for Drug Discovery")
        goal_desc = st.text_area("Objective Details", placeholder="Understand node embeddings, message passing, and molecular graph representation.")
    with col2:
        target_level = st.selectbox("Target Level", ["Beginner", "Intermediate", "Advanced"])
        if st.button("Save Goal", use_container_width=True):
            if goal_title.strip():
                create_learning_goal(user_id, goal_title, goal_desc, target_level)
                st.success(f"Goal '{goal_title}' created!")

    st.divider()
    st.subheader("2. Search OpenAlex Papers & Ingest into Lakebase")
    search_query = st.text_input("Search Academic Papers (OpenAlex API)", placeholder="e.g. Transformers in Natural Language Processing")
    if st.button("Search & Ingest Papers", type="primary"):
        if search_query.strip():
            with st.spinner("Querying OpenAlex API & running PySpark batch ingestion..."):
                papers = tool_search_openalex_papers(search_query, limit=6)
                st.success(f"Ingested & embedded {len(papers)} papers into Lakebase!")

                for p in papers:
                    st.markdown(f"""
                    <div class="paper-card">
                        <h3>{p['title']} <span class="badge">{p.get('publication_year', 'N/A')}</span></h3>
                        <p><strong>Citations:</strong> {p.get('citation_count', 0)} | <strong>Topics:</strong> {p.get('topics', 'General')}</p>
                        <p>{p['abstract'][:300]}...</p>
                        {f'<a href="{p["open_access_url"]}" target="_blank">🔗 Read Open Access PDF</a>' if p.get('open_access_url') else ''}
                    </div>
                    """, unsafe_allow_headers=True)

# --- Tab 2: Collections & Study Plan ---
with tab_collections:
    st.subheader("Saved Collections & Sequenced Progress")
    user_collections = get_user_collections(user_id)
    if not user_collections:
        st.info("No collections created yet. Use the discovery tab or ask the AI agent to save papers!")
    else:
        coll_names = [c["name"] for c in user_collections]
        selected_coll = st.selectbox("Select Collection", coll_names)
        target_coll = next(c for c in user_collections if c["name"] == selected_coll)
        c_papers = get_collection_papers(target_coll["collection_id"])

        st.markdown(f"**Collection:** `{selected_coll}` ({len(c_papers)} papers)")
        for p in c_papers:
            st.markdown(f"""
            <div class="paper-card">
                <h4>{p['title']} ({p.get('publication_year', 'N/A')})</h4>
                <p>{p['abstract'][:250]}...</p>
            </div>
            """, unsafe_allow_headers=True)

    st.divider()
    st.subheader("Reading Progress & Sequence")
    progress_records = get_user_reading_progress(user_id)
    if not progress_records:
        st.info("No reading progress recorded yet. Ask the AI agent to generate a sequenced study plan!")
    else:
        for rp in progress_records:
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**Step {rp['sequence_order']}: {rp['title']}**")
            with c2:
                new_status = st.selectbox("Status", ["unread", "in_progress", "completed"], index=["unread", "in_progress", "completed"].index(rp.get("status", "unread")), key=f"status_{rp['progress_id']}")
                if new_status != rp.get("status"):
                    update_reading_progress(user_id, rp["paper_id"], status=new_status, sequence_order=rp["sequence_order"])
                    st.rerun()

# --- Tab 3: AI Research Agent Chatbot ---
with tab_agent:
    st.subheader("🤖 AI Research Assistant Chatbot")
    st.caption("Ask the agent to find papers, sequence a study plan, summarize research, or save items to your collection.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Research Assistant. Tell me your learning objective or ask me to search papers, sequence a study plan, or manage your collections!"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a research question or request a study plan..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI Agent thinking, executing tools, and retrieving multi-paper evidence..."):
                agent_res = agent.process_user_request(user_prompt)
                ans = agent_res["response"]
                if agent_res.get("actions_taken"):
                    ans += "\n\n**Agent Tool Actions Executed:**\n" + "\n".join(f"- `{act}`" for act in agent_res["actions_taken"])

                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
