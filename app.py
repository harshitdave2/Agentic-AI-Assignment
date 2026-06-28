"""
app.py — Streamlit web interface for the AI Store Agent.
Run:  streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from agent.agent import run_agent
from config.settings import LLM_PROVIDER, GEMINI_API_KEY

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Store Assistant",
    page_icon="🛒",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; }
    .header-tag {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1rem; text-align: center;
    }
    .planner-badge {
        display: inline-block;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-tag">
    <h2>🛒 AI Store Assistant</h2>
    <p>Ask me about your orders, products, or find alternatives!</p>
</div>
""", unsafe_allow_html=True)

# ─── Sample Questions ─────────────────────────────────────────────────────────

st.markdown("**💡 Try one of these:**")
sample_questions = [
    "Where is my order ORD-1002?",
    "Is there a cheaper alternative to the shoes in ORD-1001?",
    "Show me wireless headphones",
    "Find affordable running shoes",
]

cols = st.columns(2)
for i, q in enumerate(sample_questions):
    if cols[i % 2].button(q, use_container_width=True, key=f"sample_{i}"):
        st.session_state["prefill"] = q

st.divider()

# ─── Chat History ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! I'm your AI store assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input ────────────────────────────────────────────────────────────────────

prefill     = st.session_state.pop("prefill", "")
user_input  = st.chat_input("Ask about an order, product, or search…", key="chat_input")
question    = user_input or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking that up for you…"):
            response = run_agent(question)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Agent Info")

    from agent.planner import get_planner
    active_planner = get_planner()
    is_gemini = active_planner.__class__.__name__ == "GeminiProvider"
    planner_label = "🧠 Gemini Planner" if is_gemini else "⚙️ Deterministic Planner"
    planner_color = "green" if is_gemini else "blue"
    st.markdown(f"**Planner Mode:** :{planner_color}[{planner_label}]")

    st.info(
        "**Tools Available:**\n"
        "- `get_order(order_id)` — Fetch order status\n"
        "- `search_products(query)` — Search the catalog\n"
        "- `get_product(product_id)` — Get product details\n\n"
        "The agent picks the right tools automatically!"
    )

    st.markdown("### 🧪 Sample Order IDs")
    for oid in ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004"]:
        st.code(oid)

    st.markdown("### 🛍️ Sample Product IDs")
    for pid in ["PROD-201", "PROD-305", "PROD-101"]:
        st.code(pid)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi there! I'm your AI store assistant. How can I help you today?"}
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    st.markdown(
        "To use Gemini planner:\n"
        "```bash\n"
        "export LLM_PROVIDER=gemini\n"
        "export GEMINI_API_KEY=your-key\n"
        "```"
    )
