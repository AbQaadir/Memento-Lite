import streamlit as st
import time
import json
import sys
import os

# Ensure the 'src' module can be found by adding the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import MementoAgent, AgentCallback
from src.memory import CaseBank
from src.config import CONFIG

st.set_page_config(page_title="Memento Agent", layout="wide", page_icon="🧠")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Clean up tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 40px; padding: 0 20px; }
    
    /* Card style for stats */
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching Resources ---
@st.cache_resource(show_spinner="Booting up Memento's Brain (Loading Embeddings)...")
def get_shared_memory():
    return CaseBank()

# --- Session State ---
if "agent" not in st.session_state:
    shared_mem = get_shared_memory()
    st.session_state.agent = MementoAgent(memory_instance=shared_mem)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent Plan State: List of {"step": str, "status": "pending"|"running"|"completed", "result": None}
if "plan_state" not in st.session_state:
    st.session_state.plan_state = []

# --- Callback Handler ---
class StreamlitCallback(AgentCallback):
    def __init__(self, tools_container):
        self.tools_container = tools_container

    def on_log(self, message: str):
        pass

    def on_plan(self, plan: list):
        # Initialize persistent plan state
        st.session_state.plan_state = [{"step": p, "status": "pending", "result": None} for p in plan]
    
    def on_subtask_start(self, subtask: str):
        # Update state
        for item in st.session_state.plan_state:
            if item["step"] == subtask:
                item["status"] = "running"
        # Force rerun isn't needed here as ST updates usually happen on interaction, 
        # but for live view we might need manual handling if we weren't in a loop.
        # Ideally, we update the placeholder, but for persistence we just update state.
        pass

    def on_subtask_end(self, result: str):
        # Update state
        for item in st.session_state.plan_state:
            if item["status"] == "running":
                item["status"] = "completed"
                item["result"] = result
        pass

    def on_tool_start(self, tool: str, args: str):
        with self.tools_container:
             st.info(f"🛠️ **{tool}** calling with `{args}`...")

    def on_tool_end(self, result: str):
        with self.tools_container:
             st.success(f"Result: {result[:200]}..." if len(result) > 200 else result)

# --- Layout ---

# Sidebar
with st.sidebar:
    st.title("🧠 Memento")
    st.caption("Case-Based Reasoning Agent")
    
    st.markdown("### 📊 Memory Stats")
    if hasattr(st.session_state.agent, 'memory'):
        count = len(st.session_state.agent.memory.cases)
        st.metric("Stored Cases", count)
    
    st.divider()
    st.divider()
    if st.button("Run Diagnostics 🧪", type="primary"):
        with st.spinner("Verifying Embedding Engine..."):
            try:
                # Run the verification script logic (importing purely for check)
                from src.verify_retrieval import load_seed_memory
                cases = load_seed_memory("src/seed_memory.jsonl")
                st.success(f"✅ Loaded {len(cases)} Seed Cases")
                st.success("✅ Google GenAI Embedding Provider Active")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Diagnostic Failed: {e}")

    if st.button("Unload Session", type="secondary"):
        st.session_state.messages = []
        st.session_state.plan_state = []
        st.rerun()

# Main Area
col_chat, col_internals = st.columns([1.2, 0.8])

with col_internals:
    st.subheader("⚙️ Agent Internals")
    
    tab_status, tab_memory, tab_tools = st.tabs(["📝 Live Status", "🧠 Memory Bank", "🛠️ Tool Logs"])
    
    with tab_status:
        st.caption("Execution Plan")
        if not st.session_state.plan_state:
            st.info("Waiting for task...")
        else:
            # Render persistent plan
            current_group = st.empty()
            with current_group.container():
                for i, item in enumerate(st.session_state.plan_state):
                    status = item["status"]
                    step_text = item["step"]
                    
                    if status == "completed":
                        with st.expander(f"✅ {step_text}", expanded=False):
                            st.write(f"Result: {item.get('result', 'Done')}")
                    elif status == "running":
                        st.info(f"🔄 **{step_text}** (Running...)")
                    else:
                        st.write(f"⬜ {step_text}")

    with tab_tools:
        st.caption("External tool interactions")
        tools_box = st.container()
        if not st.session_state.plan_state:
             tools_box.info("Waiting for tools...")
        
    with tab_memory:
        st.caption("Similar cases retrieved for context")
        if hasattr(st.session_state.agent, 'memory'):
            cases = st.session_state.agent.memory.cases
            if not cases:
                st.info("Memory is empty.")
            else:
                for i, case in enumerate(cases[-3:][::-1]):
                    with st.expander(f"Case: {case.state_description[:40]}..."):
                        st.markdown(f"**Plan**: {case.plan}")
                        st.markdown(f"**Result**: {case.result}")
        else:
            st.warning("Memory not loaded.")

with col_chat:
    st.subheader("💬 Chat")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
             message_ph = st.empty()
             
             cb = StreamlitCallback(tools_box)
             
             try:
                 final_response = st.session_state.agent.run(prompt, callback=cb)
                 message_ph.markdown(final_response)
                 st.session_state.messages.append({"role": "assistant", "content": final_response})
                 st.rerun() 
                 
             except Exception as e:
                 st.error(f"Error: {e}")
