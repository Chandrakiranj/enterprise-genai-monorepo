import streamlit as st
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.guardrails import BoundaryGuardrail
from src.retrieval.engine import ProductionHybridEngine
from src.agents.orchestrator import LangGraphOrchestrator

st.set_page_config(page_title="Enterprise GenAI Platform Command Center", layout="wide")

st.title("🛡️ Enterprise GenAI Platform Control Dashboard")
st.subheader("Automated LangGraph Orchestration & FAISS Search Node")
st.markdown("---")

# Load configuration metadata
with open("config/platform_meta.yaml", "r") as f:
    config = yaml.safe_load(f)

# Internal persistent corporate corpus
knowledge_base = [
    "Section 4.1: Financial institutions must review third-party risk protocols quarterly to protect consumer ledger environments.",
    "Section 9.3: All data retention policies must encrypt transactional log footprints at rest using AES-256 multi-tenant key pairs.",
    "Section 11.2: Identity access management systems must enforce hardware token MFA for all privileged administrative accounts."
]

# Initialize core systems
@st.cache_resource
def bootstrap_platform_engines():
    guard = BoundaryGuardrail(config=config)
    retrieval = ProductionHybridEngine(corpus=knowledge_base)
    orchestrator = LangGraphOrchestrator(
        guardrail_engine=guard, 
        retrieval_engine=retrieval, 
        max_iterations=config['safety_configuration']['max_agent_iterations']
    )
    return orchestrator

engine = bootstrap_platform_engines()

# Setup UI layout splits
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ⚡ Live Query Interface")
    user_input = st.text_input("Enter Compliance/Audit Vector Search Target Query:", 
                              value="What are the encryption mandates under Section 9.3?")
    
    if st.button("Execute Intent Query Execution", use_container_width=True):
        initial_state = {
            "query": user_input,
            "context": [],
            "raw_response": "",
            "iteration_count": 0,
            "security_clearance": False
        }
        
        with st.spinner("Processing through localized agents and FAISS layers..."):
            final_state = engine.app.invoke(initial_state)
            
        st.success("Execution Subsystem Completed Successfully.")
        st.markdown("#### **System Output Response:**")
        st.info(final_state["raw_response"])

with col2:
    st.markdown("### 📊 Active Node Metrics")
    st.metric(label="Active Knowledge Base Context Records", value=len(knowledge_base))
    st.metric(label="Max Agent Safety Iteration Boundaries", value=config['safety_configuration']['max_agent_iterations'])
    st.markdown("### 📁 Core Reference Knowledge base")
    st.json(knowledge_base)
