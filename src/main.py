import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from src.agents.guardrails import BoundaryGuardrail
import argparse
import yaml
from src.agents.guardrails import BoundaryGuardrail
from src.retrieval.engine import HybridRetrievalEngine
from src.agents.orchestrator import LangGraphOrchestrator

def run_native_cli_context():
    parser = argparse.ArgumentParser(description="Unified Enterprise GenAI Platform Command-Line Application.")
    parser.add_argument("--query", type=str, required=True, help="Compliance target query.")
    args = parser.parse_args()

    with open("config/platform_meta.yaml", "r") as f:
        config = yaml.safe_load(f)

    mock_corpus = [
        "Section 4.1: Financial institutions must review third-party risk protocols quarterly.",
        "Section 9.3: All data retention policies must encrypt transactional log footprints at rest."
    ]

    guard = BoundaryGuardrail(config=config)
    retrieval = HybridRetrievalEngine(corpus=mock_corpus, dimensions=config['retrieval_configuration']['vector_dimension'])
    orchestrator = LangGraphOrchestrator(guardrail_engine=guard, retrieval_engine=retrieval, max_iterations=config['safety_configuration']['max_agent_iterations'])

    state_context = {
        "query": args.query,
        "context": [],
        "raw_response": "",
        "iteration_count": 0,
        "security_clearance": False
    }

    print(f"\n[CLI ENGAGEMENT] Dispatched execution thread for: '{args.query}'")
    final_output = orchestrator.app.invoke(state_context)
    
    print("\n" + "═"*60)
    print(f" GRAPH GENAI OUTPUT LOGS:\n {final_output['raw_response']}")
    print(f" METRIC GRAPH CYCLES RECORDED: {final_output['iteration_count']}")
    print("═"*60 + "\n")

if __name__ == "__main__":
    run_native_cli_context()
