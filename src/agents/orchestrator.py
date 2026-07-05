from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    context: list[str]
    raw_response: str
    iteration_count: int
    security_clearance: bool

class LangGraphOrchestrator:
    def __init__(self, guardrail_engine, retrieval_engine, max_iterations: int = 3):
        self.guard = guardrail_engine
        self.retrieval = retrieval_engine
        self.max_iterations = max_iterations
        self.workflow = StateGraph(AgentState)
        self._assemble_graph_topology()

    def _security_gate_node(self, state: AgentState) -> dict:
        assessment = self.guard.screen_input_payload(state['query'])
        # Return exact state keys to patch state dictionary registers
        return {
            "security_clearance": assessment.is_safe, 
            "iteration_count": state.get('iteration_count', 0) + 1
        }

    def _context_retrieval_node(self, state: AgentState) -> dict:
        if not state.get('security_clearance', False):
            return {"context": []}
        found_chunks = self.retrieval.query_context(state['query'])
        return {"context": found_chunks}

    def _inference_generation_node(self, state: AgentState) -> dict:
        if not state.get('security_clearance', False):
            return {"raw_response": "Access Denied: Malicious Security Payloads Identified."}
        
        joined_context = " | ".join(state.get('context', []))
        simulated_generation = (
            f"Processed enterprise request for query '{state['query']}'. "
            f"Verified data points extracted from backend knowledge layer: [{joined_context}]."
        )
        return {"raw_response": simulated_generation}

    def _evaluate_loop_termination(self, state: AgentState) -> str:
        # If security cleared is False, or we've done a pass, exit instantly
        if not state.get('security_clearance', False) or state.get('iteration_count', 0) >= self.max_iterations:
            return "halt"
        return "halt"  # Forced termination after a successful execution loop to safeguard compute parameters

    def _assemble_graph_topology(self):
        # 1. Define explicit operational nodes
        self.workflow.add_node("verify_security", self._security_gate_node)
        self.workflow.add_node("gather_context", self._context_retrieval_node)
        self.workflow.add_node("execute_inference", self._inference_generation_node)

        # 2. Wire static path transitions
        self.workflow.set_entry_point("verify_security")
        self.workflow.add_edge("verify_security", "gather_context")
        self.workflow.add_edge("gather_context", "execute_inference")
        
        # 3. Establish strict conditional edge router parameters
        self.workflow.add_conditional_edges(
            "execute_inference",
            self._evaluate_loop_termination,
            {
                "halt": END,
                "proceed": "gather_context"
            }
        )
        self.app = self.workflow.compile()
