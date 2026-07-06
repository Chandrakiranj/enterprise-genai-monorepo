import json

class ComplianceAgent:
    def __init__(self):
        print("[AGENT-INIT] Deploying Stateful Agentic Guardrail Engine...")
        self.tools = {
            "fetch_encryption_policy": self.fetch_encryption_policy,
            "audit_ledger_retention": self.audit_ledger_retention
        }

    def fetch_encryption_policy(self, layer: str) -> str:
        return f"[TOOL-OUTPUT] Verified policy: {layer} requires strict AES-256 protocols."

    def audit_ledger_retention(self, duration_years: int) -> str:
        return f"[TOOL-OUTPUT] Verified policy: Data logs must be maintained for {duration_years} years."

    def run_react_loop(self, user_intent: str):
        print(f"\n[AGENT-START] Received Prompt: '{user_intent}'")
        
        print("[THOUGHT] Prompt demands system policy checks. Selecting tool: 'fetch_encryption_policy'")
        print("[ACTION] Invoking tool function 'fetch_encryption_policy' with parameters={'layer': 'multi-tenant'}")
        
        tool_result = self.tools["fetch_encryption_policy"](layer="multi-tenant")
        print(tool_result)
        
        print("[THOUGHT] Tool output satisfies constraint mapping requirements. Formulating finalized observation.")
        final_answer = f"Agent Executive Decision: The requested system satisfies all enterprise compliance boundaries. Reason: {tool_result}"
        
        print(f"\n[FINAL-OUTPUT] {final_answer}")
        return final_answer

if __name__ == "__main__":
    agent = ComplianceAgent()
    agent.run_react_loop("Check security constraints regarding multi-tenant data structures.")
