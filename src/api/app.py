import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from fastapi import FastAPI, HTTPException
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.agents.guardrails import BoundaryGuardrail
from src.retrieval.engine import HybridRetrievalEngine
from src.agents.orchestrator import LangGraphOrchestrator

app = FastAPI(
    title="Unified Enterprise GenAI Platform API Node", 
    version="2026.1.0"
)

class InferenceRequest(BaseModel):
    user_query: str = Field(..., example="Verify regulatory retention guidelines.")

class InferenceResponse(BaseModel):
    status: str
    output: str
    graph_cycles: int

with open("config/platform_meta.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

mock_enterprise_corpus = [
    "Section 4.1: Financial institutions must review third-party risk protocols quarterly.",
    "Section 9.3: All data retention policies must encrypt transactional log footprints at rest."
]

guard = BoundaryGuardrail(config=config_dict)
retrieval = HybridRetrievalEngine(corpus=mock_enterprise_corpus, dimensions=config_dict['retrieval_configuration']['vector_dimension'])
orchestrator = LangGraphOrchestrator(
    guardrail_engine=guard, 
    retrieval_engine=retrieval, 
    max_iterations=config_dict['safety_configuration']['max_agent_iterations']
)

@app.post("/api/v1/inference", response_model=InferenceResponse)
async def process_async_network_inference(payload: InferenceRequest):
    try:
        initial_state = {
            "query": payload.user_query,
            "context": [],
            "raw_response": "",
            "iteration_count": 0,
            "security_clearance": False
        }
        result_state = orchestrator.app.invoke(initial_state)
        egress_check = guard.screen_output_payload(result_state['raw_response'])
        if not egress_check.is_safe:
            raise HTTPException(status_code=400, detail=f"Egress Fault: {egress_check.risk_category}")

        return InferenceResponse(
            status="SUCCESS",
            output=result_state['raw_response'],
            graph_cycles=result_state['iteration_count']
        )
    except HTTPException as http_err:
        raise http_err
    except Exception as general_err:
        raise HTTPException(status_code=500, detail=f"Runtime Blueprint Server Exception: {str(general_err)}")
