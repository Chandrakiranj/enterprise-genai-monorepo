PLATFORM ANCHOR: APPLE SILICON / MPS
                                
  +---------------------------------------------------------------------------------------+
  |                        PROJECT 1: TRAINING & OPTIMIZATION PIPELINE                     |
  |  - Dataset Engine: src/training/dataset.json (Domain Compliance Ingestion)           |
  |  - PEFT Scaling: QLoRA Optimization Loop -> 1,081,344 Trainable Parameters Flagged    |
  |  - Convergence Telemetry: True Cross-Entropy Loss Reduced Systematically to 5.353     |
  +-------------------------------------------+-------------------------------------------+
                                              |
                                              | Exports Validated
                                              | Adapter Weights
                                              v
  +-------------------------------------------+-------------------------------------------+
  |                        PROJECT 3: LOCAL PRODUCTION APPLICATION ENGINE                 |
  +---------------------------------------------------------------------------------------+
  |  - User Interaction Layer: src/evaluation/dashboard.py (Premium Cyber Dashboard)     |
  |  - Orchestration Node: LangGraph Deterministic State Machine (Cyclic Routing Node)    |
  |  - Spatial Search Index: FAISS Vector Core (Local Cosine Similarity Store)            |
  +-------------------------------------------+-------------------------------------------+
                                              |
                                              | Inference Responses
                                              | Dispatched for Audit
                                              v
  +-------------------------------------------+-------------------------------------------+
  |                        PROJECT 2: CI/CD EVALUATION & DIAGNOSTICS                      |
  +---------------------------------------------------------------------------------------+
  |  - Automation Harness: src/evaluation/test_gateway.py (Native DeepEval Execution)     |
  |  - Verification Matrix A: Faithfulness Metric Check (100% Real-World Grounding Pass)   |
  |  - Verification Matrix B: Answer Relevancy Metric Check (100% Structural Intent Pass)  |
  +---------------------------------------------------------------------------------------+
