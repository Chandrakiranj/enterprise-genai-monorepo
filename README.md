# ⚡ Enterprise GenAI Production Monorepo

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware Acceleration](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-success?logo=apple&logoColor=white)](#-project-1-hardware-accelerated-model-engineering-simulation)
[![LLMOps Gating](https://img.shields.io/badge/LLMOps-DeepEval%20%7C%20CI%2FCD-orange?logo=pytest&logoColor=white)](#-project-2-multi-stage-hybrid-rag--llm-as-a-judge-gate)
[![Agentic Framework](https://img.shields.io/badge/Agent-Stateful%20ReAct%20Loop-purple?logo=ai&logoColor=white)](#-project-3-stateful-autonomous-react-orchestrator)

A highly optimized, production-engineered monorepo showcasing local Large Language Model (LLM) fine-tuning topologies, multi-stage hybrid retrieval architectures, local automated evaluation suites, and stateful autonomous agentic systems.

---

## 🏗️ System Architecture Topology

```text
==================================================================================================
                                CENTRAL RUNTIME REPOSITORY ENVIRONMENT
==================================================================================================
                                               │
        ┌──────────────────────────────────────┼──────────────────────────────────────┐
        ▼                                      ▼                                      ▼
 🏋️  PROJECT 1: MODEL ENG              🔍  PROJECT 2: LLMOPS RAG              🤖  PROJECT 3: AGENTS
 ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
 │ • PEFT / QLoRA Framework  │         │ • BM25 + FAISS Indexing   │         │ • Stateful ReAct Loop     │
 │ • Target Attention Layers │   ───>  │ • FlashRank Cross-Encoder │   ───>  │ • Tool Call Reflection    │
 │ • Native Apple MPS (FP16) │         │ • Local Qwen 3B Judge     │         │ • Policy Guardrails       │
 └───────────────────────────┘         └───────────────────────────┘         └───────────────────────────┘
