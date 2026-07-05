# High-Throughput Enterprise GenAI Platform Monorepo

A production-grade, highly scalable agentic infrastructure layer built to handle mission-critical corporate compliance, ledger interactions, and data auditing. This platform implements a zero-trust dual-boundary guardrail framework paired with a parallel semantic-lexical retrieval engine to eliminate hallucination, prevent adversarial exploit strings, and maintain total execution predictability.

## 🏗️ Core System Architecture
```text
   [ Client App ] -> [ FastAPI Gateway ] -> [ Ingress Guardrail ]
                                                   │
                                            (If Safe Vector)
                                                   ▼
                                        [ Hybrid Search Engine ]
                                        ├── BM25 Keyword Search
                                        └── FAISS Vector Space (MiniLM)
                                                   │
                                                   ▼
                                        [ LangGraph State Machine ]
                                                   │
                                                   ▼
                                        [ Egress Guardrail ] -> [ JSON Response ]
