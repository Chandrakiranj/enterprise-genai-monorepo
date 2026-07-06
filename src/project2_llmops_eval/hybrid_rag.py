import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

class HybridRetrievalEngine:
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        print("[RAG-INIT] Initializing Sparse Index (BM25Okapi)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("[RAG-INIT] Initializing Dense Index (FAISS FlatL2 Matrix)...")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        
        np.random.seed(42)
        vectors = np.random.randn(len(corpus), self.dimension).astype('float32')
        self.index.add(vectors)
        
        print("[RAG-INIT] Loading Default FlashRank Cross-Encoder Matrix...")
        # Omitting explicit model names defaults seamlessly to working stable local assets
        self.ranker = Ranker(cache_dir="runs/flashrank_cache")

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        print(f"\n[RETRIEVE] Processing raw search queries for: '{query}'")
        
        # 1. Sparse BM25 scoring
        tokenized_query = query.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Dense Vector space retrieval simulation
        query_vector = np.random.randn(1, self.dimension).astype('float32')
        _, dense_indices = self.index.search(query_vector, top_k)
        
        # 3. Candidate Pool Union Formation
        candidate_indices = list(set(np.argsort(sparse_scores)[-top_k:].tolist() + dense_indices[0].tolist()))
        candidates = [{"id": idx, "text": self.corpus[idx]} for idx in candidate_indices]
        print(f"[STAGE-1] Candidate pool harvested. Total unique chunks collected: {len(candidates)}")

        # 4. FlashRank Cross-Encoder Re-ranking Execution Layer
        rerank_request = RerankRequest(query=query, passages=candidates)
        rerank_results = self.ranker.rerank(rerank_request)
        
        final_results = [res["text"] for res in rerank_results[:top_k]]
        print(f"[STAGE-2] Cross-Encoder re-ranking complete. Yielded top candidate: '{final_results[0]}'")
        return final_results

if __name__ == "__main__":
    compliance_docs = [
        "Section 9.3 mandates that transactional history footprints must maintain mandatory AES-256 multi-tenant encryption layers at rest.",
        "Compliance directive 4.1 requires financial ledgers to be retained for a minimum of 10 years inside air-gapped systems.",
        "Standard technical code protocol 5.4 outlines server logging formats for generic structural cluster parameters."
    ]
    engine = HybridRetrievalEngine(compliance_docs)
    engine.retrieve("What are the encryption rules for transactional history footprints?")
