import sys
import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class ProductionHybridEngine:
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        print("[INIT] Loading SentenceTransformer embedding model 'all-MiniLM-L6-v2'...")
        # Load embedding model locally on Mac CPU / Apple Silicon Core
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 1. Initialize Sparse Matrix (BM25 Lexical Engine)
        self.tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 2. Initialize Dense Matrix (FAISS Vector Index)
        # MiniLM produces vectors with exactly 384 dimensions
        self.dimensions = 384
        self.index = faiss.IndexFlatL2(self.dimensions)
        
        # Encode real corpus into 384-dimensional dense vectors
        print("[INIT] Encoding knowledge corpus into semantic vector space...")
        embeddings = self.embedding_model.encode(corpus, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"[SUCCESS] Production index stable with {len(corpus)} records.")

    def query_context(self, query: str, top_k: int = 1) -> list[str]:
        """
        Executes parallel lexical and semantic search to return highly accurate context.
        """
        # A. Sparse Keyword Matching (BM25)
        tokenized_query = query.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_idx = np.argsort(sparse_scores)[-top_k:]
        
        # B. Dense Semantic Search (FAISS + Embeddings)
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        _, dense_indices = self.index.search(query_vector, top_k)
        
        # C. Secure De-duplication and Extraction
        combined_indices = set(list(top_sparse_idx) + list(dense_indices[0]))
        retrieved_documents = [self.corpus[idx] for idx in combined_indices if idx < len(self.corpus)]
        
        return retrieved_documents
