import numpy as np
import faiss
from rank_bm25 import BM25Okapi

class HybridRetrievalEngine:
    def __init__(self, corpus: list[str], dimensions: int = 128):
        self.corpus = corpus
        self.dimensions = dimensions
        self.tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        self.index = faiss.IndexFlatL2(dimensions)
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(corpus), dimensions).astype('float32')
        faiss.normalize_L2(mock_embeddings)
        self.index.add(mock_embeddings)

    def query_context(self, query: str, top_k: int = 1) -> list[str]:
        tokenized_query = query.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[-top_k:]
        
        query_vector = np.random.randn(1, self.dimensions).astype('float32')
        faiss.normalize_L2(query_vector)
        _, dense_indices = self.index.search(query_vector, top_k)
        
        combined_indices = set(list(top_sparse_indices) + list(dense_indices[0]))
        retrieved_docs = [self.corpus[idx] for idx in combined_indices if idx < len(self.corpus)]
        return retrieved_docs
