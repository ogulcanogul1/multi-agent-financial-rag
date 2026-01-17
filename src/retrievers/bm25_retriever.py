import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from src.retrievers.base import BaseRetriever
from src.schemas.scored_chunk import ScoredChunk
from src.schemas.chunk import Chunk

class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        # keyword search için lowercase
        tokenized_corpus = [doc.text.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # En yüksek skorlu dökümanların indexlerini al
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                results.append(ScoredChunk(
                    chunk=self.chunks[idx],
                    score=float(scores[idx])
                ))
        return results