from typing import List
from flashrank import Ranker, RerankRequest
from src.rerankers.base import BaseReranker
from src.schemas.scored_chunk import ScoredChunk, Chunk


class FlashRankReranker(BaseReranker):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        
        self.ranker = Ranker(model_name=model_name, cache_dir="cache")

    def rerank(self, query: str, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        if not chunks:
            return []

        # FlashRank'in beklediği format
        passages = [
            {
                "id": res.chunk.chunk_id,
                "text": res.chunk.text,
                "meta": res.chunk.metadata
            }
            for res in chunks
        ]

        # 2. Rerank isteği oluştur
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # 3. Yeniden sırala
        results = self.ranker.rerank(rerank_request)

        # 4. Sonuçları tekrar ScoredChunk formatına çevir
        reranked_chunks = []
        for res in results:
            chunk = Chunk(
                chunk_id=res["id"],
                text=res["text"],
                parent_doc_id=res["meta"].get("parent_doc_id", ""),
                metadata=res["meta"],
                start_offset=0,
                end_offset=0
            )
            reranked_chunks.append(ScoredChunk(chunk=chunk, score=res["score"]))

        return reranked_chunks