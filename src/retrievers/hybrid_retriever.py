from typing import List, Dict
from src.retrievers.base import BaseRetriever
from src.schemas.scored_chunk import ScoredChunk
from src.rerankers.base import BaseReranker
from src.rerankers.flashrank_reranker import FlashRankReranker
from typing import Optional

class HybridRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_retriever: BaseRetriever, 
        bm25_retriever: BaseRetriever, 
        alpha: float = 0.6, # semantic search oranı
        k: int = 60,
        reranker:Optional[BaseReranker] = None
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # 1.0 = Sadece Vektör, 0.0 = Sadece BM25
        self.k = k
        self.reranker = reranker if reranker is not None else FlashRankReranker()



    def retrieve(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        # 1. Her iki taraftan adayları topla
        # Genelde hibrit aramada top_k'dan biraz fazlasını çekeriz ki 
        # harmanlandığında elimizde kaliteli veri kalsın.

        fetch_k = top_k*4
        vector_results = self.vector_retriever.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)
        
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, ScoredChunk] = {}

        # 2. Vector Search (Alpha ile ağırlıklandırılmış)
        for rank, res in enumerate(vector_results):
            chunk_id = res.chunk.chunk_id
            score = self.alpha * (1.0 / (self.k + rank + 1))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score
            chunk_map[chunk_id] = res

        # 3. BM25 (Keyword) Search ( (1 - Alpha) ile ağırlıklandırılmış)
        for rank, res in enumerate(bm25_results):
            chunk_id = res.chunk.chunk_id
            score = (1 - self.alpha) * (1.0 / (self.k + rank + 1))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = res

        # 4. Sırala ve Dön
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Aday listesini oluştur (Örn: Rerank için ilk 20 aday)
        candidate_chunks = [chunk_map[chunk_id] for chunk_id, _ in sorted_ids[:20]]
        
        # 5. RERANK İŞLEMİ (Opsiyonel Kontrol ile)
        if self.reranker and candidate_chunks:
            print(f"--- RERANKING: {len(candidate_chunks)} aday yeniden sıralanıyor ---")
            # Reranker'a sorguyu ve aday metinlerini gönderiyoruz
            final_results = self.reranker.rerank(
                query=query, 
                chunks=candidate_chunks, 
                top_k=top_k
            )
            return final_results
            
        # Eğer reranker yoksa RRF sonucunu dön
        return candidate_chunks[:top_k]