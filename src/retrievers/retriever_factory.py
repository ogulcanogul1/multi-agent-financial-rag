from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.rerankers.flashrank_reranker import FlashRankReranker

class RetrieverFactory:
    """
    Retriever bileşenlerini merkezi bir noktadan yöneten Factory sınıfı.
    """
    
    @staticmethod
    def create_hybrid_retriever(vector_db, all_chunks, alpha=0.6) -> HybridRetriever:
        # 1. Vektör tabanlı getirme birimi (Semantic)
        vector_retriever = VectorRetriever(vector_db=vector_db)
        
        # 2. Kelime bazlı getirme birimi (Lexical/BM25)
        bm25_retriever = BM25Retriever(documents=all_chunks)
        
        # 3. Yeniden sıralama birimi (Cross-Encoder)
        reranker = FlashRankReranker()
        
        # Hepsini HybridRetriever içinde birleştiriyoruz
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            alpha=alpha,
            reranker=reranker
        )