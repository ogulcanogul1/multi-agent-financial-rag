from src.embeddings.huggingface import HuggingFaceEmbedder
from src.vectorstores.pinecone_db import PineconeVectorStore
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.rerankers.flashrank_reranker import FlashRankReranker
import pickle
from src.settings.configurations import PINECONE_API_KEY,PINECONE_INDEX_NAME 

def main():
    
    with open("data/processed/chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)

    embedder = HuggingFaceEmbedder()
    v_store = PineconeVectorStore(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
    
    
    v_retriever = VectorRetriever(embedder, v_store)
    b_retriever = BM25Retriever(all_chunks)
    h_retriever = HybridRetriever(v_retriever, b_retriever, alpha=0.6)
    
    
    reranker = FlashRankReranker()

    query = "Deep learning pioneers" # Kendi döküman içeriğine göre bir soru sor
    print(f"\nSorgu: {query}")
    
    candidates = h_retriever.retrieve(query, top_k=10)
    
    final_results = reranker.rerank(query, candidates)[:3]

    # 5. Sonuçları Bas
    for i, res in enumerate(final_results):
        print(f"\n[{i+1}] Skor: {res.score:.4f} | ID: {res.chunk.chunk_id}")
        print(f"Metin: {res.chunk.text[:200]}...")

if __name__ == "__main__":
    main()