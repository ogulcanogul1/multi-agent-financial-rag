import os
import pickle
from src.loaders.text_loader import LocalTextLoader
from src.preprocess.cleaner import TextCleaner
from src.preprocess.chunking.fixed_chunker import FixedChunker 
from src.embeddings.huggingface import HuggingFaceEmbedder
from src.vectorstores.pinecone_db import PineconeVectorStore # Bunu ekledik
from src.settings.configurations import PINECONE_API_KEY,PINECONE_INDEX_NAME 

def run_ingestion(namespace:str="txt"):
    # Load
    loader = LocalTextLoader()
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} raw documents.")

    # Clean
    cleaner = TextCleaner()
    cleaned_docs = [cleaner.clean(doc) for doc in raw_docs]
    print(f"Cleaned docs: {len(cleaned_docs)}")
    
    # Split: 
    splitter = FixedChunker(chunk_size=500, overlap=50)
    final_chunks = splitter.split_documents(cleaned_docs)
    print(f"Total chunks created: {len(final_chunks)}")
    
    # Embeddings
    embedder = HuggingFaceEmbedder()
    content = [chunk.text for chunk in final_chunks]
    embeddings = embedder.embed_documents(content)

    for chunk, vector in zip(final_chunks, embeddings):
        chunk.embedding = vector

    # Vector DB: Pinecone'a Upsert (Yükleme)
    # NOT: API Key ve Index Name bilgilerini kendi bilgilerinizle güncelleyin
    v_store = PineconeVectorStore(
        api_key=PINECONE_API_KEY, 
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace
    )
    
    print("Uploading vectors to Pinecone...")
    v_store.upsert_chunks(final_chunks)
    print("Pinecone upload complete.")

    # Local Save: BM25 (Hybrid Search) için yerel kayıt
    # Bu adım BM25Retriever'ın metinleri okuyabilmesi için gerekir ve şarttır.
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/chunks.pkl", "wb") as f:
        pickle.dump(final_chunks, f)
    print(f"Chunks saved to data/processed/chunks.pkl for BM25.")

    print("\n--- INGESTION COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    run_ingestion() # her bir kaynak için ayrı metot çağrılır.