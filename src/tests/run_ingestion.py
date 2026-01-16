from src.loaders.text_loader import LocalTextLoader
from src.preprocess.cleaner import TextCleaner
from src.preprocess.chunking.fixed_chunker import FixedChunker # Kendi yazdığın splitter


def main():
    # 1. Load: Raw veriyi oku
    loader = LocalTextLoader()
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} raw documents.")

    # 2. Clean: Veriyi temizle
    cleaner = TextCleaner()
    cleaned_docs = [cleaner.clean(doc) for doc in raw_docs]
    print(f"cleaned docs :{len(cleaned_docs)}")
    
    # 3. Split: Chunk'lara ayır
    splitter = FixedChunker(chunk_size=500, overlap=50)

    
    final_chunks = splitter.split_documents(cleaned_docs)
    
    print(f"Total chunks created: {len(final_chunks)}")
    
    # 4. Veri Kontrolü (Görsel Test)
    for i, chunk in enumerate(final_chunks[:3]): # İlk 3 chunk'ı kontrol et
        print(f"--- Chunk {i} ---")
        print(f"Content: {chunk.text[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print(f"Token Count: {len(chunk.text.split())}") # Basit kontrol

    print(final_chunks[-1].metadata)
if __name__ == "__main__":
    main()