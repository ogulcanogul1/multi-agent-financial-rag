from typing import List
from src.retrievers.base import BaseRetriever
from src.embeddings.base_embedder import BaseEmbedder
from src.vectorstores.base import BaseVectorStore
from src.schemas.scored_chunk import ScoredChunk
from src.schemas.chunk import Chunk

class VectorRetriever(BaseRetriever):
    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        # 1. Sorguyu vektöre çevir
        query_vector = self.embedder.embed_queries([query])[0]
        
        # 2. Vektör DB'de ara
        matches = self.vector_store.search(query_vector, top_k=top_k)
        
        # 3. Sonuçları ScoredChunk'a dönüştür
        results = []
        for match in matches:
            metadata = match.get("metadata", {}).copy()
            text = metadata.pop("text", "")
           

            start_off = metadata.get("start_offset", 0) 
            end_off = metadata.get("end_offset", 0)

            print(match) # test
            
            chunk = Chunk(
                chunk_id=match["id"],
                text=text,
                parent_doc_id=metadata.get("parent_doc_id", ""),
                metadata=metadata,
                start_offset=start_off, # Pinecone'dan dönmüyorsa default
                end_offset=end_off
            )
            results.append(ScoredChunk(chunk=chunk, score=match["score"]))
        return results
    

if __name__ == "__main__":
    pass