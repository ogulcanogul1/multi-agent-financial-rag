import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from src.vectorstores.base import BaseVectorStore
from src.schemas.chunk import Chunk

logger = logging.getLogger(__name__)

class PineconeVectorStore(BaseVectorStore):
    def __init__(self, api_key: str, index_name: str, namespace: str = "default"):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        
        vectors_to_upsert = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                continue

            # Metadata'da 'text' saklamak zorunludur (Retrieval sonrası LLM'e vermek için)
            payload = {
                "text": chunk.text,
                "parent_doc_id": chunk.parent_doc_id,
                **chunk.metadata
            }

            vectors_to_upsert.append({
                "id": chunk.chunk_id, # ekleme işleminde id'lere göre ekler aynı id var ise o id'deki chunkı günceller.
                "values": chunk.embedding,
                "metadata": payload
            })

        # Batch processing
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.info(f"Uploaded batch of {len(batch)} vectors.")
            except Exception as e:
                logger.error(f"Error during upsert: {e}")
                raise

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Gelişmiş arama: Metadata ile birlikte sonuçları döner.
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            return results.get("matches", [])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []