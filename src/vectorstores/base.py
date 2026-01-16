from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.schemas.chunk import Chunk

class BaseVectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]):
        """Chunk'ları veritabanına yükler."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Vektör bazlı arama yapar."""
        pass