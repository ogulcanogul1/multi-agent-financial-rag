from abc import ABC, abstractmethod
from typing import List
from src.schemas.scored_chunk import ScoredChunk

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        """Sorgu alır ve skorlanmış chunk listesi döner."""
        pass