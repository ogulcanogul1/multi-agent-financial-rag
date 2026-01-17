from abc import ABC, abstractmethod
from typing import List
from src.schemas.scored_chunk import ScoredChunk

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        
        pass