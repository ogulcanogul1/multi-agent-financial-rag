from abc import ABC,abstractmethod
from typing import List

class BaseEmbedder:

    @abstractmethod
    def embed_queries(self,queries:List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_documents(self,texts:List[str]) -> List[List[float]]:
        pass
