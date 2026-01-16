from abc import ABC, abstractmethod
from typing import List
from src.schemas.document import Document


class BaseLoader(ABC):


    @abstractmethod
    def load(self) -> List[Document]:
        """Raw data → standardized Document list"""
        pass