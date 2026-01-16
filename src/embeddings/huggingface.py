from typing import List
from sentence_transformers import SentenceTransformer

from src.embeddings.base_embedder import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self,model_name:str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        return self.model.encode(queries).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()