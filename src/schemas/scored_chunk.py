from dataclasses import dataclass
from src.schemas.chunk import Chunk

@dataclass
class ScoredChunk:
    chunk: Chunk    # Asıl veri (id, text, metadata)
    score: float    # Benzerlik skoru (0 ile 1 arası veya BM25 skoru)

    def __repr__(self):
        # Debug yaparken kolaylık sağlar
        return f"ScoredChunk(score={self.score:.4f}, id={self.chunk.chunk_id})"