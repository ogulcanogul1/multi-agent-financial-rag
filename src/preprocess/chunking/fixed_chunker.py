import uuid 
from typing import List

from src.schemas.document import Document
from src.schemas.split_chunk import SplitChunk
from src.schemas.chunk import Chunk
from src.preprocess.splitter import TextSplitter

class FixedChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.text_splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        
        all_chunks: List[Chunk] = []

        for doc in documents:
            # Her bir döküman için ham parçaları oluştur
            raw_chunks: List[SplitChunk] = self.text_splitter.split(doc.content)

            for idx, raw_chunk in enumerate(raw_chunks):
                # Benzersiz chunk ID oluşturma
                # document.doc_id alanının varlığından emin ol (veya doc.id kullan)
                chunk_id = f"{doc.id}_{idx}_{uuid.uuid4().hex[:8]}"

                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=raw_chunk.text,
                    parent_doc_id=doc.id,
                    start_offset=raw_chunk.start_index,
                    end_offset=raw_chunk.end_index,
                    metadata={
                        **doc.metadata.copy(), # Döküman metadatasını buraya taşıyoruz
                        "chunk_index": idx,
                        "chunking_strategy": "fixed",
                    },
                )
                all_chunks.append(chunk)

        return all_chunks