from typing import Optional,Dict


class Chunk:

    def __init__(
            self,
            chunk_id:str,
            text:str,
            parent_doc_id:str,
            start_offset:int,
            end_offset:int,
            metadata:Optional[Dict]
            ):
        
        self.chunk_id = chunk_id
        self.text = text
        self.parent_doc_id = parent_doc_id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.metadata = metadata
    

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id}, "
            f"doc={self.parent_doc_id}, "
            f"span=({self.start_offset},{self.end_offset}))"
        )