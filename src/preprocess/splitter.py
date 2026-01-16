
from typing import List
from src.schemas.split_chunk import SplitChunk

class TextSplitter:
    def __init__(self,chunk_size:int,overlap:int):

        if(overlap > chunk_size):
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size=chunk_size
        self.overlap = overlap

    def split(self,text:str)-> List[SplitChunk]:
        chunks = []
        text_words = text.split()


        chunk_size = self.chunk_size
        overlap = self.overlap

        for i in range(0,len(text_words),chunk_size):

            start = max(i - overlap, 0)

            end = min(i + self.chunk_size, len(text_words))

            chunk_words = text_words[start:end]

            chunk = " ".join(chunk_words)

            last_index = end - 1

            chunks.append(SplitChunk(text=chunk,start_index=start,end_index=last_index))
        
        return chunks