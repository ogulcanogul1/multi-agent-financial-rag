
from src.loaders.base import BaseLoader
from src.schemas.document import Document
from pathlib import Path
from typing import List

class LocalTextLoader(BaseLoader) :
    
    def __init__(self):
        self.path = Path("src/data/txt/")


    def load(self) -> List[Document]:

        documents:List[Document] = []
        
        for file in self.path.glob("*.txt"):
            content = file.read_text(encoding="utf-8")

            documents.append(
                Document(
                    content=content,
                    metadata={
                        "source":"text",
                        "file_name":file.name
                    }
                )
            )

        return documents
    

if __name__ == "__main__":

    text_loader = LocalTextLoader() 

    docs = text_loader.load()

    print(docs[0].content)

    print("-")