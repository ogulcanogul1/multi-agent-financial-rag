from src.loaders.base import BaseLoader
from src.schemas.document import Document
from langchain_community.document_loaders import PyPDFLoader
from src.mapper.lc_document_to_my_document import lc_list_to_my_docs
from typing import List
from pathlib import Path

class LocalPDFLoader(BaseLoader):

    def __init__(self, folder_path: str = "src/data/pdf/"):
        self.pdf_path = Path(folder_path)

    def load(self) -> List[Document]:
        documents: List[Document] = []

        for pdf_file in self.pdf_path.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            lc_docs = loader.load()  # List[LangChain Document]
            my_docs = lc_list_to_my_docs(lc_docs) # Langchain Document to My Document Object 
            documents.extend(my_docs)

        return documents
