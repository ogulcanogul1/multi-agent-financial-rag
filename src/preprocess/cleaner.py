import re
import unicodedata
from src.schemas.document import Document

class TextCleaner:
    def clean(self,document:Document) -> Document:

        content = document.content

        
        content = unicodedata.normalize("NFKC",content)

        content = re.sub(r"\s+"," ", content).strip()

        return Document(content=content,metadata=document.metadata.copy(),id=document.id)