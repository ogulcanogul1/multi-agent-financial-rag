from src.schemas.document import Document as MyDocument
from langchain_core.documents import Document as LC_Document
import uuid

def lc_to_my_doc(lc_doc: LC_Document) -> MyDocument:
    
    return MyDocument(
        doc_id=str(uuid.uuid4()),
        text=lc_doc.page_content,
        metadata=lc_doc.metadata
    )

def lc_list_to_my_docs(lc_docs: list[LC_Document]) -> list[MyDocument]:
    return [lc_to_my_doc(doc) for doc in lc_docs]