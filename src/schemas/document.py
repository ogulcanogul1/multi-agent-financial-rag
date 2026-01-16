import hashlib
from typing import Optional, Dict, Any

class Document:
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, doc_id: str = None):
        self.content = content
        self.metadata = metadata or {}
        
        
        if doc_id is None:
            self.id = self._generate_id()
        else:
            self.id = doc_id

    def _generate_id(self) -> str:
        """
        Dosya adına veya içeriğe dayalı benzersiz, 
        tekrar edilebilir bir ID üretir.
        """
        
        identifier = self.metadata.get("file_name") or self.content[:100]
        return hashlib.md5(identifier.encode()).hexdigest()[:12]

    def __repr__(self):
        return f"Document(id={self.id}, file={self.metadata.get('file_name', 'unknown')})"