from typing import TypedDict, Optional, Dict, Any
import uuid

class Document:

    def __init__(self,content:str,metadata:Optional[Dict[str,Any]],id:str=None):

        if(id == None):
            self.id= str(uuid.uuid4())

        self.id = id
        self.content = content
        self.metadata = metadata
