from enum import Enum
from pydantic import BaseModel,Field

class BinaryScore(str, Enum):
    yes = "yes"
    no = "no"

class GradeDocuments(BaseModel):
    """Scoring schema to assess document relevance."""
    binary_score: BinaryScore = Field(
        description="Whether the documents are sufficient to answer the financial question"
    )