from pydantic import BaseModel, Field
from typing import List

class FinancialReport(BaseModel):
    """Final financial report output."""

    summary: str = Field(
        description="A concise and high-level summary of the financial situation."
    )

    details: str = Field(
        description="A detailed analysis supported by figures and facts derived from the provided documents."
    )

    key_metrics: List[str] = Field(
        description="A list of key financial metrics mentioned in the analysis."
    )

    sources: List[str] = Field(
        description="Names of the documents used as sources for the information."
    )