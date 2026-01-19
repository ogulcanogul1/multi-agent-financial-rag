from pydantic import BaseModel, Field
from typing import List

class AnalystOutput(BaseModel):
    """Structured output for the Financial Analyst node."""
    
    analysis_summary: str = Field(
        description="The concise executive summary of the analysis and calculation results."
    )
    calculation_steps: List[str] = Field(
        description="Sequential mathematical or logical steps taken to reach the result (e.g., 'Assets / Liabilities = 1.5')."
    )
    identified_ratios: List[str] = Field(
        description="Standard financial ratios identified or calculated (e.g., 'Current Ratio: 1.5', 'ROI: 12%')."
    )
    confidence_score: int = Field(
        description="A score from 1 to 10 indicating confidence in the accuracy based on data availability."
    )