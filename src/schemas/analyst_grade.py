from pydantic import BaseModel, Field
from typing import Literal

class AnalystGrade(BaseModel):
    """Evaluation result of the Financial Analyst's output."""
    
    binary_score: Literal["yes", "no"] = Field(
        description="Set to 'yes' if the analysis is logical, mathematically correct, and directly answers the question; otherwise 'no'."
    )
    explanation: str = Field(
        description="A concise rationale for the grading decision (e.g., 'Calculation failed due to missing data' or 'The logic is sound')."
    )