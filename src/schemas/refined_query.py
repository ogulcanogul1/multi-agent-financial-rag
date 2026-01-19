from pydantic import BaseModel, Field, field_validator
from typing import Literal

class RefinedQuery(BaseModel):
    """
    The output model of the LLM, including its search strategy and optimized query.
    """

    # 1. ADIM: Düşünce Zinciri (Kaliteyi Artırır)
    reasoning: str = Field(
        description="The logic behind the refinement. Explain why the original query failed and which strategy (Internal Synonyms vs. External Pivot) was chosen."
    )

    # 2. ADIM: Sinyal (Router'a Kopya Verir)
    suggested_tool: Literal["archive", "web"] = Field(
        description="The recommended tool for the refined query. Use 'web' if looking for current/external data, otherwise 'archive'."
    )

    # 3. ADIM: Eylem (Yeni Sorgu)
    optimized_query: str = Field(
        min_length=5, # 10 biraz fazla olabilir, 5 iyidir
        description="The final refined query string optimized for the chosen tool."
    )

    @field_validator("optimized_query")
    @classmethod
    def no_empty_queries(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("optimized_query cannot be empty")
        return v