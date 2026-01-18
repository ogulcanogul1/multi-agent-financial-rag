from pydantic import BaseModel, Field, field_validator

class RefinedQuery(BaseModel):
    """Query refinement output for vector database retrieval."""

    optimized_query: str = Field(
        min_length=10,
        description=(
            "Final refined query optimized for vector similarity search. "
            "Should include relevant financial terminology, technical concepts, "
            "and domain-specific keywords to improve embedding recall."
        )
    )

    @field_validator("optimized_query")
    @classmethod
    def no_empty_queries(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("optimized_query cannot be empty")
        return v