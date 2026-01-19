from pydantic import BaseModel, Field
from typing import List, Literal


from enum import Enum

class ToolName(str, Enum):
    """Sistemdeki mevcut araçların sabit listesi."""
    
    ARCHIVE = "internal_archive"
    WEB = "web_search"
    ANALYST = "financial_analyst"

# 1. Tekil Görev Yapısı
class PlanTask(BaseModel):
    """Tek bir analiz adımının yapılandırılmış hali."""
    
    # İşte senin istediğin profesyonel seçim kısmı burası:
    tool: Literal[ToolName.ARCHIVE, ToolName.ANALYST, ToolName.WEB] = Field(
        description="The specific tool required to perform this step."
    )
    
    description: str = Field(
        description="Clear instruction for the tool (e.g. 'Retrieve 2024 balance sheet for Company X')."
    )

# 2. Ana Plan Yapısı
class ExecutionPlan(BaseModel):
    """Strategic roadmap generated to solve the user's financial inquiry."""
    
    reasoning: str = Field(
        description="Logic behind the plan. Explain why specific tools were selected."
    )
    
    # Artık string listesi değil, PlanTask listesi dönüyoruz
    tasks: List[PlanTask] = Field(
        min_items=1,
        max_items=5,
        description="Sequential list of structured tasks."
    )