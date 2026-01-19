from typing import Annotated, List, Union, Optional
from typing_extensions import TypedDict
import operator
from src.schemas.analyst_output import AnalystOutput
from src.schemas.financial_report import FinancialReport
from src.schemas.execution_plan import PlanTask

# Bu metotla sayesinde aynı verilerin retrieval_docs da bulunma ihtimalini kaldırıyoruz eğer kodla yapmak istesek state'den retrieval docs'u her almak istediğimizde önce set'e çevir sonra tekrar list'e çevirmemiz gerekirdi bunu her yerde yapmaktansa bu yol daha iyi. 
def reduce_docs(existing: List[str], new: List[str]) -> List[str]:
    """
    Mevcut dökümanlarla yeni gelenleri birleştirir 
    ve AYNI OLANLARI (Duplicate) otomatik siler.
    Ayrıca sıralamayı korur (Önemli olan üstte kalır).
    """
    if existing is None:
        existing = []
    if new is None:
        new = []
    
    # İki listeyi birleştir
    combined = existing + new
    
    # Sıralamayı bozmadan tekilleştir (Deduplication)
    seen = set()
    unique_list = []
    for item in combined:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
            
    return unique_list

# State: Node'lar arasında taşınan ortak 'Session' objesi
class AgentState(TypedDict):
    # Kullanıcı girişi
    input: str
    
    # Sohbet geçmişi (operator.add sayesinde her yeni mesaj listeye eklenir)
    chat_history: Annotated[List[dict], operator.add]
    
    # Planner (Node 1) tarafından oluşturulan görev listesi
    plan: List[PlanTask]
    
    # Şu an işlenen alt görev (Node 2 tarafından belirlenir)
    current_task: Optional[PlanTask]
    
    # Tamamlanan görevlerin kaydı
    completed_tasks: Annotated[List[PlanTask], operator.add]
    
    # Node 3A, 3B ve 3C'den gelen ham veriler
    retrieved_docs: Annotated[List[str], reduce_docs]
    
    # Grader (Node 4) sonucu: 'good' veya 'bad'
    grade_status: Optional[str]
    
    # Final Rapor (Node 7)
    final_report: str
    
    # Sonsuz döngüyü kırmak için sayaç (grader,refiner,archive)
    retry_count:int

    # Planner -> Router arasındaki sonsuz döngüyü kırmak için

    # TODO: main.py dosyasında, her yeni /chat isteği geldiğinde bu değer 0 olarak set edilmeli!
    total_steps: int

    last_analysis_result:Optional[AnalystOutput]

    analyst_grade_status:str

    # Refiner : "Ben analizi yaptım, Arşivde iş yok, lütfen WEB'e git."
    refiner_tool_suggestion:Optional[str]

    final_response:Optional[FinancialReport]


