from typing import Annotated, List, Union, Optional
from typing_extensions import TypedDict
import operator

# State: Node'lar arasında taşınan ortak 'Session' objesi
class AgentState(TypedDict):
    # Kullanıcı girişi
    input: str
    
    # Sohbet geçmişi (operator.add sayesinde her yeni mesaj listeye eklenir)
    chat_history: Annotated[List[dict], operator.add]
    
    # Planner (Node 1) tarafından oluşturulan görev listesi
    plan: List[str]
    
    # Şu an işlenen alt görev (Node 2 tarafından belirlenir)
    current_task: Optional[str]
    
    # Tamamlanan görevlerin kaydı
    completed_tasks: Annotated[List[str], operator.add]
    
    # Node 3A, 3B ve 3C'den gelen ham veriler
    retrieved_docs: List[str]
    
    # Grader (Node 4) sonucu: 'good' veya 'bad'
    grade_status: str
    
    # Final Rapor (Node 7)
    final_report: str
    
    # Sonsuz döngüyü kırmak için sayaç (grader,refiner,archive)
    retry_count:int

    # Planner -> Router arasındaki sonsuz döngüyü kırmak için

    # TODO: main.py dosyasında, her yeni /chat isteği geldiğinde bu değer 0 olarak set edilmeli!
    total_steps: int