from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import (
    planner_node, human_check_node, task_router_node,
    archive_rag_node, web_search_node, analyst_node,
    grader_node, query_refiner_node, sub_task_reporter_node,
    final_report_node,analyst_grader_node
)

from src.schemas.execution_plan import ToolName,PlanTask



def route_planner(state: AgentState):
    """
    NODE 1 KARARLARI:
    Planner'dan sonra nereye gidilecek?
    """
    plan = state.get("plan", [])
    completed = state.get("completed_tasks", [])
    
    # 1. SENARYO: Plan daha yeni oluşturuldu (Hiçbir görev tamamlanmadı)
    # Bu durumda önce İnsan Onayına (Human Check) gitmeliyiz.
    if len(plan) > 0 and len(completed) == 0:
        return "human_check"
    
    # 2. SENARYO: Plandaki tüm görevler bitti
    if len(plan) > 0 and len(completed) >= len(plan):
        return "finalizer"
    
    # 3. SENARYO: Planın bir kısmı bitti, sıradaki görev için Router'a git
    if len(plan) > 0:
        return "router"

    # Hata durumunda veya plan boşsa (Güvenlik için)
    return "human_check"

def route_router(state: AgentState):
    """
    NODE 2: Enum ve Literal Tabanlı Yönlendirme
    """
    
    # 1. Refiner Sinyali (Öncelik)
    suggestion = state.get("refiner_tool_suggestion")
    if suggestion:
        return suggestion 

    # 2. Mevcut Görev Nesnesi
    task_object:PlanTask = state.get("current_task")
    
    # Eğer görev yoksa varsayılan arşiv
    if not task_object:
        return "archive"

    
    tool = task_object.tool
    
    if tool == ToolName.WEB:
        return "web"
        
    elif tool == ToolName.ANALYST:
        return "analyst"
        
    elif tool == ToolName.ARCHIVE:
        return "archive"
    
    # Fallback
    return "archive"
    
    
    

def route_grader(state: AgentState):
    """NODE 4: Veri yeterliliğine göre Refiner (5) veya Reporter (6)."""
    status = state.get("grade_status")
    retries = state.get("retry_count", 0)
    MAX_RETRIES = 2 

    if status == "yes":
        return "reporter"
    
    
    if status == "no" and retries < MAX_RETRIES:
        return "refiner"
    
    
    return "reporter"

def route_analyst_grader(state: AgentState):
    """
    NODE 4B KARAR MEKANİZMASI:
    Analistin yaptığı hesaplama ve mantık geçerli mi?
    
    - 'yes' -> Raporlamaya geç (Reporter)
    - 'no'  -> İyileştirmeye git (Refiner)
    """
    
    status = state.get("analyst_grade_status", "no") 
    
    if status == "yes":
        return "reporter"
    
    return "refiner"



workflow = StateGraph(AgentState)


workflow.add_node("planner", planner_node)
workflow.add_node("human_check", human_check_node)
workflow.add_node("router", task_router_node)
workflow.add_node("archive", archive_rag_node)       # Node 3A
workflow.add_node("web", web_search_node)         # Node 3B
workflow.add_node("analyst", analyst_node)       # Node 3C
workflow.add_node("grader", grader_node)         # Node 4
workflow.add_node("refiner", query_refiner_node) # Node 5
workflow.add_node("reporter", sub_task_reporter_node) # Node 6
workflow.add_node("finalizer", final_report_node)   # Node 7 (Eski adı final_report_node idi)
workflow.add_node("analyst_grader",analyst_grader_node)




workflow.set_entry_point("planner")


workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "human_check": "human_check",
        "finalizer": "finalizer",
        "router": "router"
    }
)


workflow.add_edge("human_check", "planner")

workflow.add_conditional_edges(
    "router",
    route_router,
    {
        "archive": "archive",
        "web": "web",
        "analyst": "analyst"
    }
)


workflow.add_edge("archive", "grader")
workflow.add_edge("web", "grader")
workflow.add_edge("analyst", "analyst_grader")

workflow.add_conditional_edges(
    "analyst_grader",        
    route_analyst_grader,    
    {
        "reporter": "reporter", 
        "refiner": "refiner"    
    }
)


workflow.add_conditional_edges(
    "grader",
    route_grader,
    {
        "reporter": "reporter",
        "refiner": "refiner"
    }
)


workflow.add_edge("refiner", "router")


workflow.add_edge("reporter", "planner")

# Bitiş
workflow.add_edge("finalizer", END)



app = workflow.compile()

def save_graph_image(app, filename="src/graph/graph_output.png"):
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"✅ Grafik başarıyla '{filename}' olarak kaydedildi.")
    except Exception as e:
        print(f"❌ Görselleştirme hatası: {e}")

if __name__ == "__main__":
    save_graph_image(app)