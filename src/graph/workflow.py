from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import *

# --- 1. KARAR MEKANİZMALARI (Logic) ---

def route_planner(state: AgentState):
    """
    DİYAGRAMA GÖRE NODE 1 KARARLARI:
    1. Anlamadım/Eksik Bilgi -> Human Check (1B)
    2. Tüm Görevler Tamam -> Final Report Builder (Node 7) 
    3. Sıradaki Alt Görevi Al -> Task Router (Node 2)
    """
    plan = state.get("plan", [])
    completed = state.get("completed_tasks", [])
    
    # KURAL 1: Plan boşsa veya net değilse -> İnsana sor
    if not plan: 
        return "human_check"
    
    # KURAL 2: Yapılacak iş kalmadıysa -> BİTİR (Node 7'ye git)
    # (Plan listesindeki her şey completed listesinde varsa)
    if len(completed) >= len(plan):
        return "finalizer"
    
    # KURAL 3: İş varsa -> Router'a git
    return "router"

def route_router(state: AgentState):
    """
    DİYAGRAMA GÖRE NODE 2 KARARLARI:
    Sadece Arşiv, Web veya Analyst'e yollar.
    ASLA Bitiş'e yollamaz.
    """
    task = state.get("current_task")
    
    # DÜZELTME: Eğer task boş gelirse (bir hata oluşursa),
    # diyagramda Router'dan Planner'a geri dönüş oku OLMADIĞI için
    # mecburen varsayılan araca (Archive) yönlendiriyoruz.
    if not task: return "archive" 

    task_lower = task.lower()
    
    if "arşiv" in task_lower or "eski" in task_lower:
        return "archive"
    elif "web" in task_lower or "haber" in task_lower:
        return "web"
    elif "hesap" in task_lower or "analiz" in task_lower:
        return "analyst"
    
    return "archive" # default

def route_grader(state: AgentState):
    status = state.get("grade_status")
    retries = state.get("retry_count", 0)
    MAX_RETRIES = 2 # Sigorta Sınırı (3 kez dener, olmazsa pes eder)

    # 1. Veri İyiyse -> Reporter'a git
    if status == "yes":
        return "reporter"
    
    # 2. Veri Kötü AMA Hakkımız Var -> Refiner'a git (Döngüye devam)
    elif status == "no" and retries < MAX_RETRIES:
        return "refiner"
    
    # 3. Veri Kötü VE Hakkımız Bitti -> SİGORTAYI ATTIR (Reporter'a git)
    else:
        print(f"--- MAX RETRY ({MAX_RETRIES}) ULAŞILDI. DÖNGÜ KIRILIYOR ---")
        # Burada dilersen state'e "Bulunamadı" notu düşebilirsin.
        return "reporter"

# --- 2. GRAF İNŞASI ---
workflow = StateGraph(AgentState)

# Node'ları Tanımla
workflow.add_node("planner", planner_node)
workflow.add_node("human_check", human_check_node)
workflow.add_node("router", task_router_node)
workflow.add_node("archive", archive_rag_node) # Node 3A
workflow.add_node("web", web_search_node)      # Node 3B
workflow.add_node("analyst", analyst_node)     # Node 3C
workflow.add_node("grader", grader_node)       # Node 4
workflow.add_node("refiner", query_refiner_node)# Node 5
workflow.add_node("reporter", sub_task_reporter_node) # Node 6
workflow.add_node("finalizer", final_report_node)     # Node 7

# --- 3. BAĞLANTILAR (EDGES - DİYAGRAMA SADIK) ---

# Başlangıç -> Planner
workflow.set_entry_point("planner")

# Planner -> (Human Check / Router / Finalizer)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "human_check": "human_check",
        "finalizer": "finalizer", # OK: Tüm görevler tamam
        "router": "router"        # OK: Sıradaki görevi al
    }
)

# Human Check -> Planner (Döngü)
workflow.add_edge("human_check", "planner")

# Router -> Araçlar (Sadece 3 seçenek)
workflow.add_conditional_edges(
    "router",
    route_router,
    {
        "archive": "archive",
        "web": "web",
        "analyst": "analyst"
    }
)

# Araçlar -> Grader veya Reporter
workflow.add_edge("archive", "grader")
workflow.add_edge("web", "grader")

# Analyst -> Reporter (Diyagramda 'Başarılı' oku Node 6'ya gidiyor)
workflow.add_edge("analyst", "reporter") 
# (Not: Diyagramda Analyst -> Bitiş (Hata Aldı) oku da var, basitlik için şimdilik success yolunu çizdik)

# Grader -> (İyi/Kötü)
workflow.add_conditional_edges(
    "grader",
    route_grader,
    {
        "reporter": "reporter", # Node 6
        "refiner": "refiner"    # Node 5
    }
)

# Refiner -> Arşiv (Diyagramda Node 5 -> Node 3A'ya dönüyor)
workflow.add_edge("refiner", "archive")

# Reporter -> PLANNER (Döngü: Sırada başka görev var mı?)
workflow.add_edge("reporter", "planner")

# Finalizer -> END
workflow.add_edge("finalizer", END)

# Derle
app = workflow.compile()



def save_graph_image(app, filename="src/graph/graph_output.png"):
    try:
        # Mermaid formatında görseli oluşturur ve dosyaya yazar
        png_data = app.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Grafik başarıyla '{filename}' olarak kaydedildi.")
    except Exception as e:
        # Eğer sisteminde gerekli kütüphaneler (pypydot, graphviz) eksikse hata verebilir
        print(f"Görselleştirme sırasında hata oluştu: {e}")

if __name__ == "__main__":

    save_graph_image(app)
