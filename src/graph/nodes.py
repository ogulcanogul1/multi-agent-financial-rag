from src.graph.state import AgentState
# Buraya ileride kendi sınıflarını import edeceksin:
# from src.loaders.pdf_loader import MyCustomLoader 

# --- 1. NODE: PLANNER ---
def planner_node(state: AgentState):
    print("--- NODE 1: PLANNER ---")
    steps = state.get("total_steps", 0)
    steps += 1
    MAX_GLOBAL_STEPS = 15

    if steps > MAX_GLOBAL_STEPS:
        # Zorla bitir
        return {
            "plan": [], # Planı boşalt
            "completed_tasks": state.get("plan", []), # Hepsini bitmiş say
            "final_report": "Üzgünüm, işlem çok uzadı ve güvenlik sınırına takıldı. Şu ana kadar bulduklarımı raporluyorum.",
            "total_steps": steps
        }
    # Burada LLM'e 'input'u verip bir plan (list) oluşturmasını isteyeceğiz
    # Şimdilik örnek bir plan dönüyoruz:
    return {"plan": ["Arşiv verisini tara", "Hesaplama yap"], "completed_tasks": [],"total_steps":steps}

# --- 1B. NODE: HUMAN CHECK ---
def human_check_node(state: AgentState):
    print("--- NODE 1B: HUMAN CHECK ---")
    # Bu node genelde 'interrupt' ile durdurulur.
    return {}

# --- 2. NODE: TASK ROUTER ---
def task_router_node(state: AgentState):
    print("--- NODE 2: TASK ROUTER ---")
    plan = state.get("plan", [])
    completed = state.get("completed_tasks", [])
    
    # Yapılmamış ilk görevi seç
    next_task = next((t for t in plan if t not in completed), None)
    return {"current_task": next_task, "retry_count": 0} # Yeni Göreve başlarken sayaç 0'lanır.

# --- 3A. NODE: RAG (ARŞİV) ---
def archive_rag_node(state: AgentState):
    print("--- NODE 3A: RAG (ARŞİV) ---")
    # TODO: Buraya  src/loaders içindeki kod bağlanacak
    return {"retrieved_docs": ["Arşivden gelen döküman örneği"]}

# --- 3B. NODE: WEB SEARCH ---
def web_search_node(state: AgentState):
    print("--- NODE 3B: WEB SEARCH ---")
    return {"retrieved_docs": ["Webden gelen güncel veri"]}

# --- 3C. NODE: ANALYST TOOL ---
def analyst_node(state: AgentState):
    print("--- NODE 3C: ANALYST ---")
    return {"retrieved_docs": ["Analiz sonucu: Veriler tutarlı"]}

# --- 4. NODE: GRADER ---
def grader_node(state: AgentState):
    print("--- NODE 4: GRADER ---")
    # Burada LLM gelen veriyi puanlar. 
    # Şimdilik başarılı (good) geçiyoruz.
    return {"grade_status": "good"}

# --- 5. NODE: QUERY REFINER ---
def query_refiner_node(state: AgentState):
    print("--- NODE 5: QUERY REFINER ---")
    current_retries = state.get("retry_count", 0)
    new_count = current_retries + 1

    print(f"--- REFINE STEP: Deneme {new_count} ---")

    return {"input": state["input"] + " (daha spesifik sorgu)", "retry_count": new_count}

# --- 6. NODE: SUB-TASK REPORTER ---
def sub_task_reporter_node(state: AgentState):
    print("--- NODE 6: SUB-TASK REPORTER ---")
    return {"completed_tasks": [state["current_task"]]}

# --- 7. NODE: FINAL REPORT BUILDER ---
def final_report_node(state: AgentState):
    print("--- NODE 7: FINAL REPORT BUILDER ---")
    return {"final_report": "Tüm görevler bitti. Analiz hazır."}