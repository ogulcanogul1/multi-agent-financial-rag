from src.graph.state import AgentState
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.retriever_factory import RetrieverFactory
from src.vectorstores.pinecone_db import PineconeVectorStore
from src.settings.configurations import PINECONE_API_KEY , PINECONE_INDEX_NAME , NAMESPACE_FINANCE_RAG_ARCHIVE
from src.schemas.graded_documents import GradeDocuments
from src.schemas.refined_query import RefinedQuery
from src.schemas.financial_report import FinancialReport
from src.models.llm_factory import get_grader_llm,get_refiner_llm,get_reporter_llm
# src/graph/nodes.py
import pickle

# Singleton referansı tek bir yerde duruyor
_retriever_instance = None

def get_retriever():
    global _retriever_instance
    
    # Eğer daha önce oluşturulmadıysa (Lazy Initialization)
    if _retriever_instance is None:
        
        # 1. Yerel veriyi oku
        with open("data/processed/chunks.pkl", "rb") as f:
            all_chunks = pickle.load(f)
            
        # 2. Pinecone bağlantısını kur
        pinecone_manager = PineconeVectorStore(api_key=PINECONE_API_KEY,index_name=PINECONE_INDEX_NAME,namespace=NAMESPACE_FINANCE_RAG_ARCHIVE)
        
        # 3. Factory aracılığıyla hibrit yapıyı kur
        _retriever_instance = RetrieverFactory.create_hybrid_retriever(
            vector_db=pinecone_manager, 
            all_chunks=all_chunks,
            alpha=0.6
        )
        print("--- RETRIEVER HAZIR ---")
        
    return _retriever_instance





# --- 1. NODE: PLANNER ---
def planner_node(state: AgentState):
    print("--- NODE 1: PLANNER ---")
    steps = state.get("total_steps", 0)
    steps += 1
    MAX_GLOBAL_STEPS = 15

    if steps > MAX_GLOBAL_STEPS:
        
        return {
            "plan": [], # Planı boşalt
            "completed_tasks": state.get("plan", []), # Hepsini bitmiş say
            "final_report": "Üzgünüm, işlem çok uzadı ve güvenlik sınırına takıldı. Şu ana kadar bulduklarımı raporluyorum.",
            "total_steps": steps
        }
    
    return {"plan": ["Arşiv verisini tara", "Hesaplama yap"], "completed_tasks": [],"total_steps":steps}

# --- 1B. NODE: HUMAN CHECK ---
def human_check_node(state: AgentState):
    print("--- NODE 1B: HUMAN CHECK ---")
    
    return {}

# --- 2. NODE: TASK ROUTER ---
def task_router_node(state: AgentState):
    print("--- NODE 2: TASK ROUTER ---")
    plan = state.get("plan", [])
    completed = state.get("completed_tasks", [])
    
    
    next_task = next((t for t in plan if t not in completed), None)
    return {"current_task": next_task, "retry_count": 0} 

# --- 3A. NODE: RAG (ARŞİV) ---
def archive_rag_node(state: AgentState):

    print("\n--- NODE 3A: RAG (ARŞİV / HYBRID + RERANK) ---")
    
    
    query = state.get("input")
    
    
    scored_chunks = _retriever_instance.retrieve(query, top_k=5)      
    
    
    retrieved_contents = [sc.chunk.content for sc in scored_chunks]
    
    
    print(f"DEBUG: Arşivden {len(retrieved_contents)} döküman başarıyla getirildi.")

    
    return {
        "retrieved_docs": retrieved_contents,
        "grade_status": "" 
    }

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
    print("\n--- NODE 4: GRADER (DÖKÜMAN DENETİMİ) ---")
    
    question = state.get("input")
    docs = state.get("retrieved_docs")
    
    
    llm_with_grader = get_grader_llm()
    
    
    context = "\n\n".join(docs)
    result:GradeDocuments = llm_with_grader.invoke(f"Soru: {question}\n\nFinansal Metinler: {context}")
    
    print(f"DEBUG: Grader Kararı -> {result.binary_score.value}")
    return {"grade_status": result.binary_score.value}

# --- 5. NODE: QUERY REFINER ---
def query_refiner_node(state: AgentState):
    print("\n--- NODE 5: QUERY REFINER (ARAMA OPTİMİZASYONU) ---")
    
    original_query = state.get("input")
    current_retries = state.get("retry_count", 0)
    


    refine_prompt = f"""
    You are refining a user query for hybrid search over an internal financial document archive.

    Goal:
    - Improve recall in vector and keyword-based retrieval
    - Expand the query with relevant financial terminology
    - Use alternative technical terms and abbreviations
    - Do NOT add explanations or metadata

    Original query:
    {original_query}

    Output:
    A single refined query string suitable for hybrid vector search.
    """
    
    
    llm_refiner = get_refiner_llm()

    refined_query:RefinedQuery = llm_refiner.invoke(refine_prompt)

    refined_query_text = refined_query.optimized_query
    new_count = current_retries + 1
    
    print(f"--- REFINE STEP: Deneme {new_count} ---")
    print(f"DEBUG: Yeni Sorgu -> {refined_query}")

    return {
        "input": refined_query_text, 
        "retry_count": new_count
    }

# --- 6. NODE: SUB-TASK REPORTER ---
def sub_task_reporter_node(state: AgentState):
    print("\n--- NODE 6: REPORTER (FINANSAL RAPOR ÜRETİMİ) ---")
    
    question = state.get("input")
    docs = state.get("retrieved_docs")
    
    
    report_llm = get_reporter_llm()
    
    context = "\n\n".join(docs)
    
    prompt = f"""
    You are a senior financial analyst.
    Using only the documents provided below, prepare a professional, objective,
    and data-driven response to the user's question.

    Question:
    {question}

    Documents:
    {context}

    Rules:
    - Use only the information contained in the documents.
    - If there are conflicting data points, explicitly point them out.
    """
    
    
    report:FinancialReport = report_llm.invoke(prompt)
    
    print("--- RAPOR BAŞARIYLA ÜRETİLDİ ---")
    
    
    return {"final_response": report}

# --- 7. NODE: FINAL REPORT BUILDER ---
def final_report_node(state: AgentState):
    print("--- NODE 7: FINAL REPORT BUILDER ---")
    return {"final_report": "Tüm görevler bitti. Analiz hazır."}