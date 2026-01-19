from src.graph.state import AgentState
from src.retrievers.retriever_factory import RetrieverFactory
from src.vectorstores.pinecone_db import PineconeVectorStore
from src.settings.configurations import PINECONE_API_KEY , PINECONE_INDEX_NAME , NAMESPACE_FINANCE_RAG_ARCHIVE
from src.schemas.graded_documents import GradeDocuments
from src.schemas.refined_query import RefinedQuery
from src.schemas.financial_report import FinancialReport
from src.schemas.analyst_output import AnalystOutput
from src.schemas.analyst_grade import AnalystGrade
from src.schemas.execution_plan import ExecutionPlan,ToolName
from src.models.llm_factory import get_grader_llm,get_refiner_llm,get_reporter_llm,get_analyst_llm,get_analyst_grader_llm,get_planner_llm
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

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





# --- 1. NODE: PLANNER (LLM DESTEKLİ) ---
def planner_node(state: AgentState):
    print("\n--- NODE 1: PLANNER (STRATEJİK KARAR MERKEZİ) ---")
    
    # Adım sayacı ve durum kontrolü
    steps = state.get("total_steps", 0) + 1
    MAX_GLOBAL_STEPS = 15
    current_plan = state.get("plan", [])
    question = state.get("input", "")

    # 1. GÜVENLİK KONTROLÜ
    if steps > MAX_GLOBAL_STEPS:
        return {"plan": [], "total_steps": steps, "final_report": "İşlem sınırı aşıldı."}

    # 2. KARAR MEKANİZMASI
    if not current_plan:
        print(f"DEBUG: LLM plan oluşturuyor...")
        
        
        llm = get_planner_llm()
        planner_runnable = llm.with_structured_output(ExecutionPlan)

        
        prompt = f"""
        Role: Strategic Financial Project Manager.
        Task: Create a structured 'ExecutionPlan' for the user question.
        User Question: {question}

        # TOOLS (Select strictly from this list):
        - '{ToolName.ARCHIVE.value}': Historical reports, PDF balance sheets, internal data.
        - '{ToolName.WEB.value}': Real-time stock prices, news, market sentiment (2024-2025).
        - '{ToolName.ANALYST.value}': Calculations (P/E, ROI) and logic. USE ONLY AFTER DATA RETRIEVAL.

        # OUTPUT RULES:
        1. reasoning: Brief strategy explanation.
        2. tasks: Sequential steps.
        - If query needs 1 source -> Create 1 task.
        - If query needs multiple sources (e.g. Archive AND Web) -> YOU MUST SPLIT into separate tasks.
        - Do NOT combine different tools in one task.

        # EXAMPLES:

        ### CASE A: SINGLE SOURCE
        User: "Current stock price of NVIDIA?"
        Output:
        - reasoning: "Real-time data needed. Web search only."
        - tasks:
        1. tool: '{ToolName.WEB.value}' | description: "Search for NVIDIA current stock price."

        ### CASE B: MULTI-SOURCE (SPLIT REQUIRED)
        User: "Compare Q3 internal sales with competitor reports."
        Output:
        - reasoning: "Needs internal (Archive) and external (Web) data. Must split tasks."
        - tasks:
        1. tool: '{ToolName.ARCHIVE.value}' | description: "Retrieve Q3 internal sales figures."
        2. tool: '{ToolName.WEB.value}'     | description: "Search for competitor quarterly reports."
        3. tool: '{ToolName.ANALYST.value}' | description: "Compare internal vs external data."
        """

        
        try:
            result: ExecutionPlan = planner_runnable.invoke(prompt)
            new_plan = result.tasks 
            
            print(f"DEBUG: Plan Oluşturuldu ({len(new_plan)} Adım)")
            for task in new_plan:
                print(f"  - [{task.tool.upper()}]: {task.description}")
            
            return {
                "plan": new_plan,
                "completed_tasks": [],
                "total_steps": steps
            }
            
        except Exception as e:
            print(f"HATA (Planner): {e}")
            # Hata durumunda boş plan dönerek Human Check'e düşmesini sağla
            return {"plan": [], "total_steps": steps}

    # 3. MEVCUT PLANI KORU
    return {"plan": current_plan, "total_steps": steps}


# --- 1B. NODE: HUMAN CHECK ---
def human_check_node(state: AgentState):
    print("--- NODE 1B: HUMAN CHECK ---")
    
    return {}

# --- 2. NODE: TASK ROUTER ---
def task_router_node(state: AgentState):
    
    print("--- NODE 2: TASK ROUTER ---")
    
    
    # Veri refinerdan mı gelicek onu kontrol edeceğiz
    refiner_suggestion = state.get("refiner_tool_suggestion")
    
    if refiner_suggestion:
        print(f"--- ROUTER: Refiner sinyali algılandı ({refiner_suggestion.upper()}) ---")
        print("--- ROUTER: Mevcut görev korunuyor, yönlendirme bekleniyor... ---")
        
        # Gerekli işlemler refinerdan yapıldı
        
        return {} 

    # Refiner değil plannerdan gelen veri
    plan = state.get("plan", [])
    completed = state.get("completed_tasks", [])
    
    # Planda olup bitenler listesinde olmayan ilk görevi bul
    next_task = next((t for t in plan if t not in completed), None)

    if not next_task:
        print("--- ROUTER: Yapılacak yeni görev kalmadı. ---")
        # Normalde route_planner buraya göndermemeli ama güvenlik için:
        return {"current_task": None}
    
    print(f"--- ROUTER: Yeni Görev Seçildi -> {next_task} ---")
    
    return {
        "current_task": next_task,      # Yeni görevi state'e yaz
        "retry_count": 0,               # Yeni görev için sayaç sıfırlanır
        "refiner_tool_suggestion": None, # Eski refine'dan gelen verileri temizle
        "last_analysis_result":None,
        "analyst_grade_status": None,
        "grade_status": None
    }

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
    print("\n--- NODE 3B: WEB SEARCH (CANLI İNTERNET ARAMASI) ---")
    
    question = state.get("input")
    
    print(f"DEBUG: Aranıyor -> '{question}'")

    try:
        
        web_tool = TavilySearchResults(k=3)
        
        search_results = web_tool.invoke({"query": question})
        
        
        docs = []
        if search_results:
            for result in search_results:
                content = result.get("content", "")
                url = result.get("url", "")
                # Metni ve kaynağı birleştirip listeye ekliyoruz
                docs.append(f"Content: {content}\nSource: {url}")
        else:
            docs = ["Web aramasında sonuç bulunamadı."]

        print(f"DEBUG: {len(docs)} adet web sonucu bulundu.")

        # 5. State Güncelle
        return {
            "retrieved_docs": docs, 
            "current_task": "web_search_complete" 
        }

    except Exception as e:
        print(f"HATA (Web Search): {e}")
        return {
            "retrieved_docs": [f"Arama sırasında hata oluştu: {str(e)}"],
            "current_task": "web_search_failed"
        }

# --- 3C. NODE: ANALYST TOOL ---
def analyst_node(state: AgentState):
    print("\n--- NODE 3C: ANALYST (HESAPLAMA VE MANTIK) ---")
    
    question = state.get("input")
    
    
    context_docs = state.get("retrieved_docs", [])
    context_text = "\n".join(context_docs) if context_docs else "No additional documents needed, just focus on the question."

    
    analyst_runnable = get_analyst_llm()

    prompt = f"""
    You are an Expert Quantitative Financial Analyst.
    
    Your Task: Perform a deep logical analysis or mathematical calculation based on the user's query.
    
    User Query: {question}
    Available Context (if any): {context_text}
    
    Instructions:
    1. Identify the core financial question (e.g., specific ratio, trend, or comparison).
    2. Show your work steps clearly in 'calculation_steps'.
    3. Be precise with numbers. If data is missing, state it in the summary.
    4. Do not just summarize text; derive new insights or numbers.

    Guidelines for filling the output:
    1. **analysis_summary**: Provide a concise executive summary. If the data implies a negative trend, explicitly state the risk.
    2. **calculation_steps**: Show the math. For example, if calculating a ratio, show 'A / B = C'. Do not skip steps.
    3. **identified_ratios**: List only standard financial ratios (e.g., P/E, ROI, Debt/Equity) found or calculated.
    4. **confidence_score**: Rate 1-10 based on data availability. If context is missing, give a low score (1-3).
    
    CRITICAL: Do not invent numbers. If data is missing for a calculation, explicitly state it in the summary.
    """

    try:
        analysis_result: AnalystOutput = analyst_runnable.invoke(prompt)
        
        print(f"DEBUG: Analiz Tamamlandı. Güven Skoru: {analysis_result.confidence_score}/10")
        print(f"DEBUG: Sonuç Özeti: {analysis_result.analysis_summary[:50]}...")

        return {
            "last_analysis_result": analysis_result, # Nesne olarak saklıyoruz
            "current_task": "analyst_processing_complete"
        }

    except Exception as e:
        print(f"HATA (Analyst Node): {e}")
        # Hata durumunda boş bir model dönüp akışın kırılmasını önleyelim
        return {
            "last_analysis_result": AnalystOutput(
                analysis_summary="A technical error occurred during the analysis.",
                calculation_steps=[],
                identified_ratios=[],
                confidence_score=0
            )
        }

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

# --- 4B. NODE: ANALYST GRADER ---
def analyst_grader_node(state: AgentState):
    print("\n--- NODE 4B: ANALYST GRADER (ANALİZ DENETİMİ) ---")
    
    
    question = state.get("input")
    analysis_result = state.get("last_analysis_result") 
    
    
    if not analysis_result:
        print("DEBUG: Grader -> Analiz sonucu bulunamadı. REDDEDİLDİ.")
        return {"analyst_grade_status": "no"}
        
    if analysis_result.confidence_score <= 3:
        print(f"DEBUG: Grader -> Analyst güven skoru çok düşük ({analysis_result.confidence_score}). REDDEDİLDİ.")
        return {"analyst_grade_status": "no"}

    
    
    grader_runnable = get_analyst_grader_llm()

    
    prompt = f"""
    You are a Senior Financial Reviewer.
    
    Your Task: Review the structured output produced by a Junior Analyst to ensure it meets high financial standards.
    
    User Query: {question}
    
    Analyst's Output to Evaluate:
    - Summary: {analysis_result.analysis_summary}
    - Calculation Steps: {analysis_result.calculation_steps}
    - Confidence Score: {analysis_result.confidence_score}/10
    
    Guidelines for filling the output fields:
    
    1. **binary_score**: 
       - Set to 'yes' ONLY if the analysis directly answers the query with logical, error-free math.
       - Set to 'no' if there are math errors, hallucinations, or if the analyst admitted they don't have enough data.
       
    2. **explanation**:
       - Briefly explain your reasoning for the score. 
       - If you output 'no', specify exactly what is missing or wrong (e.g., "The calculation for Debt/Equity ratio used the wrong denominator").
       - If you output 'yes', mention why it is sufficient.
    
    Evaluation Criteria:
    - Consistency: Do the 'calculation_steps' actually lead to the 'summary' results?
    - Relevance: Did the analyst answer the specific question or just provide general info?
    - Accuracy: Are the financial formulas used (if any) standard and correct?
    """
    
    try:
        
        grade_result: AnalystGrade = grader_runnable.invoke(prompt)
        
        print(f"DEBUG: Denetçi Kararı -> {grade_result.binary_score.upper()}")
        print(f"DEBUG: Sebep -> {grade_result.explanation}")

        
        return {
            "analyst_grade_status": grade_result.binary_score
        }

    except Exception as e:
        print(f"HATA (Analyst Grader): {e}")
        
        return {"analyst_grade_status": "no"}

# --- 5. NODE: QUERY REFINER ---
def query_refiner_node(state: AgentState):
    print("\n--- NODE 5: QUERY REFINER (ARAMA OPTİMİZASYONU & YÖNLENDİRME) ---")
    
    original_query = state.get("input")
    current_retries = state.get("retry_count", 0)
    
    
    refine_prompt = f"""
    You are a Strategic Query Refiner for a financial RAG system.
    
    Context:
    - User's Original Query: "{original_query}"
    - Current Status: Failed to find relevant info in the Internal Archive (Attempt #{current_retries + 1}).
    
    YOUR MISSION:
    Analyze why the search failed and generate a new structured object with a better query and a tool suggestion.
    
    STRATEGY & INSTRUCTIONS FOR FIELDS:
    
    1. **suggested_tool**: 
       - If the query is about specific historical data, balance sheets, or internal reports -> Set to 'archive'.
       - If the query asks for "latest", "current", "2024/2025" data, market news, or generic comparisons -> Set to 'web'.
       - CRITICAL: Switching to 'web' prevents infinite loops in the archive.
       
    2. **optimized_query**:
       - If tool is 'archive': Use financial synonyms (e.g., 'revenue' instead of 'sales').
       - If tool is 'web': Create a search engine optimized query (e.g., "Company X stock price 2025 news").
       
    3. **reasoning**:
       - Briefly explain logic. Example: "Archive failed, user asks for 2025 data, switching to Web."
    """
    
    
    llm_refiner = get_refiner_llm() 
    
    
    refined_result: RefinedQuery = llm_refiner.invoke(refine_prompt)
    
    
    refined_query_text = refined_result.optimized_query
    tool_suggestion = refined_result.suggested_tool
    reasoning = refined_result.reasoning
    new_count = current_retries + 1
    
    
    print(f"--- REFINE STEP: Deneme {new_count} ---")
    print(f"DEBUG: Mantık      -> {reasoning}")
    print(f"DEBUG: Yeni Araç   -> {tool_suggestion.upper()}")
    print(f"DEBUG: Yeni Sorgu  -> {refined_query_text}")
    
    
    return {
        "input": refined_query_text,  # Neden? Çünkü Archive veya Web düğümü çalıştığında eski (hatalı) soruyu değil,
        "current_task": refined_query_text, # Bu alanı da güncelliyoruz ki Router eski göreve takılı kalmasın,
        "retry_count": new_count, 
        "refiner_tool_suggestion": tool_suggestion # Refiner diyor ki: "Ben analizi yaptım, Arşivde iş yok, lütfen WEB'e git."
    }

# --- 6. NODE: SUB-TASK REPORTER ---
def sub_task_reporter_node(state: AgentState):
    print("\n--- NODE 6: REPORTER (FINANSAL RAPOR ÜRETİMİ) ---")
    
    question = state.get("input")
    docs = state.get("retrieved_docs")

    current_task = state.get("current_task")
    
    
    report_llm = get_reporter_llm()
    
    context = "\n\n".join(docs)
    
    prompt = f"""
    You are a Senior Financial Analyst creating a structured report.
    
    User Question: {question}
    
    Available Documents:
    {context}
    
    ### INSTRUCTIONS FOR REPORT FIELDS:
    
    1. **summary** (Executive Overview):
       - Provide a concise, high-level answer (BLUF: Bottom Line Up Front).
       - Focus on the direct answer to the user's question.
       - Avoid minor details here.
       
    2. **details** (Deep Dive):
       - Provide comprehensive evidence, figures, and context from the documents.
       - Explain the 'Why' and 'How'.
       - If documents contain conflicting data (e.g., different dates), explicitly mention the discrepancy.
       
    3. **key_metrics** (List[str]):
       - Extract specific numbers, percentages, and financial ratios found in the text.
       - Format: "Metric Name: Value" (e.g., "Net Profit: $1.2B", "YoY Growth: 15%").
       - If no metrics are found, return an empty list.
       
    4. **sources** (List[str]):
       - List the names or titles of the documents used to derive this information.
       - This is critical for trust and verification.
    
    ### STRICT RULES:
    - **No Hallucinations**: Use ONLY the provided documents. If the answer is not in the text, state "Data not available in provided documents" in the summary.
    - **Objectivity**: Maintain a neutral, professional tone.
    """
    
    
    report:FinancialReport = report_llm.invoke(prompt)
    
    print("--- RAPOR BAŞARIYLA ÜRETİLDİ ---")
    
    
    return {
        "final_response": report,
        "completed_tasks":[current_task], 
        "retry_count":0,
        "refiner_tool_suggestion":None}

# --- 7. NODE: FINAL REPORT BUILDER ---
def final_report_node(state: AgentState):
    print("\n--- NODE 7: FINALIZER (REPORT SEALING) ---")
    
    report = state.get("final_response")  # FinancialReport object from Node 6
    
    final_output = f"""
    # FINANCIAL ANALYSIS REPORT
            
    ## Summary
    {report.summary}

    ## Detailed Analysis
    {report.details}

    ## Key Metrics
    {", ".join(report.key_metrics)}

    ---
    **LEGAL DISCLAIMER:** This report was generated by an artificial intelligence system based on internal archival documents. It does not constitute investment advice.
    """
    
    print("--- PROCESS COMPLETED: END ---")
    
    return {"final_report": final_output}