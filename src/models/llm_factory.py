from langchain_community.chat_models import ChatOllama
from src.schemas.graded_documents import GradeDocuments
from src.schemas.refined_query import RefinedQuery
from src.schemas.financial_report import FinancialReport
from src.schemas.analyst_output import AnalystOutput
from src.schemas.analyst_grade import AnalystGrade
from src.schemas.execution_plan import ExecutionPlan

llama_model = "llama3"

def get_grader_llm():
    # Modelin yapılandırılmış çıktı (Structured Output) vermesini sağlıyoruz
    # Llama 3 8B veya benzeri bir model finansal analiz denetimi için yeterlidir
    llm = ChatOllama(model=llama_model, temperature=0) # Analiz için 0 temperature şart
    return llm.with_structured_output(GradeDocuments)


def get_refiner_llm():
    
    llm = ChatOllama(model=llama_model, temperature=0) 
    return llm.with_structured_output(RefinedQuery)


def get_reporter_llm():
    llm = ChatOllama(model=llama_model,temperature=0.2)
    return llm.with_structured_output(FinancialReport)

def get_analyst_llm():
    llm = ChatOllama(model="llama3", temperature=0.1)
    return llm.with_structured_output(AnalystOutput)

def get_analyst_grader_llm():
    llm = ChatOllama(model="llama3", temperature=0) 
    return llm.with_structured_output(AnalystGrade)

def get_planner_llm():
    llm = ChatOllama(model="llama3", temperature=0, format="json")
    return llm.with_structured_output(ExecutionPlan)