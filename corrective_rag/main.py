from langchain_core.output_parsers import StrOutputParser
from os import mkdir
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, List
from pydantic import BaseModel,  Field
from langgraph.types import interrupt, Command
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm =  ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key = os.environ["GROQ_API_KEY"]
)

embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

pdf_path = r"corrective_rag\Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise ValueError("File not found")

loader = PyPDFLoader(pdf_path)

try:
    pages = loader.load()
    print(f"Length of pages: {len(pages)}")
except Exception as e:
    raise ValueError("Error while loading pdf")


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
)

try:
    chunks = splitter.split_documents(pages)
    print(f"Length of chunks: {len(chunks)}")
except Exception as e:
    raise ValueError(f" Error while splitting documents {e}")


persist_directory = r"C:\Users\JANARTHAN\graphcraft-rag\corrective_rag"
collection_name = "Crag-Agent"

try:
    vectorstore = Chroma.from_documents(
        persist_directory = persist_directory,
        collection_name = collection_name,
        embedding = embeddings,
        documents= chunks
    )
except Exception as e:
    raise ValueError("Error while creating vectorstore")

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k" : 5
    }
)

class CragState(TypedDict):
    question: str
    document: List[str]
    generation: str
    web_search_need: bool

class llm_schema(BaseModel):
    binary_score: str = Field(description = "yes or no wheather query is revelent to document")

llm_with_schema = llm.with_structured_output(llm_schema)


def retriever_node(state:CragState) -> CragState:

    question = state['question']
    result = retriever.invoke(question)

    return {"document" : result}

def grader(state: CragState) -> CragState:

    document =  state['document']
    question = state['question']
   
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "you're an grader agent you job is to validate wheather the give query revelent to document or not"),
        ("user", "Here document:/n{document}/n query:/n{question}/n you have to check revelent or not then return ONLY yes or no  ")
    ])

    grader_chain = grader_prompt | llm_with_schema

    filtere_docs = []
    web_search = False

    for doc in document:
        result = grader_chain.invoke({"document": doc.page_content, "question": question})

        if result.binary_score.lower() == "yes":
            filtere_docs.append(doc)
            web_search = False
        else:
            web_search = True
        
    if len(filtere_docs) == 0:
        web_search =  True
    
    return {"document": filtere_docs, "web_search_need": web_search}


def should_decide(state: CragState) -> str:

    web_search = state['web_search_need']

    if web_search:
        return "transform_query"
    return "generate"

def transfromQuery(state: CragState) -> CragState:

    question = state['question']

    transform_prompt = ChatPromptTemplate.from_messages([
        ("system",
     "You rewrite questions to be better optimized for web search. "
     "Look at the input and reason about the underlying semantic intent."),
     ("user", "here's the question: \n\n{question}\n\nFormulate an improved question.")
    ])

    transform_chain =  transform_prompt | llm | StrOutputParser()

    result = transform_chain.invoke({"question": question})

    return {"question": result}


def web_search(state: CragState) -> CragState:

    query =  state['question']

    decision = interrupt({
        "action": "Web search",
        "query": query,
        "approval": "Allow the web search?"
    })

    if decision != "y":
        return "Web Search not approved by user, Web search Canceled due to permission"
        
 
    search  =  TavilySearchResultsy(max_results=5)
    result = search.invoke(query)
    web_docs = [Document(page_content=r['content']) for r in result]

    return {"document": state['document'] + web_docs}


def generate(state : CragState) -> CragState:

    question =  state['question']
    document = state['document']

    generate_prompt =  ChatPromptTemplate.from_template(
       """you're an helpful AI assistant, answer the user question ONLY using context
       
       context : {context}

       question: {question}

       be concise an make sure that answer should be understanble by user.
       """   
    )

    generate_chain = generate_prompt | llm | StrOutputParser()
    context = "\n\n".join(doc.page_content for doc in document)  

    result = generate_chain.invoke({
        "context": context, "question": question
    })

    return {"generation": result}


graph = StateGraph(CragState)

graph.add_node("retriever", retriever_node)
graph.add_node("grader", grader)
graph.add_node("transform_query", transfromQuery)
graph.add_node("web_search", web_search)
graph.add_node("generater", generate)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "grader")
graph.add_conditional_edges(
    "grader",
    should_decide,
    {
        "transform_query":"transform_query",
        "generate":"generater"
    }
)

graph.add_edge("transform_query", "web_search")
graph.add_edge("web_search", "generater")
graph.add_edge("generater", END)

checkpoint = InMemorySaver()

Crag_agent  = graph.compile(checkpointer=checkpoint)


if __name__ == "__main__":

    config = {"configurable" : {"thread_id": "yogi1"}}
    stream_input = {"question": "what will be total returns of sp 500 at end of 2026 and also sp 500 returns rn?"}

    while True:
        response =  Crag_agent.invoke(stream_input, config , stream_mode= "values")

        if "__interrupt__" not in response:
            print(f"question: {response["question"]}")
            print(f"Response:\n{response["generation"]}")
            break
        
        interrupt_payload = response['__interrupt__'][0].value

        print(f"Permission needed:\n{interrupt_payload}")

        anwser = input("[y/n] Approve WebSearch? ")

        stream_input = Command(resume=anwser)
    

        









    



    




