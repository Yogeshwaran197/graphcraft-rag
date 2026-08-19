from boto3.resources import factory
from boto3.resources import factory
from boto3.resources import factory
from boto3.resources import factory
from boto3.resources import factory
from boto3.resources import factory
from boto3.resources import factory
from chromadb.api.types import Document
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from  typing import TypedDict, Annotated, Sequence , List
from pydantic import BaseModel, Field


load_dotenv()


llm = ChatGroq(
    model ="llama-3.1-8b-instant"
)

embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)


document_path = "projects\Stock_Market_Performance_2024.pdf"

if not os.path.exist(document_path):
    raise ValueError("file not exist")


loader = PyPDFLoader(document_path)
try:
    pages = loader.load()
    print(f"Length : {len(pages)}")
except Exception as e:
    print("Error while loading file, {e}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(splitter)

persist_directory = r"C:\Users\JANARTHAN\graphcraft-rag\projects\chroma_db"
collection_name = "Corrective Rag"

try:
    vectorstore = Chroma.from_documents(
        persist_directory=persist_directory,
        collection_name=collection_name,
        documents=chunks,
        embedding=embeddings
    )
except Exception as e:
    print(f"Error creating VectorStore {e}")


retirever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k" : 5
    }
)


class CragAgent(TypedDict):
    question: str
    document : List[Document]
    generation : str
    web_search_need: bool


def retriver(state: CragAgent) -> CragAgent:

    question = state['question']
    result  = retirever.invoke(question)
    return {"document": result}


class grader(BaseModel):
    binary_score: str = Field(..., description="yes or no weather the document  related to given user query")

llm_with_schema = llm.with_structured_output(grader)

def grader_node(state: CragAgent) -> CragAgent:

    prompt = ChatPromptTemplate.format_messages([
    ("system",
     "You grade whether a retrieved document is relevant to a user question. "
     "Give 'yes' if it contains keywords or semantic meaning related to the "
     "question. This is a lenient filter to catch clearly irrelevant docs, "
     "not a strict correctness check."),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
   ])
    
    filtered_docs = []
    web_search_needed = False

    chain = prompt  | llm_with_schema 

    for doc in state['document']:
        result = chain.invoke({
            "document": doc.page_content,
            "question": state['question']
            })
        
        if result.binary_score.lower() == "yes" :
            filtered_docs.append(filtered_docs)
            web_search_needed = False
        
        if len(filtered_docs) == 0:
            web_search_needed = True

       
    return {"document": filtered_docs, "web_search_need": web_search_needed}


def should_decide(state: CragAgent) -> Literal["transform_query", "generate"]:

    websearch = state['web_search_need']

    if websearch == True:
        return "Transformquery"
    else:
        return "generator"


def transformer_query(state: CragAgent):

    rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite questions to be better optimized for web search. "
     "Look at the input and reason about the underlying semantic intent."),
    ("human", "Initial question:\n\n{question}\n\nFormulate an improved question.")
    ])
    
    rewriter = rewrite_prompt | llm | StrOutputParser()

    better_questions =  rewriter.invoke({
        "question": state['question']
    })

    return { "question" : better_questions}



def web_search(state : CragAgent):

    query = state['question']

    decision = interrupt({
        "action": "web search",
        "query": query,
        "approval": "approve web search?"
    })

    if decision != 'y':
        return "web search cancelled  by user, don't retry until user ask gain"

    search = TavilySearchResults(k=5)
    result =  search.invoke(query)
    web_docs = [Document(page_content = r['cotent']) for r in result]

    return {"document" :  state["document"] + web_docs}



def generate(state : CragAgent):

    generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the provided context. "
               "If the context doesn't contain the answer, say so plainly."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    generate_chain = generate_prompt | llm | StrOutputParser()

    context  = "/n/n".join(doc.page_content for doc in state['document'])

    answer = generate_chain.invoke({
        "context": context,
        "question": state["query"]
    })

    return {"generation" : answer}






    
    
    













