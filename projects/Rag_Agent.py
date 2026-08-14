from langchain_core.messages import SystemMessage
from os import mkdir
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Annotated, Sequence
from pydantic import BaseModel,  Field

load_dotenv()

llm =  ChatGroq(
    model = "llama-3.1-8b-instant",
    api_key = os.environ["GROQ_API_KEY"]
)

embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

pdf_path = r"projects\Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise ValueError("file not found")


pdfloader = PyPDFLoader(pdf_path)
try:
    pages = pdfloader.load()
    print(f"Length of pdf {len(pages)}")
except Exception as e:
    print(f"Error loading pages {e}")
    raise


text_splitters = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
try:
    chunks = text_splitters.split_documents(pages)
    print(f"Length of Chunks: {len(chunks)}")
except Exception as e:
    print(f"Error while splitting documents")
    raise
    

persist_directory = r"C:\Users\JANARTHAN\graphcraft-rag\projects\chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    mkdir(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents = chunks,
        persist_directory = persist_directory,
        collection_name = collection_name,
        embedding = embeddings
    )
except Exception as e:
    print(f"Error while creating vectorstore")
    raise

retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k" : 5
        }
)

@tool
def retrivever_tool(query: str):
    """ This tool searches and returns the information from the Stock Market Performance 2024 document."""

    docs = retriever.invoke(query)

    if not docs:
        print("I have not found revelence document")
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retrivever_tool]

llm_with_tool = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


def llm_node(state:AgentState) -> AgentState:

    messages = state['messages']
    messages = [SystemMessage(content=system_prompt)] + messages

    response = llm_with_tool.invoke(messages)

    state['messages'] = response

    return state


tool_map = {t.name:t for t in tools}

def tool_node(state:AgentState) -> AgentState:

    messages = state["messages"][-1]

    results = []

    for tool_call in messages.tool_calls:

        tool = tool_map[tool_call['name']]
        observation = tool.invoke(tool_call['args'])

        tool_message = ToolMessage(content=observation, tool_call_id = tool_call['id'])

        results.append(tool_message)

    state['messages'] =  [messages] + results

    return state


def should_continue(state: AgentState) -> AgentState:

    last_message =state['messages'][-1]

    if last_message.tool_calls:
        return "continue"
    else:
        return "end"


graph =  StateGraph(AgentState)

graph.add_node("llm",  llm_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue":  "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "llm")

Rag_Agent = graph.compile()


def running_agent():

    print("=" * 60)
    print("Rag Agent Stock Price")
    print("=" * 60)

    while True:
        user_query = input("\nAsk : ")

        if user_query == "exit":
            break

        message = [HumanMessage(content=user_query)]

        response = Rag_Agent.invoke({
            "messages" : message
        })

        print(response['messages'][-1].content)

if __name__ == "__main__":
    running_agent()


