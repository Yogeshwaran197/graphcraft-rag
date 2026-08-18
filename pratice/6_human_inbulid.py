from errno import ENFILE
from langchain_core.messages import BaseMessage
from ast import Raise
from  dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Sequence, Annotated
from langchain_community.tools import TavilySearchResults
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


@tool
def multifly(num1, num2):
    """ This node is to multiply two numbers"""
    return num1 * num2


@tool 
def WebSearch(query):
    """This node is to fetch real time date by using web searc"""

    decision = interrupt({
        "action":"search",
        "query": query,
        "approval": "Allow the web search?"
    })

    if decision != 'y':
        return "web search cancelled  by user, don't retry until user ask gain"

    
    search = TavilySearchResults()
    result = search.invoke(query)

    return str(result)


tools = [multifly, WebSearch]

llm_with_tools =  llm.bind_tools(tools)

def llm_node(state: AgentState) -> AgentState:

    messages = state['messages']

    result = llm_with_tools.invoke(messages)

    return {"messages": result}


def should_continue(state: AgentState) -> str:

    last_messages = state['messages'][-1]

    if last_messages.tool_calls:
        return "continue"
    else:
        return "end"

tool_node = ToolNode(tools)
graph =  StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "tool",
        "end": END
    }
)

graph.add_edge("tool", "llm")

checkpointer = InMemorySaver()

app = graph.compile(checkpointer=checkpointer)


stream_input = {"messages": [("user", "what current situation of tehren?")]}

config = {"configurable" : {"thread_id": "yogi1"}}

while True:
    result =  app.invoke(stream_input, config=config)

    if "__interrupt__" not in result:
        print(result['messages'][-1].content)
        break
    
    interrupt_payload = result["__interrupt__"][0].value

    print(f"\nApproval for web searh : {interrupt_payload}")

    answer = input("[y/n] Allow Websearch? ")

    stream_input = Command(resume  = answer)
 