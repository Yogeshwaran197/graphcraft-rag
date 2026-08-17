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
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

#tools
@tool
def multilpy(num1, num2):
    """This node is to multiply two nums """

    return num1*num2

@tool
def search(query):
    """this tool is to fetch real time data by web serach"""

    search = TavilySearchResults()
    response = search.invoke(query)

    return response


tools = [multilpy, search]

llm_with_tools = llm.bind_tools(tools)

#nodes

def llm_node(state: AgentState) -> AgentState:

    messages = state['messages']

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


tool_map = {t.name : t for t in tools}

def tool_node(state: AgentState) -> AgentState:

    messages = state['messages'][-1]

    result = []

    for tool_call in messages.tool_calls:

        if tool_call is None:
            print(f"tool not found")

        if tool_call["name"] == "search":
            response = input("[y/n] continue web search? ")
            if response == 'n':
                raise ValueError("Web Search Discard")
    
        
            tool = tool_map[tool_call['name']] 
            observation = tool.invoke(tool_call['args'])

            result.append(ToolMessage(content=observation, tool_call_id=tool_call['id']))


    return {"messages": result}

def should_continue(state:AgentState) -> AgentState:

    last_messages = state["messages"][-1]

    if last_messages.tool_calls:
        return "continue"
    
    return "End"

graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "tool",
        "End": END
    }
)

graph.add_edge("tool", END)

app = graph.compile()


for s in app.stream({"messages": "what is current issues in github right now"}):
    print(list(s.values()))
    print("-" * 30)






    

    

    

