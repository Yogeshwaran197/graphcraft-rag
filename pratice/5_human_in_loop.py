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

            decision = interrupt({
                "action": tool_call['name'],
                "query": tool_call['args']
            })

            if decision != "y":
                result.append(ToolMessage(content="Search was rejected by the user. Do not retry unless asked again.",
                tool_call_id= tool_call['id']))
                continue

        tool = tool_map[tool_call['name']] 
        observation = tool.invoke(tool_call['args'])

        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call['id']))


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

graph.add_edge("tool", "llm")

checkpointer = InMemorySaver()

app = graph.compile(checkpointer=checkpointer)


stream_input = {"messages": [("user","what is current issues in github rn?")]}

config = {"configurable" : {"thread_id": "yogi-1"}}

while True:

    result =  app.invoke(stream_input, config = config)
    if "__interrupt__" not in result:
        print(result['messages'][-1].content)
        break
    
    interrupt_payload = result["__interrupt__"][0].value
    print(f"\n[Approval needed] {interrupt_payload}")
    answer = input("[y/n] approval? ")

    stream_input = Command(resume = answer)









    

    

    

