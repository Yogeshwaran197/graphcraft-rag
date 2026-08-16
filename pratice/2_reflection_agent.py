from langgraph.graph import state
from langgraph.constants import END
from langchain_groq import ChatGroq
from langgraph.graph import MessageGraph
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


def generator_node(state):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | llm

    response = chain.invoke({"messages": state})

    return response

def reflector_node(messages):


    prompt = ChatPromptTemplate.from_messages([
        ("system",  "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain =  prompt | llm

    response = chain.invoke({"messages": messages})

    return [HumanMessage(content=response.content)]


def should_continue(state) -> str:

    if (len(state) < 6):
        return "continue"
    
    return "End"


graph = MessageGraph()

graph.add_node("Generator", generator_node)
graph.add_node("Reflector", reflector_node)

graph.set_entry_point("Generator")

graph.add_conditional_edges(
    "Generator",
    should_continue,
    {
        "continue" : "Reflector",
        "End" : END
    }
)

graph.add_edge("Reflector", "Generator")

app = graph.compile()

print(app.get_graph().draw_mermaid())


response = app.invoke(HumanMessage(content="give me an tweet about iran vs usa war and how others affected by  that"))


for msg in response:
    print(msg.content)



