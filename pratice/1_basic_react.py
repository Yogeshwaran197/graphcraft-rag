from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

search_tool = TavilySearchResults()

@tool
def wiki_search(query: str) -> str:
    """use this tool to fetch real time data and use this as web browsing tool"""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.invoke(query)

@tool
def get_current_data() -> str:
    """this is to get current date and time"""
    current_time = datetime.datetime.now()
    return str(current_time)

tools = [search_tool, wiki_search, get_current_data]

agent = create_react_agent(model=llm, tools=tools)

response = agent.invoke({"messages": [("human", "situation between iran and us and how many days passed after war started?")]})
print(response['messages'][-1].content)