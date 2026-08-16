from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, TypedDict
import datetime



load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

class Reflection(BaseModel):

    missing : str = Field(..., description="critique of what is missing")    
    superfluous: str = Field(description="Critique of what is superfluous")


class Answer(BaseModel):

    answer: str = Field(..., description="250 words of detailed response to answer")
    search_queries: List[str] = Field(..., description="search queries to improve the response further more")
    reflection: Reflection = Field(..., description="critique for the generated response")


llm_with_schema = llm.with_structured_output(Answer)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time = lambda: datetime.datetime.now().isoformat(),)


first_responder_prompt =  actor_prompt_template.partial(
    first_instruction = "250 words detailed answer for given query"
)

pydantic_parser = PydanticToolsParser(tools=[Answer])

responder_chain = first_responder_prompt | llm_with_schema

result =  responder_chain.invoke({
    "messages": [HumanMessage(content = "tell about how ai replace developers in future is that ai engineer role is safe or not")]
})

print(result.answer)
print(result.search_queries)
print(result.reflection)
