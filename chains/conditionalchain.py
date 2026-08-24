from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.7-flash')

class Review(BaseModel):
    feedback:Literal["positive","negative"]=Field(description="give the sentimennt about the feedback")

pdparser=PydanticOutputParser(pydantic_object=Review)

prompt1=PromptTemplate(
    template="Give me the sentimental analysis of the feedback as positive or negative {feedback},{format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction':pdparser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="give me a response according to the positive feedback{feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="give me a response according to the negative feedback {feedback}",
    input_variables=["feedback"]
)

parser=StrOutputParser()

chain=prompt1 | model |pdparser

conditional_chain=RunnableBranch(
    (lambda x:x.feedback=="positive",prompt2 | model | parser),
    (lambda x:x.feedback=="negative",prompt3 | model | parser),
    RunnableLambda(lambda x:"could not find the sentiment")
)

parallel=chain | conditional_chain
result= parallel.invoke({'feedback':"The first glimpse of S.S. Rajamouli's Varanasi starring Mahesh Babu delivers a world-class, breathtaking visual spectacle"})

print(result)

parallel.get_graph().print_ascii()
