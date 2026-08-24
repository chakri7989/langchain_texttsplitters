from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

url="https://ttdevasthanams.ap.gov.in/home/dashboard"
obj=WebBaseLoader(url)

doc=obj.load()

test=doc[0].page_content

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt=PromptTemplate(
    template="Where i can book special entry darshan tickets in following {text}",
    input_variables=["text"]
)
parser=StrOutputParser()

chain=prompt | model | parser

res=chain.invoke({'text':test})

print(res)