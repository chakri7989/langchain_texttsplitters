from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.7-flash')

prompt1=PromptTemplate(
    template="Give the meaning of {name} in the one word from the upcoming of rajamouli",
    input_variables=["name"]
)
parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser)

res=chain.invoke({'name':'astreoid shambavi'})

print(res)

