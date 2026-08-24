from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()

model1=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
model2=ChatGoogleGenerativeAI(model='gemini-3.7-flash')

prompt1=PromptTemplate(
    template="How RCb is better than {team} in IPL 2026",
    input_variables=["team"]
)

prompt2=PromptTemplate(
    template="Analyze and give the drawbacks made by {team} in IPL 2026",
    input_variables=["team"]
)
parser=StrOutputParser()

parallel=RunnableParallel({
    'comp':RunnableSequence(prompt1,model1,parser),
    'analyze':RunnableSequence(prompt2,model2,parser)
})

res=parallel.invoke({'team':'MI'})

print(res)
