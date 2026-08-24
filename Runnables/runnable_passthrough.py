from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough

load_dotenv()

model1=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
model2=ChatGoogleGenerativeAI(model='gemini-3.7-flash')

prompt1=PromptTemplate(
    template="Generate a simple joke in {lang}",
    input_variables=["lang"]
)

parser=StrOutputParser()

joke_gen=RunnableSequence(prompt1,model1,parser)

Prompt2=prompt1=PromptTemplate(
    template="Explain the joke in simple {lang}",
    input_variables=["lang"]
)

final_res=RunnableParallel({
    'joke':RunnablePassthrough(),
    'exp':RunnableSequence(prompt1,model2,parser)
})

chain=RunnableSequence(joke_gen,final_res)

res=chain.invoke({'lang':'telugu'})

print(res['joke'])

