from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

temp1=PromptTemplate(
    template="give me a report on the topic {review}",
    input_variables=["review"],
)

temp2=PromptTemplate(
    template="give a summarize on the topic {text}",
    input_variables=["text"],
)

parser=StrOutputParser()

chain=temp1 | model | parser |temp2 | model | parser

response=chain.invoke({"review":"langchain"})

print(response)


