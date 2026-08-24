from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
print("Hello World")

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Give the information about the tech company {name}",
    input_variables=["name"]
)

chain=prompt | model | StrOutputParser()

result = chain.invoke({"name":"LiveRamp"})

print(result)

