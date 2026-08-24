from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
x=PromptTemplate(
    template="give me a 5 facts about hero {movie} \n {x}",
    input_variables=[{"movie"}],
    partial_variables={'x': parser.get_format_instructions()}
)

chat=x|model|parser
res=chat.invoke({"movie":"Mahesh Babu"})
print(res)