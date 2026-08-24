from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import Prompt_template
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name="summary",description="A brief summary of the product review"),
    ResponseSchema(name="specifications",description="Specifications related to the product"),
    ResponseSchema(name="cost",description="The cost of the product in USD, if available")
    ]

