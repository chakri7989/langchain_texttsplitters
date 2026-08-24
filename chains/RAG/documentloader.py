from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

obj = TextLoader(
    r"C:\Users\Tanush\Desktop\langchain models\chains\RAG\varanasi.txt",
    encoding="UTF-8"
)

text = obj.load()

data = text[0].page_content

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = PromptTemplate(
    template="Predict the story of the movie based on the following information:\n\n{x}",
    input_variables=["x"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"x": data})

print(result)