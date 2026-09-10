from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

doc1=Document(
    page_content="""The Grand Canyon is one of the most visited natural wonders in the world. Photosynthesis is the process by which green plants convert sunlight into energy. Millions of tourists travel to see it every year. The rocks date back millions of years."""
)

doc2=Document(
    page_content="""In medieval Europe, castles were built primarily for defense. The chlorophyll in plant cells captures sunlight during photosynthesis. Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
)

doc3=Document(page_content="""Basketball was invented by Dr. James Naismith in the late 19th century. It was originally played with a soccer ball and peach baskets. NBA is now a global league.""")

doc4=Document(page_content="""The history of cinema began in the late 1800s. Silent films were the earliest form. Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells. """)


doc=[doc1, doc2, doc3, doc4]

vector_store=Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    ),
    persist_directory="my_chroma_db",
    collection_name="sample"
)

vector_store.add_documents(doc)

llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)
compressor=LLMChainExtractor.from_llm(llm)

contextual_compression_retriever = ContextualCompressionRetriever(
    base_retriever=vector_store.as_retriever(
        search_kwargs={"k": 1}
    ),
    base_compressor=compressor
)

query=" how basket ball will be played ?"

result=contextual_compression_retriever.invoke(query)

for i in range(len(result)):
    print(result[i].page_content)






