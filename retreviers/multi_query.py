
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Python is a popular programming language known for simple, readable syntax."
)

doc2 = Document(
    page_content="JavaScript runs in web browsers and powers interactive websites."
)

doc3 = Document(
    page_content="Machine learning models require large amounts of training data to perform well."
)

doc4 = Document(
    page_content="Cloud computing platforms like AWS allow companies to scale applications without owning physical servers."
)

doc5 = Document(
    page_content="Cybersecurity practices like encryption protect sensitive data from hackers."
)

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    ),
    persist_directory="my_chroma_db",
    collection_name="sample2"
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store.add_documents(docs)

multiquery = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 1}
    ),
    llm=ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    )
)

query = "What should a beginner learn in tech?"

result = multiquery.invoke(query)

for doc in result:
    print(doc.page_content)

