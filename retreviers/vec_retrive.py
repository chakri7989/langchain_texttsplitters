from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

doc1=Document(page_content="""Mahesh Babu
A superstar known for his charming screen presence and powerful performances.
His movies have earned him a huge fan following across India.""")

doc2=Document(page_content="""Dulquer Salmaan
A versatile actor admired for his natural acting and stylish personality.
He has impressed audiences across Malayalam, Tamil, Telugu, and Hindi cinema.""")

doc3=Document(page_content="""Nani
Known as the “Natural Star,” Nani is loved for his realistic and emotional performances.
His unique choice of stories has made him one of Telugu cinema’s popular actors.""")

doc=[doc1,doc2,doc3]

vector_store=Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    persist_directory='my_chroma_db',
    collection_name='sample1'
)
vector_store.add_documents(doc)

retriever=vector_store.as_retriever(search_kwargs={"k":2})

query="Who is versatile actor and charming screen presence?"

result=retriever.invoke(query)

for i in range(len(result)):
    print(result[i].page_content)