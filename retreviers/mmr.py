from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

doc1=Document(page_content="""Superman: The symbol of hope and the ultimate protector of Earth.""")
doc2=Document(page_content="""Captain America: The embodiment of courage, honor, and unwavering leadership.""")
doc3=Document(page_content="""Thor: The mighty God of Thunder who fights for his people and the universe.""")
doc4=Document(page_content="""Iron Man: The genius inventor who turns technology and determination into heroism.""")
doc5=Document(page_content="""Spider-Man: The friendly neighborhood hero with incredible web-slinging abilities.""")

vectore_store=Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    persist_directory="my_chroma_db",
    collection_name="text1"
)
doc=[doc1,doc2,doc3,doc4,doc5]
vectore_store.add_documents(doc)

retriever=vectore_store.as_retriever(search_type="mmr",search_kwargs={"k":1,"lambda_unit":0.5})

query="Tell me about powerful superheroes"

result=retriever.invoke(query)

for i in range(len(result)):
    print(result[i].page_content)

# Note: type of retrievers