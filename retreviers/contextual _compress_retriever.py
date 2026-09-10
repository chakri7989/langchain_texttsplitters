from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

doc1=Document(
    page_content="""Python was created by Guido van Rossum.
It was first released in 1991.
It is widely used for AI and web development."""
)

doc2=Document(
    page_content="""The Eiffel Tower is located in Paris.
It was completed in 1889.
It was built for the World's Fair."""
)

doc3=Document(page_content="""Mount Everest is the tallest mountain on Earth.
It is located in the Himalayas.
Its peak is 8,849 meters above sea level.""")

doc4=Document(page_content="""The human heart pumps blood through the body.
It has four chambers.
An average heart beats about 100,000 times a day.""")

doc5=Document(page_content="""The Great Wall of China was built for defense.
Construction started over 2,000 years ago.
It stretches thousands of kilometers across China.""")

doc=[doc1, doc2, doc3, doc4, doc5]

vector_store=Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    ),
    persist_directory="my_chroma_db",
    collection_name="sample"
)

vector_store.add_documents(doc)

contextual_compression_retriever = ContextualCompressionRetriever(
    base_compressor=ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    ),
    base_retriever=vector_store.as_retriever(
        search_kwargs={"k": 1}
    )
)

query="what was python created ?"

result=contextual_compression_retriever.invoke(query)

for i in range(len(result)):
    print(result[i].page_content)






