from langchain_community.document_loaders import PyPDFLoader

obj = PyPDFLoader(r"C:\Users\Tanush\Desktop\langchain models\chains\RAG\AIE.pdf")
documents = obj.load()
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)