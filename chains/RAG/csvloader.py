from langchain_community.document_loaders import CSVLoader

obj=CSVLoader(r"C:\Users\Tanush\Desktop\langchain models\chains\RAG\ttd.csv")

doc=obj.load()
print(len(doc))

for x in doc:
    print(x.page_content)
    print(x.metadata)

