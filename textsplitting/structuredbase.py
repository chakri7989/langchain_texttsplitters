from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

obj = TextLoader(
    r"C:\Users\Tanush\Desktop\langchain models\textsplitting\aieng.txt",
    encoding="UTF-8"
)

res = obj.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

split_text=splitter.split_documents(res);
for x in split_text:
    print(x.page_content)
