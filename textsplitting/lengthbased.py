from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

obj = TextLoader(
    r"C:\Users\Tanush\Desktop\langchain models\textsplitting\aieng.txt",
    encoding="UTF-8"
)

res = obj.load()

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separator=""
)

split_res = splitter.split_documents(res)

print(split_res[5].page_content)