from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=1,
    lang="en"
)

query = "Who is Joseph Vijay?"

docs = retriever.invoke(query)

for i in range(len(docs)):
    print(docs[i].page_content)
