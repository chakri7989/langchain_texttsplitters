from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=ChatGoogleGenerativeAI(model='gemini-3.6-flash')

model2=ChatGoogleGenerativeAI(model='gemini-3.7-flash')
prompt1=PromptTemplate(
    template="Give a notes on the topic {name}",
    input_variables=["name"]
)

prompt2=PromptTemplate(
    template="Give me a quiz on the {name}",
    input_variables=["name"]
)

prompt3=PromptTemplate(
    template="merege the both provided notes and quizz into a single doc {notes} and {quiz}",
    input_variables=["notes","quiz"]
)
parser=StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 |model2 |parser
})

merge_chain=prompt3 | model1 | parser

chain =  parallel_chain | merge_chain

matter="""## Normalization in DBMS

Normalization is a process of organizing data in a database to **reduce data redundancy and avoid data anomalies**. It involves dividing a large table into smaller related tables and establishing relationships between them. The main purpose of normalization is to improve data consistency, integrity, and efficient storage. It helps prevent three major types of anomalies: **insertion anomaly, update anomaly, and deletion anomaly**. Normalization is carried out using different normal forms such as **First Normal Form (1NF), Second Normal Form (2NF), Third Normal Form (3NF), Boyce-Codd Normal Form (BCNF), Fourth Normal Form (4NF), and Fifth Normal Form (5NF)**.

A relation is said to be in **First Normal Form (1NF)** when every attribute contains only atomic or single values and there are no repeating groups. **Second Normal Form (2NF)** is achieved when a relation is in 1NF and there is no partial dependency, meaning every non-key attribute must depend on the entire primary key. **Third Normal Form (3NF)** is achieved when the relation is in 2NF and there is no transitive dependency, meaning non-key attributes should not depend on other non-key attributes. **BCNF** is a stronger version of 3NF in which every determinant must be a super key. **4NF** deals with multivalued dependencies, while **5NF** deals with join dependencies and ensures that a relation cannot be further decomposed without losing information.

Thus, normalization helps in designing a **well-structured and efficient database** by minimizing duplicate data and maintaining data accuracy. However, excessive normalization can increase the number of tables and may require more joins during queries, so in practical database design, normalization is applied according to the requirements of the system.
"""

result=chain.invoke({'name':matter})

print(result)


chain.get_graph().print_ascii()
