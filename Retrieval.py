from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


persistent_directory = "db/chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})

retriever = db.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 20, "k": 5})

query = "When was Google launched and who are the founders?"

relevant_docs = retriever.invoke(query)
print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

