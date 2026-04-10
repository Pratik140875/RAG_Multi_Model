import os
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader

load_dotenv()


# def load_documents(docs_path):
#     """Load all text files from the docs directory"""
#     print(f"Loading documents from {docs_path}...")

#     if not os.path.exists(docs_path):
#         raise FileNotFoundError(f"Directory {docs_path} does not exist. Please create it and add some .txt files.")

    
#     loader = DirectoryLoader(
#         path = docs_path, 
#         glob="**/*.txt", 
#         loader_cls=TextLoader,
#         loader_kwargs={'encoding': 'utf-8'}, # Tell the text loader to use UTF-8 encoding 
#         show_progress=True)
    
#     documents = loader.load()
    
#     if len(documents) == 0:
#         raise ValueError(f"No .txt files found in {docs_path}. Please add some text files to the directory.")
    
#     # for i,doc in enumerate(documents):
#     #     print(f"Document {i+1}: {doc.metadata['source']}")
#     #     print(f"length: {len(doc.page_content)} characters)")
#     #     print(f"Content preview: {doc.page_content[:200]}...\n")

#     return documents

# def split_documents(documents, chunk_size=1000, chunk_overlap=200):
#     """Split documents into smaller chunks"""
#     print(f"Splitting documents into chunks of {chunk_size} characters with {chunk_overlap} characters overlap...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = text_splitter.split_documents(documents)
#     # for i,chunk in enumerate(chunks[:5]):  # Print the first 5 chunks for preview
#     #     print(f"Chunk {i+1}: length {len(chunk.page_content)} characters")
#     #     print(f"Content preview: {chunk.page_content[:200]}...\n")
    
#     # if len(chunks) > 5:
#     #     print(f"...and {len(chunks)-5} more chunks.")
#     return chunks

def create_vector_store(chunks, persistent_directory="db/chroma_db"):
    """Create a Chroma vector store from the document chunks"""
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
#     vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persistent_directory, collection_metadata={"hnsw:space": "cosine"})
#     collection = vector_store._collection
#     # 1. Check the total count
#     count = collection.count()
#     print(f"Total items in collection: {count}")

# # 2. Peek at the data (get IDs and Metadata)
# # This will show us if the same 'source' appears multiple times
#     data = collection.get(include=['metadatas', 'documents'])

#     print("\n--- Collection Metadata Summary ---")
#     sources = [m.get('source') for m in data['metadatas']]
#     unique_sources = set(sources)

#     for source in unique_sources:
#         print(f"Source: {source} | Occurrences: {sources.count(source)}")

#     # 3. Check for exact text duplicates
#     print("\n--- Content Sample ---")
    if data['documents']:
        print(f"First document snippet: {data['documents'][0][:100]}...")
    # print(f"Vector store created at location: {persistent_directory}")
    # return vector_store
    query = "When was Google launched and who are the founders?"

    retriever = vector_store.as_retriever(search_type ="mmr",search_kwargs={"fetch_k":20,"k": 5})

    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "k": 5,
    #         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    #     }
    # )

    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    # Display results
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

def main():
    docs_path = "Doc"
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":    main()

