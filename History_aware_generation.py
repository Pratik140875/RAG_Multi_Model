from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

persistant_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
db = Chroma(persist_directory=persistant_directory, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
retriever = db.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 20, "k": 5})
    
model = ChatOpenAI(model="gpt-4o", temperature=0.2)

chat_history = []


def start_conversation():
    print("Welcome to the History-Aware RAG System!"
          "\nYou can ask questions about the documents in the vector store, and I'll provide answers based on the retrieved context."
          "\nType 'exit'/'quit' to end the conversation.")
    
    while True:
        user_query = input("\nYour Question: ")

        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        ask_question(user_query)
        
def ask_question(query):

    print(f"\nProcessing your question: {query}")
    if chat_history:
        messages = [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")] + chat_history + [HumanMessage(content=f"new question: {query}")]
        result = model.invoke(messages)
        search_query = result.content.strip()
        print(f"Reformulated Query: {search_query}")
    else:        
        search_query = query
        
    relevant_docs = retriever.invoke(search_query)     
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
    Comb_input = f"""Based on th following documents, answer the question: {search_query}
    Documents:  
    {chr(10).join([doc.page_content for doc in relevant_docs])}
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."""
    message  = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents. Always use only the information from the documents to answer the question."),
        HumanMessage(content=Comb_input)
    ]   
    response = model.invoke(message)
    print(f"Answer:\n{response.content}")
    chat_history.append(HumanMessage(content=search_query))
    chat_history.append(AIMessage(content=response.content))
        
    return response.content
    
if __name__ == "__main__":
   
    start_conversation()