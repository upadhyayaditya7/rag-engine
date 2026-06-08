import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Setup file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
FILE_PATH = os.path.join(DATA_DIR, "company_policy.txt")

# Initialize the embedding model globally
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- NEW MEMORY STORAGE ---
# Global in-memory dictionary to hold message logs for different sessions
# Format: { "session_id": [{"role": "user/assistant", "content": "text"}] }
sessions_chat_history = {}

def initialize_rag_system():
    """Reads the document, applies advanced chunking, and indexes it into ChromaDB."""
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found. Please create it first.")
        return

    print("--- Starting Document Processing ---")
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    raw_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Successfully split document into {len(chunks)} individual, overlapping chunks.")

    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=local_embeddings, 
        persist_directory=DB_DIR
    )
    print("Database indexing complete. Smart chunks successfully stored!")

def query_rag_system(user_question: str, session_id: str = "default_user"):
    """Contextualizes follow-up questions using session history, searches smart chunks, and answers via Groq."""
    if not os.environ.get("GROQ_API_KEY"):
        return "RAG Engine Error: The GROQ_API_KEY environment variable is missing."

    try:
        # Initialize Llama 3.1 model via Groq
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

        # Initialize or fetch history list for this specific user session
        if session_id not in sessions_chat_history:
            sessions_chat_history[session_id] = []
        
        history = sessions_chat_history[session_id]

        # --- STEP 1: CONTEXTUALIZE THE QUESTION (THE BRAIN OF THE MEMORY) ---
        search_query = user_question
        if len(history) > 0:
            # Format the past history items cleanly for the model
            formatted_history = ""
            for msg in history[-4:]:  # Look at the last 4 messages to save tokens
                formatted_history += f"{msg['role'].upper()}: {msg['content']}\n"

            context_prompt = (
                f"Given the following chat history between a user and an assistant, and a new follow-up question, "
                f"rewrite the follow-up question into a standalone, complete question that can be understood on its own "
                f"without needing the chat history. Do not answer the question, just return the rewritten question text.\n\n"
                f"Chat History:\n{formatted_history}"
                f"Follow-up Question: {user_question}\n\n"
                f"Standalone Question:"
            )
            rewritten_response = llm.invoke(context_prompt)
            search_query = rewritten_response.content.strip()

        # --- STEP 2: VECTOR SEARCH USING THE REWRITTEN QUERY ---
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # We invoke retrieval using the standalone 'search_query', NOT the raw user_question
        relevant_docs = retriever.invoke(search_query)
        
        context_list = [doc.page_content for doc in relevant_docs]
        combined_context = "\n---\n".join(context_list)
        
        # --- STEP 3: GENERATE THE RAG RESPONSE WITH SYSTEM PROMPT ---
        system_prompt = (
            "You are a secure company assistant. Answer the user's question using ONLY the provided context below.\n"
            "If the answer is not explicitly found within the context, respond exactly with: 'I cannot find that in the documents.'\n"
            "Do not make up facts under any circumstances.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {user_question}"
        )
        
        response = llm.invoke(system_prompt)
        final_answer = response.content

        # --- STEP 4: UPDATE THE TRACKED SESSION HISTORY ---
        # --- STEP 4: UPDATE THE TRACKED SESSION HISTORY ---
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": final_answer})

        return {
            "answer": final_answer,
            "retrieved_context": context_list,
            "standalone_query": search_query  # <-- ADD THIS LINE
        }
        
    except Exception as e:
        return f"RAG Engine Error: {str(e)}"

if __name__ == "__main__":
    initialize_rag_system()