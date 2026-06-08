import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Setup file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Initialize the embedding model globally
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- MEMORY STORAGE ---
sessions_chat_history = {}

def initialize_rag_system():
    """Scans the data directory, processes all text and PDF files, and indexes them into ChromaDB."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found. Creating it now...")
        os.makedirs(DATA_DIR)
        return

    print("--- Starting Document Processing Pipeline ---")
    
    # Define directory loaders for both file types
    text_loader_kwargs = {'encoding': 'utf-8'}
    loaders = {
        ".txt": DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs),
        ".pdf": DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    }
    
    raw_documents = []
    for ext, loader in loaders.items():
        try:
            loaded_docs = loader.load()
            if loaded_docs:
                print(f"Found and loaded {len(loaded_docs)} document(s) matching {ext}")
                raw_documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading files with extension {ext}: {str(e)}")

    if not raw_documents:
        print("No .txt or .pdf files found inside the 'data/' directory. Skipping vectorization.")
        return

    # Advanced text splitting optimized for both prose and formatted layouts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,         # Slightly increased size to hold continuous sentences from PDFs
        chunk_overlap=100,       # Keeps semantic context intact over cut lines
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Successfully split all documents into {len(chunks)} individual chunks.")

    # Overwrite/Create the local ChromaDB deployment
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=local_embeddings, 
        persist_directory=DB_DIR
    )
    print("Database indexing complete. Multi-format documents successfully stored!")

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
            formatted_history = ""
            for msg in history[-4:]:
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
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": final_answer})

        return {
            "answer": final_answer,
            "retrieved_context": context_list,
            "standalone_query": search_query
        }
        
    except Exception as e:
        import traceback
        print("\n!!! CRITICAL RAG ENGINE ERROR !!!")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return {"answer": f"RAG Engine Error: {str(e)}", "retrieved_context": []}

if __name__ == "__main__":
    initialize_rag_system()