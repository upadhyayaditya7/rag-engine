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

def initialize_rag_system():
    """Reads the document, applies advanced chunking, and indexes it into ChromaDB."""
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found. Please create it first.")
        return

    print("--- Starting Document Processing ---")
    # 1. Load the raw text document
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    raw_documents = loader.load()
    
    # 2. Apply Smart Chunking using RecursiveCharacterTextSplitter
    # chunk_size: Max characters per block. chunk_overlap: Shared characters between adjacent blocks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Successfully split document into {len(chunks)} individual, overlapping chunks.")

    # 3. Store the chunked vectors into ChromaDB (overwriting old unchunked data)
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=local_embeddings, 
        persist_directory=DB_DIR
    )
    print("Database indexing complete. Smart chunks successfully stored!")

def query_rag_system(user_question: str):
    """Searches the smart chunks and generates a safe, factual response via Groq."""
    if not os.environ.get("GROQ_API_KEY"):
        return "RAG Engine Error: The GROQ_API_KEY environment variable is missing."

    try:
        # Connect to existing database index
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        
        # Search for the top 3 closest relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(user_question)
        
        context_list = [doc.page_content for doc in relevant_docs]
        combined_context = "\n---\n".join(context_list)
        
        # Initialize Llama 3.1 model via Groq
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        
        # System prompt ensuring no hallucinations
        system_prompt = (
            "You are a secure company assistant. Answer the user's question using ONLY the provided context below.\n"
            "If the answer is not explicitly found within the context, respond exactly with: 'I cannot find that in the documents.'\n"
            "Do not make up facts under any circumstances.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {user_question}"
        )
        
        response = llm.invoke(system_prompt)
        
        # Format return payload for the backend API
        return {
            "answer": response.content,
            "retrieved_context": context_list
        }
        
    except Exception as e:
        return f"RAG Engine Error: {str(e)}"

# Self-contained execution for indexing
if __name__ == "__main__":
    initialize_rag_system()