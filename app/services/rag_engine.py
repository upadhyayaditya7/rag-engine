import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# --- UPGRADED IMPORT: Swapped out RecursiveCharacterTextSplitter for SemanticChunker ---
from langchain_experimental.text_splitter import SemanticChunker
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
    """Scans the data directory, processes files, and indexes them into ChromaDB using Semantic Chunking."""
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

    # --- ADVANCED PRODUCTION UPGRADE: SEMANTIC CHUNKING ---
    print("Initializing Semantic Chunker (evaluating sentence embedding variances)...")
    
    # We pass your local_embeddings model directly into the chunker. 
    # breakpoint_threshold_type="percentile" calculates distances between sentences 
    # and splits them if the difference falls into the top 5% of variance (95th percentile).
    text_splitter = SemanticChunker(
        embeddings=local_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.95
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Successfully split all documents into {len(chunks)} semantic chunks.")

    # Overwrite/Create the local ChromaDB deployment
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=local_embeddings, 
        persist_directory=DB_DIR
    )
    print("Database indexing complete. Semantic boundaries successfully stored!")

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

        # --- STEP 2: VECTOR SEARCH WITH MAXIMAL MARGINAL REVERANCE (MMR) ---
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        
        # Swapped standard similarity search for 'mmr' to eliminate redundant duplicates
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,             # Feed the LLM the top 5 diverse context blocks
                "fetch_k": 20,      # Initially evaluate 20 candidates from ChromaDB
                "lambda_mult": 0.5  # 0.5 forces a strong balance between relevance and diversity
            }
        )
        
        relevant_docs = retriever.invoke(search_query)
        
        # Extract plain text context to pass to the system prompt
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

        # --- STEP 4: EXTRACT METADATA ALONG WITH TEXT (FIXES FRONTEND ERROR) ---
        formatted_context = []
        for doc in relevant_docs:
            # Safely extract file name and page number from LangChain document metadata
            source_path = doc.metadata.get('source', 'Unknown File')
            source_file = os.path.basename(source_path)
            
            # LangChain pages are 0-indexed; add 1 so it matches human-readable pages (e.g. Page 1, Page 2)
            page_num = doc.metadata.get('page', 0) + 1  
            
            formatted_context.append({
                "text": doc.page_content,
                "source": f"{source_file} (Page {page_num})"
            })

        # --- UPDATE THE TRACKED SESSION HISTORY ---
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": final_answer})

        # Return structured metadata payload to sync with the Streamlit frontend UI
        return {
            "answer": final_answer,
            "retrieved_context": formatted_context,  
            "standalone_query": search_query
        }
        
    except Exception as e:
        import traceback
        print("\nCRITICAL RAG ENGINE ERROR !")
        traceback.print_exc()
        print("!!!!!\n")
        return {"answer": f"RAG Engine Error: {str(e)}", "retrieved_context": []}

if __name__ == "__main__":
    initialize_rag_system()