import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Dynamically locate the project root
# Assuming this file is at: .../RAG/app/services/rag_engine.py
# We go up 2 levels from here to reach the RAG/ root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Define paths relative to the project root
# This works on your machine AND on any cloud server/GitHub environment
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
print(f"DEBUG: Enforced Absolute DB_DIR: {DB_DIR}")

# 3. Create the directories if they don't exist (prevents errors on new systems)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Print for your debugging verification
print(f"DEBUG: Data Directory initialized at: {DATA_DIR}")
print(f"DEBUG: Database Directory initialized at: {DB_DIR}")

local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
sessions_chat_history = {}

def initialize_rag_system():
    print(f"DEBUG: Searching for data in: {DATA_DIR}") # <--- ADD THIS
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found at {DATA_DIR}")
        os.makedirs(DATA_DIR)
        return

    # 2. Reset Database for a clean start
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
    try:
        vector_db.delete_collection()
        print("DEBUG: Existing collection deleted.")
    except Exception as e:
        print(f"DEBUG: No collection to delete or error: {e}")
    
    # 3. Re-initialize fresh
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)

    # 4. Load Docs with explicit logging
    raw_docs = []
    for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
        loader = DirectoryLoader(DATA_DIR, glob=glob, loader_cls=loader_cls)
        try:
            docs = loader.load()
            print(f"DEBUG: Loader {loader_cls.__name__} found {len(docs)} documents.")
            for doc in docs:
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get('source', 'Unknown'))
            raw_docs.extend(docs)
        except Exception as e:
            print(f"Load Error: {e}")

    if not raw_docs:
        print("DEBUG: No documents found in data folder. Indexing aborted.")
        return

    # 5. Robust Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(raw_docs)
    
    # 6. Add to Vector DB and force sync
    vector_db.add_documents(final_chunks)
    
    # Force persistence for some versions of Chroma
    if hasattr(vector_db, 'persist'):
        vector_db.persist()
        
    print(f"Database successfully indexed: {len(final_chunks)} chunks ready.")

def query_rag_system(user_question: str, session_id: str = "default_user"):
    try:
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        
        # 1. Access the database globally
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        
        # 2. Perform a GLOBAL search across ALL files (k=10 to get diverse chunks)
        # We removed the LLM Router and the 'filter' entirely.
        print("DEBUG: Performing global similarity search...")
        docs = vector_store.similarity_search(user_question, k=10)
        
        print(f"DEBUG: Retrieved {len(docs)} chunks from across all documents.")
        for i, d in enumerate(docs):
            print(f"DEBUG: Chunk {i} from {d.metadata.get('file_name')}: {d.page_content[:60]}...")
        
        # 3. Compile context from all retrieved chunks
        context = "\n---\n".join([d.page_content for d in docs])
        
        # 4. Final Answer Generation
        # The LLM now sees chunks from both the SANS paper and the ML book at the same time
        system_msg = "You are a specialized assistant. Use the provided Context to synthesize an answer. If the answer is not present, say so."
        final_answer = llm.invoke(f"System: {system_msg}\n\nContext:\n{context}\n\nQuestion: {user_question}").content
        
        return {
            "answer": final_answer, 
            "retrieved_context": [{"text": d.page_content, "source": d.metadata.get('file_name')} for d in docs]
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to help debug
        return {"answer": "Engine Error: " + str(e), "retrieved_context": []}