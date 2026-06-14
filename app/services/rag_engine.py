import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry import context

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
        # Use temperature=0 for maximum factual consistency
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        # 1. Access the database
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        
        # 2. Perform global search
        docs = vector_store.similarity_search(user_question, k=10)
        context = "\n---\n".join([d.page_content for d in docs])
        
        # 3. Define the STRICT grounding prompt
        # This replaces the previous basic system_msg string
        system_prompt = """
You are a strict documentation-based assistant. 
Follow these instructions precisely:
1. ANSWER ONLY using the provided "Context" below.
2. If the answer is NOT present in the context, your ONLY allowed response is: "I do not have enough information to answer this based on the provided documents."
3. ABSOLUTELY NO external knowledge is permitted. Even if you know the answer, do not use it.

Context: 
{context}
"""
        # Create the template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        # 4. Generate the final answer
        # We inject the context and question into the strict template
        formatted_prompt = prompt_template.format(context=context, question=user_question)
        print(f"DEBUG: Context length: {len(context)}")
        if len(context) < 50:
            print("DEBUG: WARNING: Context is too small! Search might be failing.")
        final_answer = llm.invoke(formatted_prompt).content
        
        # Returns the format expected by your eval scripts
        return {
            "answer": final_answer, 
            "retrieved_context": [{"text": d.page_content, "source": d.metadata.get('file_name')} for d in docs]
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": "Engine Error: " + str(e), "retrieved_context": []}