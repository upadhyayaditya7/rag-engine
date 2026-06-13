import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Setup file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Initialize embedding model (cache it globally)
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

sessions_chat_history = {}

def initialize_rag_system():
    """Optimized ingestion with two-pass chunking and safe DB management."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return

    # Initialize Client
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
    
    # Safely clear old data without deleting the folder
    try:
        vector_db.delete_collection()
        # Re-init after delete
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
    except:
        pass

    # Load Docs
    raw_docs = []
    for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
        loader = DirectoryLoader(DATA_DIR, glob=glob, loader_cls=loader_cls)
        try:
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get('source', 'Unknown'))
            raw_docs.extend(docs)
        except Exception as e:
            print(f"Load Error: {e}")

    if not raw_docs: return

    # OPTIMIZATION: Two-pass chunking
    # 1. Structural Split (Fast): Breaks into large, manageable blocks
    struct_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    struct_chunks = struct_splitter.split_documents(raw_docs)

    # 2. Semantic Split (Refined): Only runs on 1500-char blocks
    sem_splitter = SemanticChunker(
        embeddings=local_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95 
    )
    final_chunks = sem_splitter.split_documents(struct_chunks)
    
    # 3. Batch Add
    vector_db.add_documents(final_chunks)
    print(f"Database indexed: {len(final_chunks)} chunks ready.")

def query_rag_system(user_question: str, session_id: str = "default_user"):
    try:
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        history = sessions_chat_history.setdefault(session_id, [])

        # 1. Contextualize
        search_query = user_question
        if history:
            formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-2:]])
            search_query = llm.invoke(f"History:\n{formatted_history}\n\nQuestion: {user_question}\n\nStandalone Question:").content.strip()

        # 2. Librarian (Dynamic Discovery)
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        unique_files = list(set(m.get('file_name') for m in vector_store.get()['metadatas'] if m.get('file_name')))
        
        choice_prompt = f"Available: {', '.join(unique_files)}\nTarget: {search_query}\nWhich file is relevant? Return ONLY the filename."
        chosen_file = llm.invoke(choice_prompt).content.strip()

        # 3. Retrieve
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
        docs = retriever.invoke(search_query)
        
        # Filter context
        if chosen_file in unique_files:
            docs = [d for d in docs if d.metadata.get('file_name') == chosen_file]
            
        context = "\n---\n".join([d.page_content for d in docs])
        
        # 4. Answer
        final_answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {user_question}").content
        history.extend([{"role": "user", "content": user_question}, {"role": "assistant", "content": final_answer}])
        
        return {"answer": final_answer, "retrieved_context": [{"text": d.page_content, "source": d.metadata.get('file_name')} for d in docs]}
    except Exception as e:
        print(f"RAG Error: {e}")
        return {"answer": "Engine Error.", "retrieved_context": []}