import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Setup file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Initialize the embedding model
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Memory storage
sessions_chat_history = {}

def initialize_rag_system():
    """Processes files with semantic chunking and dynamic metadata tagging."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return

    # Clean existing database for a fresh index
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    
    loaders = {
        ".txt": DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}),
        ".pdf": DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    }
    
    raw_documents = []
    for ext, loader in loaders.items():
        try:
            loaded_docs = loader.load()
            for doc in loaded_docs:
                # Attach file_name to metadata for dynamic retrieval
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get('source', 'Unknown'))
            raw_documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading {ext}: {e}")

    if not raw_documents:
        return

    # High-density chunking for technical documents
    text_splitter = SemanticChunker(
        embeddings=local_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95 
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    Chroma.from_documents(documents=chunks, embedding=local_embeddings, persist_directory=DB_DIR)
    print("Database indexed with high-density semantic chunks.")

def query_rag_system(user_question: str, session_id: str = "default_user"):
    """Contextualizes, selects the correct file dynamically, and answers."""
    if not os.environ.get("GROQ_API_KEY"):
        return {"answer": "RAG Engine Error: API Key missing.", "retrieved_context": []}

    try:
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        history = sessions_chat_history.setdefault(session_id, [])

        # 1. Contextualize query
        search_query = user_question
        if history:
            formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-4:]])
            search_query = llm.invoke(f"History:\n{formatted_history}\n\nQuestion: {user_question}\n\nStandalone Question:").content.strip()

        # 2. Dynamic Metadata Discovery (The "Librarian")
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        all_metadata = vector_store.get(include=['metadatas'])
        unique_files = list(set(m.get('file_name') for m in all_metadata['metadatas'] if m.get('file_name')))

        choice_prompt = f"Available files: {', '.join(unique_files)}\nQuestion: {search_query}\nWhich file is most relevant? Return ONLY the filename. If none, return 'NONE'."
        chosen_file = llm.invoke(choice_prompt).content.strip()

        # 3. Vector Search with Dynamic Filter
        search_kwargs = {
            "k": 6,           # Increased to retrieve more context
            "fetch_k": 25,    # Larger candidate pool
            "lambda_mult": 0.2 # Lowered from 0.5: This makes retrieval LESS diverse and MORE focused on the query
        }
        if chosen_file in unique_files:
            search_kwargs["filter"] = {"file_name": chosen_file}

        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(search_query)
        
        combined_context = "\n---\n".join([doc.page_content for doc in relevant_docs])
        
        # 4. Generate Response
        system_prompt = (
            "You are a Senior Technical Researcher. You must provide comprehensive, long-form answers.\n"
            "If the question asks for a 'Why' or 'How', provide all details, reasons, and technical context found in the snippets.\n"
            "DO NOT shorten or summarize your response. If a technical comparison or list of steps exists, include every single item.\n"
            f"Context:\n{combined_context}\n\nQuestion: {user_question}"
        )
        final_answer = llm.invoke(system_prompt).content

        history.extend([{"role": "user", "content": user_question}, {"role": "assistant", "content": final_answer}])
        return {"answer": final_answer, "retrieved_context": [{"text": d.page_content, "source": d.metadata.get('file_name', 'Unknown')} for d in relevant_docs]}
        
    except Exception as e:
        print(f"RAG Error: {e}")
        return {"answer": "Error processing your request.", "retrieved_context": []}

if __name__ == "__main__":
    initialize_rag_system()