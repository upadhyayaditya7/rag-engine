import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry import context
from rank_bm25 import BM25Okapi

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
bm25_index = None
final_chunks_global = None
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
                # Inject page tracking without breaking existing index compatibility
                page_val = doc.metadata.get('page', 0)
                doc.metadata["page_number"] = page_val + 1 if isinstance(page_val, int) else 1
                doc.metadata["source_ref"] = f"{doc.metadata['file_name']} (Page {doc.metadata['page_number']})"
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

    global bm25_index, final_chunks_global
    final_chunks_global = final_chunks
    # Tokenize content for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in final_chunks]
    try:
        bm25_index = BM25Okapi(tokenized_corpus)
    except Exception as e:
        print(f"DEBUG: BM25 failed to initialize: {e}")
    bm25_index = None # Explicitly set to None
    print(f"DEBUG: BM25 index created with {len(final_chunks)} chunks.")
        
    print(f"Database successfully indexed: {len(final_chunks)} chunks ready.")

def query_rag_system(user_question: str, session_id: str = "default_user"):
    # Ensure we can access the global BM25 indexer
    global bm25_index, final_chunks_global
    
    try:
        # Use temperature=0 for maximum factual consistency
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        # 1. Access the database
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
        
        # 2. Perform Hybrid Search
        # A: Vector Search (Semantic)
        vector_docs = vector_store.similarity_search(user_question, k=5)
        
        # B: BM25 Search (Keyword)
        tokenized_query = user_question.lower().split()
        if bm25_index:
            bm25_docs_content = bm25_index.get_top_n(tokenized_query, [d.page_content for d in final_chunks_global], n=5)
        else:
            # Fallback: Just use vector search results if BM25 is unavailable
            bm25_docs_content = []

        combined_context = "\n---\n".join([d.page_content for d in vector_docs] + bm25_docs_content)
        
        # 3. Define the STRICT grounding prompt
        system_prompt = """
You are an expert technical analyst. 
Follow these instructions precisely:
1. USE ONLY the provided "Context" to answer the user's question.
2. If the exact answer is NOT present, you are allowed to synthesize a response 
   based on the logic and facts provided in the context. 
3. If the context is completely unrelated to the question, your ONLY allowed 
   response is: "I do not have enough information to answer this based on the provided documents."
4. Prioritize clarity and actionable insights. If the context describes a process 
   or concept, explain it using the details found in the text.
5. ABSOLUTELY NO external knowledge is permitted.

Context: 
{context}
"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        # 4. Generate the final answer
        formatted_prompt = prompt_template.format(context=combined_context, question=user_question)
        
        print(f"DEBUG: Context length (Hybrid): {len(combined_context)}")
        final_answer = llm.invoke(formatted_prompt).content
        
        # Return format maintained for your eval scripts
        return {
            "answer": final_answer, 
            "metadata": [d.metadata for d in vector_docs]
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": "Engine Error: " + str(e), "retrieved_context": []}
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The question to ask.")
    args = parser.parse_args()
    
    initialize_rag_system()
    response = query_rag_system(args.query)
    print(f"\nResult: {response['answer']}")