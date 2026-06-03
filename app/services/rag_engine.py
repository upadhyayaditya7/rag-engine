import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_DIR = "./chroma_db"

# This reads your text into small vectors locally using almost zero RAM
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Pull key securely from your operating system environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def ingest_documents(file_path: str):
    """Splits your policy document and creates the local database index."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_text(text)

    # UPDATED 'embedding' TO 'embedding_function' HERE TOO:
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding_function=local_embeddings,
        persist_directory=DB_DIR
    )
    print("Successfully processed documents into ChromaDB!")

def query_rag_system(user_query: str) -> dict:
    """Searches local facts, then lets the cloud AI generate the response."""
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=local_embeddings)
    
    # 1. Retrieve local context chunks
    retrieved_docs = vector_store.similarity_search(user_query, k=2)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. Strict system instructions to prevent lying
    system_prompt = (
        f"You are a strict corporate assistant. Answer the user question using ONLY the provided context. "
        f"If the answer cannot be found in the context, reply exactly with: 'I cannot find that in the documents.'\n\n"
        f"Context:\n{context_text}"
    )

    # 3. Generate: Calls Groq's blazing fast cloud model
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    full_prompt = f"{system_prompt}\n\nUser Question: {user_query}\nAnswer:"
    ai_response = llm.invoke(full_prompt)

    return {
        "answer": ai_response.content,
        "retrieved_context": [doc.page_content for doc in retrieved_docs]
    }

if __name__ == "__main__":
    ingest_documents("./data/company_policy.txt")