import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Setup
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Get all metadata
all_docs = vector_store.get(include=['metadatas'])
categories = set([m.get('category') for m in all_docs['metadatas'] if 'category' in m])

print(f"DEBUG: Unique categories found in DB: {categories}")