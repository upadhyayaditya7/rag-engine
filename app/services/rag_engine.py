import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

class RAGEngine:
    def __init__(self, data_dir, db_dir):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        self.vector_store = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        self.bm25_index = None
        self.chunks = None

    def initialize(self):
        # Your existing loading logic
        raw_docs = []
        for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
            loader = DirectoryLoader(self.data_dir, glob=glob, loader_cls=loader_cls)
            raw_docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(raw_docs)
        
        self.vector_store.add_documents(self.chunks)
        
        # FIX: BM25 initialization (Removed the line that was setting it to None)
        tokenized_corpus = [c.page_content.lower().split() for c in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"DEBUG: System Initialized with {len(self.chunks)} chunks.")

    def query(self, user_question: str):
        vector_docs = self.vector_store.similarity_search(user_question, k=5)
        
        bm25_content = []
        if self.bm25_index:
            bm25_content = self.bm25_index.get_top_n(user_question.lower().split(), 
                           [d.page_content for d in self.chunks], n=5)

        combined_context = "\n---\n".join([d.page_content for d in vector_docs] + bm25_content)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert technical analyst. Use ONLY the provided context:
        {context}
        Question: {question}
        """)
        
        answer = self.llm.invoke(prompt.format(context=combined_context, question=user_question)).content
        return {"answer": answer, "metadata": [d.metadata for d in vector_docs]}