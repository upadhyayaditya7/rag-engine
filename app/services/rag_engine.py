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
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.db_dir = os.path.join(os.getcwd(), db_dir)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        self.vector_store = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        self.bm25_index = None
        self.tokenized_corpus = None
        self.chunks = []

    def initialize(self):
        raw_docs = []
        for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
            loader = DirectoryLoader(self.data_dir, glob=glob, loader_cls=loader_cls)
            docs = loader.load()
            
            # Explicitly attach filename to metadata
            for doc in docs:
                doc.metadata['file_name'] = os.path.basename(doc.metadata.get('source', 'unknown'))
            raw_docs.extend(docs)
        
        # Ensure splitter is defined clearly at the function level
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(raw_docs)
        
        # Indexing
        self.vector_store.add_documents(self.chunks)
        
        # Save corpus for BM25
        self.tokenized_corpus = [c.page_content.lower().split() for c in self.chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        print(f"DEBUG: System Initialized with {len(self.chunks)} chunks.")
        print(f"DEBUG: Sample chunk metadata: {self.chunks[0].metadata}")

    def query(self, user_question: str):
        # 1. Similarity Search (Returns list of Document objects)
        vector_docs = self.vector_store.similarity_search(user_question, k=5)
        
        # 2. BM25 Hybrid Retrieval
        bm25_docs = []
        if self.bm25_index:
            query_tokens = user_question.lower().split()
            # Get the top N tokenized matches
            top_matches = self.bm25_index.get_top_n(query_tokens, self.tokenized_corpus, n=5)
            
            # Map tokenized matches back to original Document objects
            # We use a set of strings to quickly identify original chunks
            for tokens in top_matches:
                for doc in self.chunks:
                    if doc.page_content.lower().split() == tokens:
                        bm25_docs.append(doc)
                        break

        # 3. Consolidate and Deduplicate
        # Use a dictionary to keep unique documents by page_content
        combined_docs = {d.page_content: d for d in (vector_docs + bm25_docs)}.values()
        
        # 4. Prepare Context for LLM
        combined_context = "\n---\n".join([d.page_content for d in combined_docs])
        
        # 5. Extract metadata (Now guaranteed to be from Document objects)
        metadata_list = [d.metadata for d in combined_docs]
        
        # 6. Final Answer Construction
        prompt = ChatPromptTemplate.from_template("""
        You are an expert technical analyst. Use ONLY the provided context:
        {context}
        Question: {question}
        """)
        
        answer = self.llm.invoke(prompt.format(context=combined_context, question=user_question)).content
        
        # Return answer with clean metadata
        return {"answer": answer, "metadata": metadata_list}