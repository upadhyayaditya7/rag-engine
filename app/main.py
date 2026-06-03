from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.rag_engine import query_rag_system

# This exact line MUST be here and lowercase 'app'
app = FastAPI(title="Secure Cloud-Powered RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Your RAG pipeline API is running perfectly!"}

@app.post("/api/v1/query")
def run_query(payload: QueryRequest):
    """Takes a user question, runs it through the RAG engine, and returns the cloud AI answer."""
    try:
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        # Call your secure RAG engine
        result = query_rag_system(payload.question)
        return result
        
    except Exception as e:
        # This prints the actual error safely in your local VS Code terminal for you to debug
        print(f"RAG Engine Error: {e}")
        # This sends a clean, safe message back to the user without leaking system details
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your query.")