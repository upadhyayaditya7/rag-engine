import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.rag_engine import query_rag_system

app = FastAPI(title="Memory-Aware Company RAG API")

# Enable CORS so your Streamlit frontend can communicate seamlessly with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added session_id parameter to the request model with a default fallback
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_user"

@app.get("/")
def home():
    return {"status": "Online", "engine": "RAG with Persistent Memory"}

@app.post("/query")
def handle_query(request: QueryRequest):
    # Pass both the question and the session_id to the upgraded RAG service
    result = query_rag_system(request.question, session_id=request.session_id)
    
    if isinstance(result, str) and "Error" in result:
        raise HTTPException(status_code=500, detail=result)
        
    return result

@app.post("/api/clear-history")
def clear_history():
    global sessions_chat_history
    sessions_chat_history = {}
    return {"status": "success", "message": "Chat history cleared"}