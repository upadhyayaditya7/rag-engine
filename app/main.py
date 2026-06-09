import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Imported initialize_rag_system and sessions_chat_history from your engine service
from app.services.rag_engine import query_rag_system, initialize_rag_system, sessions_chat_history

app = FastAPI(title="Memory-Aware Company RAG API")

# Enable CORS so your Streamlit frontend can communicate seamlessly with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTOMATED STARTUP LIFECYCLE HOOK ---
# This forces FastAPI to scan files and index them into ChromaDB every time it boots
@app.on_event("startup")
async def startup_event():
    print("FastAPI Boot Sequence: Syncing Data Folders with ChromaDB...")
    initialize_rag_system()

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
    # Correctly targets the global dictionary imported from the engine file
    global sessions_chat_history
    sessions_chat_history.clear()
    return {"status": "success", "message": "Chat history cleared"}