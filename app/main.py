import os
import shutil
import gc
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import engine functions
from app.services.rag_engine import query_rag_system, initialize_rag_system, sessions_chat_history

app = FastAPI(title="Memory-Aware Company RAG API")

# --- CENTRALIZED PATH CONFIGURATION ---
# This file is in 'app/', so BASE_DIR is the parent 'RAG/' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "app", "data") # Pointing explicitly to app/data
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure directories exist at startup
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print(f"--- System Initialization ---")
    print(f"Data Path: {DATA_DIR}")
    print(f"DB Path: {DB_DIR}")
    initialize_rag_system()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_user"

@app.get("/")
def home():
    return {"status": "Online", "data_path": DATA_DIR}

@app.post("/query")
def handle_query(request: QueryRequest):
    result = query_rag_system(request.question, session_id=request.session_id)
    if isinstance(result, str) and "Error" in result:
        raise HTTPException(status_code=500, detail=result)
    return result

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save file directly into the synced app/data directory
        file_path = os.path.join(DATA_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"\n[SUCCESS] Saved file to: {file_path}")
        
        # Force GC to release file handles before re-indexing
        gc.collect()
        initialize_rag_system()
        
        return {"status": "success", "message": f"Uploaded and indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/api/reset-database")
def reset_database():
    try:
        gc.collect()
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR)
        initialize_rag_system()
        return {"status": "success", "message": "Database reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")