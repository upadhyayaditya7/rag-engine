import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Imported initialize_rag_system and sessions_chat_history from your engine service
from app.services.rag_engine import query_rag_system, initialize_rag_system, sessions_chat_history
from fastapi import UploadFile, File
import shutil

app = FastAPI(title="Memory-Aware Company RAG API")

# --- FIXED: DEFINE DATA_DIR SAFELY AT THE TOP ---
# Since main.py is inside the 'app' directory, its parent is the true project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

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

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receives a file from the frontend, saves it to data/, and re-indexes the RAG system."""
    try:
        # Ensure the data directory exists before trying to write to it
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        # Define the target absolute path inside the data directory
        file_path = os.path.join(DATA_DIR, file.filename)
        
        # Save the uploaded file block-by-block to the disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"\n[SUCCESS] Uploaded successfully: New file saved at {file_path}")
        
        # Trigger the automatic ingestion pipeline to process the fresh document instantly!
        initialize_rag_system()
        
        return {"status": "success", "message": f"Successfully uploaded and indexed {file.filename}"}
    except Exception as e:
        # If anything else breaks, this will force it to print to your terminal log
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
@app.post("/api/reset-database")
def reset_database():
    """Wipes the ChromaDB collection completely and re-indexes only what is currently in the data/ folder."""
    try:
        from app.services.rag_engine import initialize_rag_system
        import shutil
        
        # Define the path to your chroma db directory
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.dirname(CURRENT_DIR)
        DB_DIR = os.path.join(BASE_DIR, "chroma_db")
        
        # 1. Delete the physical database folder if it exists
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            print("[RESET] Physical ChromaDB folder deleted.")
            
        # 2. Re-run the initialization to build a completely blank slate and read the new data folder
        initialize_rag_system()
        
        return {"status": "success", "message": "Vector database completely reset and re-indexed fresh!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")