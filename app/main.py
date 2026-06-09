import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import gc  # Added for garbage collection to unlock files on Windows

# Imported initialize_rag_system and sessions_chat_history from your engine service
from app.services.rag_engine import query_rag_system, initialize_rag_system, sessions_chat_history

app = FastAPI(title="Memory-Aware Company RAG API")

# --- FIXED: DEFINE DATA_DIR SAFELY AT THE TOP ---
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
    result = query_rag_system(request.question, session_id=request.session_id)
    
    if isinstance(result, str) and "Error" in result:
        raise HTTPException(status_code=500, detail=result)
        
    return result

@app.post("/api/clear-history")
def clear_history():
    global sessions_chat_history
    sessions_chat_history.clear()
    return {"status": "success", "message": "Chat history cleared"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receives a file from the frontend, saves it to data/, and re-indexes the RAG system safely."""
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
        
        # Force Python to instantly clear any lingering background DB client reads 
        gc.collect()
        
        # Trigger the automatic ingestion pipeline to process the fresh document instantly!
        initialize_rag_system()
        
        return {"status": "success", "message": f"Successfully uploaded and indexed {file.filename}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
@app.post("/api/reset-database")
def reset_database():
    """Wipes the ChromaDB memory cleanly without triggering Windows folder permission locks."""
    try:
        import gc
        from app.services.rag_engine import initialize_rag_system
        
        print("\n[RESET INITIATED] Cleaning database memory safely...")
        
        # 1. Force Python to clear memory buffers
        gc.collect()
        
        # 2. Check if the directory exists
        DB_DIR = os.path.join(BASE_DIR, "chroma_db")
        
        if os.path.exists(DB_DIR):
            try:
                # Loop inside the folder and delete the internal data files instead of the whole folder
                for filename in os.listdir(DB_DIR):
                    file_path = os.path.join(DB_DIR, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print("[RESET SUCCESS] Internal ChromaDB contents cleared cleanly.")
            except PermissionError:
                print("[LOCK DETECTED] File lock active, falling back to clean system re-index.")
                # If Windows still blocks it, we proceed to re-initialization which overwrites stale data
        
        # 3. Re-run initialization to sync up with whatever files are currently in the data/ folder
        initialize_rag_system()
        
        return {"status": "success", "message": "Vector database completely reset and re-indexed fresh!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")