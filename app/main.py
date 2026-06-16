import os
import shutil
import gc
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.rag_engine import RAGEngine

app = FastAPI(title="Memory-Aware Company RAG API")

# --- CENTRALIZED PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Instantiate the engine
engine = RAGEngine(DATA_DIR, DB_DIR)

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
    engine.initialize()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_user"

@app.get("/")
def home():
    return {"status": "Online", "data_path": DATA_DIR}

@app.post("/query")
def handle_query(request: QueryRequest):
    # Using the engine instance correctly
    result = engine.query(request.question)
    if isinstance(result, str) and "Error" in result:
        raise HTTPException(status_code=500, detail=result)
    return result

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        gc.collect()
        engine.initialize()
        
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
        engine.initialize()
        return {"status": "success", "message": "Database reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")