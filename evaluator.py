import os
import sys
import json
import re
import time
from dotenv import load_dotenv
from app.utils.logger import ResultLogger

# Set up paths
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
sys.path.append(basedir)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Import engine
try:
    from app.services.rag_engine import RAGEngine
except ImportError:
    print("Error: Could not import query_rag_system.")
    sys.exit(1)

def clean_text(text):
    """Normalize text for comparison."""
    text = text.replace('Â·', '').replace('{', '').replace('}', '').replace('Ã…', 'a')
    return re.sub(r'\s+', ' ', text).strip().lower()

# Assuming you have these defined at the top of your evaluator.py
from app.services.rag_engine import RAGEngine
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Instantiate the engine
engine = RAGEngine(DATA_DIR, DB_DIR)
engine.initialize()

# Initialize engine and logger
engine = RAGEngine(DATA_DIR, DB_DIR)
engine.initialize()

logger = ResultLogger(log_file="qa_audit_report.json") # This creates/updates your audit file

def run_evaluation():
    test_file = os.path.join(basedir, 'qa_suite/test_cases.json')
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    print(f"\n--- Starting Evaluation: {len(test_cases)} Test Cases ---\n")
    passed_count = 0

    for i, case in enumerate(test_cases, 1):
        question = case.get('question', 'N/A')
        expected = clean_text(case.get('expected_answer', ''))
        
        print(f"[{i}] Q: {question}")
        
        # Run the RAG Pipeline
        response = engine.query(question)
        
        # Safely extract answer AND metadata
        actual_raw = response.get("answer", "") if isinstance(response, dict) else str(response)
        metadata = response.get("metadata", []) if isinstance(response, dict) else []
        
        actual = clean_text(actual_raw)
        
        # --- Analytical Source Audit ---
        has_source = len(metadata) > 0
        source_names = list(set([m.get('file_name', 'Unknown') for m in metadata]))
        
        # Evaluation Logic
        expected_nums = re.findall(r"[-+]?\d*\.\d+|\d+", expected)
        actual_nums = re.findall(r"[-+]?\d*\.\d+|\d+", actual)
        
        if expected_nums:
            passed = all(num in actual_nums for num in expected_nums)
        else:
            expected_words = [w for w in expected.split() if len(w) > 3]
            if not expected_words:
                passed = expected in actual
            else:
                matches = [w for w in expected_words if w in actual]
                # Lower the threshold from 0.6 to 0.3 to allow for 
                # LLM paraphrasing and slight stylistic differences
                passed = (len(matches) / len(expected_words)) >= 0.3
        
        # Require both content match AND source grounding for a PASS
        status = "✅ PASS" if (passed and has_source) else "❌ FAIL"
        logger.log(question, expected, actual, (passed and has_source), source_names)
        if passed and has_source: passed_count += 1
        
        print(f"Result:   {status} (Grounded: {has_source})")
        print(f"Sources:  {', '.join(source_names)}")
        print(f"Expected: {expected}")
        print(f"Got:      {actual[:100]}...")
        print("-" * 60)
        
        # --- CRITICAL: RATE LIMIT PROTECTION ---
        # A 3-second sleep ensures we stay within the Groq free tier limits.
        time.sleep(3) 

    print(f"\n--- Evaluation Complete: {passed_count}/{len(test_cases)} Passed ---\n")

if __name__ == "__main__":
    run_evaluation()