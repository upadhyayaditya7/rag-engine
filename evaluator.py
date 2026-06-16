import os
import json
import time
from dotenv import load_dotenv
from app.core.logger import ResultLogger
from app.core.validator import LLMValidator
from app.services.rag_engine import RAGEngine

# Paths
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
DATA_DIR = os.path.join(basedir, "app", "data")
DB_DIR = os.path.join(basedir, "chroma_db")

# Initialize
engine = RAGEngine(DATA_DIR, DB_DIR)
engine.initialize()
logger = ResultLogger(log_file="qa_audit_report.json")
validator = LLMValidator(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model=os.getenv("EVAL_MODEL", "llama-3.3-70b-versatile")
)

def run_evaluation():
    print(f"DEBUG: Checking collection state...")
    collection = engine.vector_store._collection # Access the underlying chroma collection
    print(f"Total documents in database: {collection.count()}")

    # Sample a query to see if embeddings are working
    results = collection.query(query_texts=["sustainable development"], n_results=1)
    print(f"Sample retrieval results: {results['documents']}")
    test_file = os.path.join(basedir, 'qa_suite/test_cases.json')
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    print(f"\n--- Starting Evaluation: {len(test_cases)} Test Cases ---\n")
    
    for i, case in enumerate(test_cases, 1):
        question = case.get('question', 'N/A')
        expected = case.get('expected_answer', '')
        category = case.get('category')
        is_negative = case.get('is_negative', False)
        
        # 1. Search Logic: Unrestricted search for negative tests
        filter_dict = None if is_negative else {"category": category}
        
        print(f"[{i}] Testing: {question[:50]}...")
        
        # 2. Run Query (Modified to force clean retrieval)
        response = engine.query(question, filter_dict=filter_dict)
        actual = response.get("answer", "") if isinstance(response, dict) else str(response)
        metadata = response.get("metadata", []) if isinstance(response, dict) else []
        source_names = list(set([m.get('file_name', 'Unknown') for m in metadata]))
        
        # DEBUG: Check if database is actually working
        if not metadata:
            print(f"!!! DEBUG: Query returned zero chunks. Check RAGEngine/ChromaDB state.")
        
        # 3. Logic: Strict Grounding Safety Guard
        has_source = len(metadata) > 0
        if not has_source and not is_negative:
            passed = False
            result = {"score": 0, "rationale": "CRITICAL: No context retrieved for standard query."}
        
        # 4. Logic: Negative Test Verification
        elif is_negative:
            keywords = ["does not", "cannot", "no information", "not mentioned"]
            passed = any(k in actual.lower() for k in keywords)
            result = {"score": 10 if passed else 0, "rationale": "Negative inference check."}
            
        else:
            # Standard LLM-as-a-Judge
            result = validator.evaluate(question, expected, actual)
            passed = result.get('score', 0) >= 8
        
        # Final Logging
        is_valid = passed and (has_source or is_negative)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        
        logger.log(question, expected, actual, is_valid, source_names)
        
        print(f"Result:   {status} | Score: {result.get('score')}/10")
        print(f"Rationale: {result.get('rationale')}")
        print(f"Sources:  {', '.join(source_names)}")
        print("-" * 60)
        time.sleep(2)

if __name__ == "__main__":
    run_evaluation()