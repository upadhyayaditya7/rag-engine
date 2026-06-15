import os
import sys
import json
import time
from dotenv import load_dotenv
from app.core.logger import ResultLogger
from app.core.validator import LLMValidator
from app.services.rag_engine import RAGEngine
from openai import OpenAI

# Paths
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
DATA_DIR = os.path.join(basedir, "app", "data")
DB_DIR = os.path.join(basedir, "chroma_db")

# Single Initialization
engine = RAGEngine(DATA_DIR, DB_DIR)
engine.initialize()

logger = ResultLogger(log_file="qa_audit_report.json")

# Initialize Judge (Groq-compatible)
validator = LLMValidator(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model=os.getenv("EVAL_MODEL", "llama-3.3-70b-versatile")
)


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
        expected = case.get('expected_answer', '')
        
        print(f"[{i}] Q: {question}")
        
        # Run RAG
        response = engine.query(question)
        actual = response.get("answer", "") if isinstance(response, dict) else str(response)
        metadata = response.get("metadata", []) if isinstance(response, dict) else []
        
        # Validate
        has_source = len(metadata) > 0
        source_names = list(set([m.get('file_name', 'Unknown') for m in metadata]))
        
        # LLM-as-a-Judge
        result = validator.evaluate(question, expected, actual)
        passed = result.get('score', 0) >= 8
        
        # Final Pass/Fail logic
        is_valid = passed and has_source
        status = "✅ PASS" if is_valid else "❌ FAIL"
        
        # Log and Print
        logger.log(question, expected, actual, is_valid, source_names)
        if is_valid: passed_count += 1
        
        print(f"Result:   {status} (Score: {result.get('score')}/10)")
        print(f"Rationale: {result.get('rationale')}")
        print(f"Sources:  {', '.join(source_names)}")
        print("-" * 60)
        time.sleep(3)

    print(f"\n--- Evaluation Complete: {passed_count}/{len(test_cases)} Passed ---\n")

if __name__ == "__main__":
    run_evaluation()