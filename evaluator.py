import os
import sys
import json
import re
import time
from dotenv import load_dotenv

# Set up paths
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
sys.path.append(basedir)

# Import engine
try:
    from app.services.rag_engine import query_rag_system
except ImportError:
    print("Error: Could not import query_rag_system.")
    sys.exit(1)

def clean_text(text):
    """Normalize text for comparison."""
    text = text.replace('Â·', '').replace('{', '').replace('}', '').replace('Ã…', 'a')
    return re.sub(r'\s+', ' ', text).strip().lower()

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
        response = query_rag_system(question, session_id="eval_session")
        
        # Safely extract answer
        actual_raw = response.get("answer", "") if isinstance(response, dict) else str(response)
        actual = clean_text(actual_raw)
        
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
                passed = (len(matches) / len(expected_words)) >= 0.6
        
        status = "✅ PASS" if passed else "❌ FAIL"
        if passed: passed_count += 1
        
        print(f"Result:   {status}")
        print(f"Expected: {expected}")
        print(f"Got:      {actual[:100]}...")
        print("-" * 60)
        
        # --- CRITICAL: RATE LIMIT PROTECTION ---
        # A 3-second sleep ensures we stay within the Groq free tier limits.
        time.sleep(3) 

    print(f"\n--- Evaluation Complete: {passed_count}/{len(test_cases)} Passed ---\n")

if __name__ == "__main__":
    run_evaluation()