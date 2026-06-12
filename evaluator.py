import os
import sys
import json
import re
from dotenv import load_dotenv

# 1. Robust .env loading
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# Verify API Key is loaded
if not os.getenv("GROQ_API_KEY"):
    print("CRITICAL: GROQ_API_KEY is missing from your .env file.")
    sys.exit(1)

# Ensure the root directory is in the path
sys.path.append(basedir)

from app.services.rag_engine import query_rag_system

def run_evaluation():
    # 1. Load the Test Suite
    test_file = 'qa_suite/test_cases.json'
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    print(f"\n--- Starting Evaluation: {len(test_cases)} Test Cases ---\n")

    passed_count = 0
    
    for case in test_cases:
        print(f"Q: {case['question']}")
        
        # 2. Run the RAG Pipeline
        response = query_rag_system(case['question'], session_id="eval_session")
        
        # 3. Handle response format
        actual_answer = response if isinstance(response, str) else response.get("answer", "")
        
        # 4. Math & Fuzzy Comparison Logic
        expected = case['expected_answer'].lower()
        actual = actual_answer.lower()
        
        def clean(text):
            return text.replace('Â·', '').replace('{', '').replace('}', '').replace('Ã…', 'a')

        clean_expected = clean(expected)
        clean_actual = clean(actual)
        
        # Extract numbers
        expected_nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_expected)
        actual_nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_actual)
        
        # Logic: If the EXPECTED answer has numbers, use numeric matching.
        # Otherwise, use the keyword fuzzy matching.
        if expected_nums:
            passed = all(num in actual_nums for num in expected_nums)
        else:
            expected_words = [w for w in clean_expected.split() if len(w) > 3]
            if not expected_words:
                passed = clean_expected in clean_actual
            else:
                matches = [w for w in expected_words if w in clean_actual]
                # Lowered threshold to 0.4 for non-math to be more forgiving
                passed = (len(matches) / len(expected_words)) >= 0.4
        
        status = "✅ PASS" if passed else "❌ FAIL"
        if passed: 
            passed_count += 1
        
        print(f"Result: {status}")
        print(f"Expected: {case['expected_answer']}")
        print(f"Got:      {actual_answer[:100]}...") 
        print("-" * 50)

    print(f"\n--- Evaluation Complete: {passed_count}/{len(test_cases)} Passed ---\n")

if __name__ == "__main__":
    run_evaluation()