import os
import sys
import json
from dotenv import load_dotenv

# 1. Force the load of .env from the current script's directory
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# Ensure the root directory is in the path
sys.path.append(basedir)

# Now it should be able to see the GROQ_API_KEY
from app.services.rag_engine import query_rag_system


def run_evaluation():
    # 1. Load the Test Suite
    test_file = 'qa_suite/test_cases.json'
    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    print(f"\n--- Starting Evaluation: {len(test_cases)} Test Cases ---\n")

    passed_count = 0
    
    for case in test_cases:
        print(f"Q: {case['question']}")
        
        # 2. Run the RAG Pipeline
        # We assume query_rag_system takes (question, session_id)
        # We use 'eval_session' so it doesn't clutter your real chat history
        response = query_rag_system(case['question'], session_id="eval_session")
        
        # 3. Handle response format (assuming it returns a dict or string)
        # Adjust 'response.get("answer")' if your function returns just a string
        actual_answer = response if isinstance(response, str) else response.get("answer", "")
        
        # 4. Check logic (Case-insensitive keyword/content matching)
        passed = case['expected_answer'].lower() in actual_answer.lower()
        
        status = "✅ PASS" if passed else "❌ FAIL"
        if passed: passed_count += 1
        
        print(f"Result: {status}")
        print(f"Expected: {case['expected_answer']}")
        print(f"Got:      {actual_answer[:100]}...") # Printing first 100 chars
        print("-" * 50)

    print(f"\n--- Evaluation Complete: {passed_count}/{len(test_cases)} Passed ---\n")

if __name__ == "__main__":
    run_evaluation()