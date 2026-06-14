import sys
import os

# Force Python to look in the current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now import the components
from rag_engine import initialize_rag_system, query_rag_system

def run_test():
    print("--- Starting Hybrid Retrieval Test ---")
    
    # 1. Initialize
    initialize_rag_system()
    
    # 2. Run a specific test query
    test_question = "What is the primary objective of this project?" 
    
    result = query_rag_system(test_question)
    
    print("\n--- Test Results ---")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    run_test()