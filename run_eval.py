import os
from dotenv import load_dotenv
load_dotenv()
import json
from app.services.rag_engine import query_rag_system

# 1. Define your dataset
EVAL_DATASET = [
    {
        "category": "Fact Checking", 
        "question": "What are the three intelligence outputs?", 
        "expected_answer": "Strategic, Operational, and Tactical intelligence."
    },
    {
        "category": "ML Theory", 
        "question": "What is a loss function?", 
        "expected_answer": "A loss function quantifies the discrepancy between the true label and the predicted label."
    },
    {
        "category": "Synthesis", 
        "question": "How can threat intelligence inform risk prediction?", 
        "expected_answer": "By providing prioritized threat data to help quantify risk uncertainty."
    }
]

def run_evaluation():
    results = []
    # 2. Use the dataset defined above
    for item in EVAL_DATASET:
        print(f"\n--- Testing: {item['category']} ---")
        print(f"Q: {item['question']}")
        
        # 3. Call your RAG engine (ensure session_id is provided if needed)
        response = query_rag_system(item['question'], session_id="eval_test")
        
        print(f"A: {response['answer']}")
        
        # 4. Store the results
        results.append({
            "question": item['question'],
            "actual_answer": response['answer'],
            "expected_answer": item.get('expected_answer', 'N/A')
        })
    
    # Save the results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("\nEvaluation complete. Results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()