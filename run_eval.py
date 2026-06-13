import json
from app.services.rag_engine import query_rag_system

# Your JSON data
eval_data = [...] # (Paste your JSON list here)

def run_evaluation():
    results = []
    for item in eval_data:
        print(f"\n--- Testing: {item['category']} ---")
        print(f"Q: {item['question']}")
        
        # Call your RAG engine
        response = query_rag_system(item['question'])
        
        print(f"A: {response['answer']}")
        
        # Store for comparison
        results.append({
            "question": item['question'],
            "actual_answer": response['answer'],
            "expected_answer": item['expected_answer']
        })
    
    # Save the results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("\nEvaluation complete. Results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()