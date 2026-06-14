import json, os
from app.services.rag_engine import query_rag_system

questions = [
    "What are the three intelligence outputs?",
    "What is a loss function?",
    "How can threat intelligence inform risk prediction?"
]

results = []
for q in questions:
    print(f"Generating answer for: {q}")
    response = query_rag_system(q)
    results.append({
        "question": q, 
        "actual_answer": response["answer"],
        "context": str(response["retrieved_context"]) 
    })

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("eval_results.json has been successfully updated.")