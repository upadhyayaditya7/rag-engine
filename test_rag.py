import sys, os
sys.path.append(os.getcwd())
from app.services.rag_engine import query_rag_system

res = query_rag_system("What is a loss function?")
print(f"DEBUG: Retrieved {len(res['retrieved_context'])} chunks")
print(f"DEBUG: Answer: {res['answer']}")