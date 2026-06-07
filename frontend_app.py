import streamlit as st
import requests

# Set up page configurations
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Secure RAG Pipeline Chat")
st.caption("Ask questions about your company policy documents in real-time.")

# Define the backend FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/api/v1/query"

# Initialize message history in streamlit session state so chat doesn't vanish on refresh
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from the chat box
if user_question := st.chat_input("What would you like to know?"):
    
    # Display user message immediately in the UI
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Display a loading spinner while communicating with our FastAPI backend
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # Send the request to your local FastAPI server
            payload = {"question": user_question}
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                # Extract the result string returned by your rag_engine
                answer = response.json()
                response_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Handle structured backend errors gracefully
                error_msg = f"Backend Error ({response.status_code}): Could not retrieve answer."
                response_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            response_placeholder.error("Could not connect to the FastAPI backend. Is your server running?")
        except Exception as e:
            response_placeholder.error(f"An unexpected error occurred: {str(e)}")