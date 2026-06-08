import streamlit as st
import requests

# Set up page configurations
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Secure RAG Pipeline Chat")
st.caption("Ask questions about your company policy documents in real-time.")

# FIXED: Removed the /api/v1 prefix to match the FastAPI route exactly
API_URL = "http://127.0.0.1:8000/query"

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
            # Send the request to your local FastAPI server including a session_id
            payload = {
                "question": user_question,
                "session_id": "streamlit_user_session"
            }
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer field found in response.")
                context = data.get("retrieved_context", [])
                standalone_q = data.get("standalone_query", user_question) # Fallback to raw question
                
                # Render the answer string
                response_placeholder.markdown(answer)
                
                # Advanced System Diagnostics Panel
                with st.expander("⚙️ System Diagnostics (Behind the Scenes)"):
                    st.info(f"**Contextualized Search Query:** *\"{standalone_q}\"*")
                    
                    if context:
                        st.write("**Retrieved Document Context Chunks:**")
                        for idx, chunk in enumerate(context):
                            st.write(f"📁 *Chunk {idx+1}:* {chunk}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Handle structured backend errors gracefully
                error_msg = f"Backend Error ({response.status_code}): Could not retrieve answer."
                response_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            response_placeholder.error("Could not connect to the FastAPI backend. Is your server running?")
        except Exception as e:
            response_placeholder.error(f"An unexpected error occurred: {str(e)}")