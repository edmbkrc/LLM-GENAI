import streamlit as st
from llama_rag import llama_rag_main
from gemma_rag import gemma_rag_main

# User credentials
USER_CREDENTIALS = {
    "a": "1",
    "user": "userpass456"
}

# Function to authenticate user
def authenticate_user():
    st.session_state.authenticated = False  # Default state
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.write(f"Welcome, {username}!")
            st.query_params["authenticated"] = "true"  # Update query params
        else:
            st.error("Invalid username or password.")
    return st.session_state.authenticated

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Login to Access RAG Applications")
    if authenticate_user():
        st.query_params["authenticated"] = "true"  # Ensure state persists
else:
    # RAG application options
    app_options = ["RAG Chatbot with Llama", "RAG Chatbot with Gemma"]

    # User selection
    selected_app = st.selectbox("Choose an RAG application:", app_options)

    # Logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.query_params.pop("authenticated", None)  # Remove authentication query param
        st.query_params.clear()  # Clear all query parameters

    # Run the selected application
    if selected_app == "RAG Chatbot with Llama":
        llama_rag_main()
    elif selected_app == "RAG Chatbot with Gemma":
        gemma_rag_main()
