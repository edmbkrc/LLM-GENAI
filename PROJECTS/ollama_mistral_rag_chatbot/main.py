import streamlit as st
from data_loader import add_to_chroma, load_documents, split_documents
from query_data import query_rag

st.title("RAG Chatbot with File Upload")

uploaded_files = st.file_uploader("Upload PDF or DOC files", type=['pdf', 'doc', 'docx'], accept_multiple_files=True)

if uploaded_files:
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    st.success("Files processed and added to the database.")

user_query = st.text_input("Type your question here:")

if user_query:
    answer = query_rag(user_query)
    st.write("Answer:", answer)
