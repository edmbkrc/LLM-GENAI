import streamlit as st
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Başlık
st.title("RAG Chatbot Uygulaması (Streamlit ile)")

# Dosya Yükleme
uploaded_files = st.file_uploader("Bir veya daha fazla PDF dosyası yükleyin", type=["pdf"], accept_multiple_files=True)

# Config ve LLM Ayarları
local_llm = "meditron-7b.Q4_K_M.gguf"
config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="llama",
    lib="avx2",
    **config
)

st.write("LLM Başlatıldı...")

# Prompt Şablonu
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"

client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
retriever = db.as_retriever(search_kwargs={"k": 1})

# Yüklenen dosyaları işleme ve metinleri veritabanına ekleme
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Qdrant veritabanına ekleme
    Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name="vector_db"
    )
    st.success("Dosyalar veritabanına eklendi ve indekslendi!")

# Soru Girişi
query = st.text_input("Sorunuzu buraya girin:")

# Soru İşleme ve Yanıt Alma
if query:
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs, 
        verbose=True
    )
    
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    
    # Yanıtları Gösterme
    st.write("Yanıt:", answer)
    st.write("Kaynak Belge İçeriği:", source_document)
    st.write("Belge Adı:", doc)
