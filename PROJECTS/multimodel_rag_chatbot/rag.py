import streamlit as st
import os
import base64
import uuid
from langchain import PromptTemplate
from langchain.llms import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from PIL import Image
import io

# Başlık
st.title("Multi-Modal RAG Chatbot Uygulaması")

# Config ve LLM Ayarları
local_llm = "gpt-4o-mini"
config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'temperature': 0.1,
    'top_p': 0.9,
}

# LLM Başlatma
llm = ChatOpenAI(model=local_llm, max_tokens=config['max_new_tokens'])
st.write("LLM Modeli Başlatıldı...")

# Vektör Veritabanı ve Saklama
vectorstore = Chroma(
    embedding_function=SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings"),
    persist_directory="vectorstore"
)

# Dosya Yükleme
uploaded_files = st.file_uploader("PDF dosyalarını yükleyin", type="pdf", accept_multiple_files=True)

# Yüklenen dosyaları işleme
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        documents.extend(loader.load())
    
    # Metinleri Parçalama
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Vektör Veritabanına Ekleme
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    for i, text in enumerate(texts):
        document = Document(page_content=text.page_content, metadata={"doc_id": doc_ids[i]})
        vectorstore.add_document(document)

    st.success("Dosyalar işlendi ve vektör veritabanına eklendi.")

# Soru Girişi
query = st.text_input("Sorunuzu girin:")
if query:
    # Sorguya Göre Yanıt Oluşturma
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.retrieve(query, k=4)
    
    # Metin ve Görüntüleri Ayırma
    def split_image_text_types(docs):
        b64 = []
        text = []
        for doc in docs:
            try:
                base64.b64decode(doc.page_content)
                b64.append(doc.page_content)
            except Exception:
                text.append(doc.page_content)
        return {"images": b64, "texts": text}
    
    docs_by_type = split_image_text_types(relevant_docs)
    
    # Sorgu için Prompt Hazırlama
    prompt_content = "\n".join(docs_by_type["texts"])
    prompt = f"Context: {prompt_content}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(prompt)
    
    st.write("Yanıt:", response)

    # Görüntüleri Gösterme
    for image_b64 in docs_by_type["images"]:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="İlgili Görüntü", use_column_width=True)

# Özetleme Fonksiyonları
def summarize_text(text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = llm.invoke(prompt)
    return response

def summarize_table(table_element):
    prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
    response = llm.invoke(prompt)
    return response

def summarize_image(encoded_image):
    prompt = f"Describe the contents of this image."
    messages = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]
    response = llm.invoke(messages)
    return response

# Görüntüleri Kodlayarak Özetleme
for image_b64 in docs_by_type["images"]:
    summary = summarize_image(image_b64)
    st.write("Görüntü Özeti:", summary)

# Ekstra: Görüntü Boyutlandırma Fonksiyonu
def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
