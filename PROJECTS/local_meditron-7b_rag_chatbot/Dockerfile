# Base image
FROM python:3.9-slim

# Çalışma dizinini oluşturun
WORKDIR /app

# Gerekli dosyaları kopyalayın
COPY . .

# Bağımlılıkları yükleyin
RUN pip install --no-cache-dir torch sentence_transformers transformers langchain fastapi uvicorn pypdf PyPDF2 jinja2 chroma qdrant-client ctransformers python-multipart aiofiles

# Streamlit portunu açın
EXPOSE 8501

# Streamlit uygulamasını başlatın
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
