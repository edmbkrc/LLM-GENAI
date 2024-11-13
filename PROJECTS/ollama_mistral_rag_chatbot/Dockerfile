# Python'un hafif bir sürümünü kullanarak temel bir imaj oluşturuyoruz
FROM python:3.9-slim

# Sistem bağımlılıklarını yükleyin
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini belirtiyoruz
WORKDIR /app

# Gereksinim dosyasını kopyalıyoruz ve gerekli Python paketlerini kuruyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını imaj içine kopyalıyoruz
COPY . .

# Streamlit uygulaması için portu açıyoruz
EXPOSE 8501

# Uygulamayı başlatıyoruz
CMD ["streamlit", "run", "main.py"]
