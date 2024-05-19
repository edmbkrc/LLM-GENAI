import streamlit as st 
from dotenv import load_dotenv 
import os 
import google.generativeai as genai 
from PIL import Image

# Load environment variables
load_dotenv()

# Configure generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Title
st.title('Google Gemini ile Beslenme ve Diyet Sohbeti')

# Load generative model
model = genai.GenerativeModel('gemini-pro-vision')

# Image upload
resim = st.file_uploader("Bir resim seçin", type=['jpg', 'jpeg', 'png'])

if resim is not None:
    # Display the uploaded image
    img = Image.open(resim)
    st.image(img, caption='Yüklenen Resim', use_column_width=True)

    # Text input for question
    soru = st.text_input('Sorunuzu girin...')

    # Button to generate content
    if st.button('İçerik Oluştur'):
        try:
            # Generate content based on image and question
            response = model.generate_content([soru, img], stream=True)
            response.resolve()
            
            # Filter response based on keywords
            beslenme_keywords = ['beslenme', 'diyet', 'gıda', 'meyve', 'sebze', 'sağlıklı beslenme', 'dengeli beslenme', 'vitamin', 'mineral', 'lif', 'protein', 'karbonhidrat', 'yağ', 'su', 'kalori', 'obezite', 'kilo kontrolü', 'metabolizma']
            diyetisyenlik_keywords = ['diyetisyen', 'beslenme uzmanı', 'sağlık uzmanı', 'beslenme danışmanı', 'diyet planı', 'kilo verme', 'kilo alma', 'sağlık koçluğu', 'beslenme danışmanlığı', 'beslenme uzmanlığı', 'beslenme programı', 'diyetisyenlik eğitimi', 'beslenme bilimi']
            
            filtered_response = [chunk.text for chunk in response if any(keyword in chunk.text.lower() for keyword in beslenme_keywords + diyetisyenlik_keywords)]
            
            if filtered_response:
                # Display filtered response
                st.subheader('Oluşturulan Yanıt')
                for text in filtered_response:
                    st.write(text)
            else:
                st.write("Üzgünüm, beslenme ve diyetisyenlikle ilgili uygun bilgi bulunamadı.")
            
            # Provide option for user feedback
            feedback = st.selectbox('Geri Bildirim:', ['Olumlu', 'Nötr', 'Olumsuz'])

            # Save feedback to a file or database
            # For demonstration, let's print the feedback
            st.write('Geri Bildirim:', feedback)
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
