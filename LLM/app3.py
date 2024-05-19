import streamlit as st 
from dotenv import load_dotenv 
load_dotenv()
import os 
import google.generativeai as genai 

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

st.title('Analyze Image with Google Gemini')
model=genai.GenerativeModel('gemini-pro-vision')
from PIL import Image
resim=st.file_uploader("Bir resim sec", type=(['jpg','jpeg','png']))
if resim is not None:
    img=Image.open(resim)
    st.image(img)
    #response=model.generate_content(img)
    #st.write(response.text)
soru=st.text_input('Sorunuzu giriniz...!')
if st.button('Sorunuzu gönderin...'):
    response=model.generate_content([soru,img], stream=True)
    response.resolve()
    st.write(response.text)